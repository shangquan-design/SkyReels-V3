import argparse
import logging
import os
import random
import time
import subprocess
from pathlib import Path
import traceback
import imageio
import torch
import torch.distributed as dist
import wandb
import yaml
import numpy as np

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.pipelines import TalkingAvatarPipeline
from skyreels_v3.utils.avatar_preprocess import preprocess_audio

# NEW:
from skyreels_v3.utils.multispeaker_helper import (
    prepare_multispeaker_assets,
    build_pipeline_input_data,
)


# -------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - skyreels_v3_batch - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],
)


# -------------------- helpers --------------------
def init_dist_if_needed():
    """Optional: if launched with torchrun, initialize NCCL."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def barrier_if_dist(is_dist: bool):
    if is_dist:
        dist.barrier()


def destroy_dist(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def pick_device(local_rank: int):
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def parse_resolution_bucket(hw):
    """
    hw: [H, W] or tuple
    Map to pipeline bucket string: 480P/720P.
    """
    if hw is None:
        return "720P"
    h = int(hw[0])
    if h <= 480:
        return "480P"
    else:
        return "720P"


def safe_stem(path_str: str):
    """Make a stable ID from file name (without suffix)."""
    p = Path(path_str)
    return p.stem.replace(" ", "_")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def ffmpeg_mux(video_path: str, audio_path: str, out_path: str):
    """
    Mux video (no audio) + audio into one mp4.
    Uses ffmpeg; expects ffmpeg exists in PATH.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-shortest",
        out_path,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


def load_yaml_samples(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty YAML: {yaml_path}")

    if isinstance(data, dict):
        for key in ["samples", "data", "items", "benchmark"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"YAML must be a list (or dict containing a list). Got: {type(data)}")

    samples = []
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            raise ValueError(
                f"Sample #{i} must be a list of length>=5: [image_path, seed, [H,W], audio_or_media_path, prompt]. Got: {item}"
            )
        img_path = str(item[0])
        seed = int(item[1]) if item[1] is not None else None
        hw = item[2]
        audio_or_media_path = str(item[3])
        prompt = str(item[4])
        samples.append((img_path, seed, hw, audio_or_media_path, prompt))
    return samples


def shard_indices(n: int, rank: int, world_size: int):
    return [i for i in range(n) if (i % world_size) == rank]


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser("SkyReels V3 TalkingAvatar batch runner (YAML)")

    ap.add_argument("--yaml_path", type=str, required=True, help="YAML file containing samples.")
    ap.add_argument("--model_path", type=str, required=True, help="Local model directory (already downloaded).")
    ap.add_argument("--save_dir", type=str, default="result/talking_avatar_batch", help="Where to save results.")

    # W&B
    ap.add_argument("--wandb_entity", type=str, default="hedra-Hedra")
    ap.add_argument("--wandb_project", type=str, default="skyreel2-test")
    ap.add_argument("--exp_name", type=str, default="skyreel2-test")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    # generation defaults
    ap.add_argument("--default_seed", type=int, default=42)
    ap.add_argument("--motion_frame", type=int, default=5)
    ap.add_argument("--frame_num", type=int, default=81)
    ap.add_argument("--drop_frame", type=int, default=12)
    ap.add_argument("--shift", type=int, default=11)
    ap.add_argument("--text_guide_scale", type=float, default=1.0)
    ap.add_argument("--audio_guide_scale", type=float, default=1.0)
    ap.add_argument("--sampling_steps", type=int, default=4)
    ap.add_argument("--max_frames_num", type=int, default=5000)

    # performance
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--low_vram", action="store_true")
    ap.add_argument("--use_usp", action="store_true", help="Keep if your pipeline supports USP here (optional).")

    # NEW: multi-speaker helper options
    ap.add_argument("--multispeaker", action="store_true", help="Treat YAML audio_path as full media (mp4/wav) and split to personK.")
    ap.add_argument("--max_speakers", type=int, default=2)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--use_pyannote", action="store_true", help="Use pyannote diarization if installed and token is provided.")
    ap.add_argument("--pyannote_token", type=str, default="", help="HF token for pyannote (required if --use_pyannote).")
    ap.add_argument("--diar_chunk_s", type=float, default=4.0, help="Fallback chunk size for round-robin diarization.")

    args = ap.parse_args()

    is_dist, rank, world_size, local_rank = init_dist_if_needed()
    _ = pick_device(local_rank)

    samples = load_yaml_samples(args.yaml_path)
    total = len(samples)
    logging.info(f"Loaded {total} samples from {args.yaml_path}")

    save_dir = ensure_dir(args.save_dir)
    raw_video_dir = ensure_dir(os.path.join(save_dir, "raw_no_audio"))
    final_video_dir = ensure_dir(os.path.join(save_dir, "final_with_audio"))
    meta_dir = ensure_dir(os.path.join(save_dir, "meta"))
    processed_audio_root = ensure_dir(os.path.join(save_dir, "processed_audio"))
    multispeaker_root = ensure_dir(os.path.join(save_dir, "multispeaker_assets"))

    use_wandb = (args.wandb_mode != "disabled") and (rank == 0)
    if use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.exp_name,
            config={
                "yaml_path": args.yaml_path,
                "model_path": args.model_path,
                "offload": args.offload,
                "low_vram": args.low_vram,
                "use_usp": args.use_usp,
                "default_seed": args.default_seed,
                "motion_frame": args.motion_frame,
                "frame_num": args.frame_num,
                "drop_frame": args.drop_frame,
                "shift": args.shift,
                "text_guide_scale": args.text_guide_scale,
                "audio_guide_scale": args.audio_guide_scale,
                "sampling_steps": args.sampling_steps,
                "max_frames_num": args.max_frames_num,
                "world_size": world_size,
                "multispeaker": args.multispeaker,
                "max_speakers": args.max_speakers,
            },
            mode=args.wandb_mode,
        )

    config = WAN_CONFIGS["talking-avatar-19B"]
    pipe = TalkingAvatarPipeline(
        config=config,
        model_path=args.model_path,
        device_id=local_rank,
        rank=rank,
        use_usp=args.use_usp,
        offload=args.offload,
        low_vram=args.low_vram,
    )

    my_indices = shard_indices(total, rank, world_size) if is_dist else list(range(total))
    logging.info(f"Rank {rank}/{world_size} will process {len(my_indices)} samples.")

    for idx in my_indices:
        img_path, seed, hw, audio_or_media_path, prompt = samples[idx]
        if seed is None:
            seed = args.default_seed

        sample_id = f"{idx:04d}_{safe_stem(img_path)}"
        size_bucket = parse_resolution_bucket(hw)

        # -------------------------
        # build input_data
        # -------------------------
        if args.multispeaker:
            # YAML column[3] treated as full media path (mp4/wav)
            ms_out = ensure_dir(os.path.join(multispeaker_root, sample_id))
            meta = prepare_multispeaker_assets(
                media_path=audio_or_media_path,
                out_dir=ms_out,
                max_speakers=args.max_speakers,
                sample_rate=args.sample_rate,
                target_fps=25,
                use_pyannote=args.use_pyannote,
                pyannote_hf_token=(args.pyannote_token if args.pyannote_token else None),
                diar_chunk_s=args.diar_chunk_s,
            )
            input_data = build_pipeline_input_data(
                prompt=prompt,
                cond_image_path=img_path,
                multispeaker_meta=meta,
            )
        else:
            # original single-speaker path
            input_data = {
                "prompt": prompt,
                "cond_image": img_path,
                "cond_audio": {"person1": audio_or_media_path},
            }

        # -------------------------
        # preprocess audio: wav -> pt embeddings (your existing module)
        # -------------------------
        sample_audio_dir = ensure_dir(os.path.join(processed_audio_root, sample_id))
        input_data, _ = preprocess_audio(args.model_path, input_data, sample_audio_dir)
        logging.info(f"[{sample_id}] cond_audio(after preprocess) = {input_data.get('cond_audio')}")

        kwargs = {
            "input_data": input_data,
            "size_buckget": size_bucket,
            "motion_frame": args.motion_frame,
            "frame_num": args.frame_num,
            "drop_frame": args.drop_frame,
            "shift": args.shift,
            "text_guide_scale": args.text_guide_scale,
            "audio_guide_scale": args.audio_guide_scale,
            "seed": seed,
            "sampling_steps": args.sampling_steps,
            "max_frames_num": args.max_frames_num,
        }

        logging.info(f"[{sample_id}] generate: bucket={size_bucket}, seed={seed}, multispeaker={args.multispeaker}")
        t0 = time.time()

        if torch.cuda.is_available():
            free, total_mem = torch.cuda.mem_get_info()
            logging.info(f"[{sample_id}] cuda mem free={free/1e9:.2f}GB total={total_mem/1e9:.2f}GB")
        logging.info(f"[{sample_id}] img_exists={os.path.exists(img_path)} path={img_path}")

        video_out = None

        for attempt in (1, 2):
            try:
                video_out = pipe.generate(**kwargs)

                # 强校验：generate 正常返回就必须是非空帧序列/数组
                if video_out is None:
                    raise RuntimeError("pipe.generate() returned None without exception (unexpected).")

                # 可选：如果你期望至少 1 帧
                try:
                    if len(video_out) == 0:
                        raise RuntimeError("pipe.generate() returned empty video (len==0).")
                except TypeError:
                    # video_out 可能是 numpy array，没有 __len__? 一般有，这里兜底
                    pass
                print("video_out:", type(video_out), getattr(video_out, "shape", None), getattr(video_out, "dtype", None))
                if isinstance(video_out, np.ndarray):
                    print("min/max/mean:", float(video_out.min()), float(video_out.max()), float(video_out.mean()))
                    print("first frame mean:", float(video_out[0].mean()))
                break  # 成功就跳出重试循环

            except Exception:
                # 1) 打印完整 traceback（最关键）
                logging.error(f"[{sample_id}] generate crashed on attempt {attempt}. Full traceback:\n{traceback.format_exc()}")

                # 2) 清缓存（保留你原逻辑）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 3) 如果是最后一次，直接抛出，让 bug 暴露（程序会退出并显示堆栈）
                if attempt == 2:
                    raise

        dt = time.time() - t0
        logging.info(f"[{sample_id}] generate time={dt:.2f}s")

        if video_out is None:
            logging.error(f"[{sample_id}] generate returned None. time={dt:.2f}s. err={err_msg}")
            meta_path = os.path.join(meta_dir, f"{sample_id}_r{rank}.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"index: {idx}\n")
                f.write(f"sample_id: {sample_id}\n")
                f.write(f"image: {img_path}\n")
                f.write(f"audio_or_media: {audio_or_media_path}\n")
                f.write(f"seed: {seed}\n")
                f.write(f"bucket: {size_bucket}\n")
                f.write(f"rank: {rank}\n")
                f.write(f"status: failed\n")
                f.write(f"error: {err_msg}\n")
                f.write(f"prompt: {prompt}\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        logging.info(f"[{sample_id}] generated frames={len(video_out)} in {dt:.2f}s")

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        raw_path = os.path.join(raw_video_dir, f"{sample_id}_seed{seed}_{current_time}_r{rank}.mp4")
        fps = 25
        imageio.mimwrite(
            raw_path,
            video_out,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )

        # mux with audio if available (preprocess_audio usually writes input_data["video_audio"])
        audio_for_mux = kwargs["input_data"].get("video_audio", None)
        final_path = os.path.join(final_video_dir, f"{sample_id}_seed{seed}_{current_time}_r{rank}.mp4")
        if audio_for_mux and os.path.exists(audio_for_mux):
            try:
                ffmpeg_mux(raw_path, audio_for_mux, final_path)
                os.remove(raw_path)
                saved_video = final_path
            except Exception as e:
                logging.warning(f"[{sample_id}] ffmpeg mux failed: {e}. Keep raw video.")
                saved_video = raw_path
        else:
            saved_video = raw_path

        meta_path = os.path.join(meta_dir, f"{sample_id}_r{rank}.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"index: {idx}\n")
            f.write(f"sample_id: {sample_id}\n")
            f.write(f"image: {img_path}\n")
            f.write(f"audio_or_media: {audio_or_media_path}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"bucket: {size_bucket}\n")
            f.write(f"rank: {rank}\n")
            f.write(f"saved_video: {saved_video}\n")
            f.write(f"prompt: {prompt}\n")

        if use_wandb and (not is_dist):
            wandb.log(
                {
                    "idx": idx,
                    "sample_id": sample_id,
                    "seed": seed,
                    "bucket": size_bucket,
                    "prompt": prompt,
                    "video": wandb.Video(saved_video, fps=fps, format="mp4"),
                }
            )

    barrier_if_dist(is_dist)

    if use_wandb and is_dist:
        all_mp4 = sorted([str(p) for p in Path(final_video_dir).glob("*.mp4")])
        logging.info(f"[rank0] logging {len(all_mp4)} videos to wandb from {final_video_dir}")

        table = wandb.Table(columns=["idx", "sample_id", "seed", "bucket", "rank", "prompt", "video_path", "video"])

        meta_files = sorted(Path(meta_dir).glob("*.txt"))
        meta_map = {}
        for mf in meta_files:
            txt = mf.read_text(encoding="utf-8", errors="ignore").splitlines()
            d = {}
            for line in txt:
                if ":" in line:
                    k, v = line.split(":", 1)
                    d[k.strip()] = v.strip()
            meta_map[d.get("saved_video", "")] = d

        for vp in all_mp4:
            d = meta_map.get(vp, {})
            idx = int(d.get("index", "-1"))
            sample_id = d.get("sample_id", Path(vp).stem)
            seed = int(d.get("seed", "0")) if d.get("seed", "").isdigit() else 0
            bucket = d.get("bucket", "")
            r = d.get("rank", "")
            prompt = d.get("prompt", "")

            table.add_data(
                idx,
                sample_id,
                seed,
                bucket,
                r,
                prompt[:5000],
                vp,
                wandb.Video(vp, fps=25, format="mp4"),
            )

        wandb.log({"results_table": table})

    if use_wandb:
        wandb.finish()

    destroy_dist(is_dist)
    logging.info(f"Done. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()

