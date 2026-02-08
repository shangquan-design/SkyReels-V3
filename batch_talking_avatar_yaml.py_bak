import argparse
import logging
import os
import random
import time
import subprocess
from pathlib import Path

import imageio
import torch
import torch.distributed as dist
import wandb
import yaml

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.pipelines import TalkingAvatarPipeline
from skyreels_v3.utils.avatar_preprocess import preprocess_audio


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
    Map to pipeline bucket string: 480P/540P/720P.
    """
    if hw is None:
        return "720P"
    h = int(hw[0])
    if h <= 480:
        return "480P"
    elif h <= 540:
        return "540P"
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

    # Accept either a list directly, or dict with a key (e.g., "samples")
    if isinstance(data, dict):
        # try common keys
        for key in ["samples", "data", "items", "benchmark"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"YAML must be a list (or dict containing a list). Got: {type(data)}")

    # Validate each sample
    samples = []
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            raise ValueError(
                f"Sample #{i} must be a list of length>=5: [image_path, seed, [H,W], audio_path, prompt]. Got: {item}"
            )
        img_path = str(item[0])
        seed = int(item[1]) if item[1] is not None else None

        hw = item[2]
        audio_path = str(item[3])
        prompt = str(item[4])
        samples.append((img_path, seed, hw, audio_path, prompt))
    return samples


def shard_indices(n: int, rank: int, world_size: int):
    """Simple sharding: each rank handles i where i % world_size == rank."""
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

    # generation defaults (can override)
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

    args = ap.parse_args()

    is_dist, rank, world_size, local_rank = init_dist_if_needed()
    device = pick_device(local_rank)

    # ---- load samples ----
    samples = load_yaml_samples(args.yaml_path)
    total = len(samples)
    logging.info(f"Loaded {total} samples from {args.yaml_path}")

    # ---- output dirs ----
    save_dir = ensure_dir(args.save_dir)
    raw_video_dir = ensure_dir(os.path.join(save_dir, "raw_no_audio"))
    final_video_dir = ensure_dir(os.path.join(save_dir, "final_with_audio"))
    meta_dir = ensure_dir(os.path.join(save_dir, "meta"))
    processed_audio_root = ensure_dir(os.path.join(save_dir, "processed_audio"))

    # ---- init wandb (rank0 only; others skip to avoid multi-proc conflicts) ----
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
            },
            mode=args.wandb_mode,
        )

    # ---- init pipeline (each rank builds its own; model_path is local and already present) ----
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

    # ---- sharding (optional) ----
    my_indices = shard_indices(total, rank, world_size) if is_dist else list(range(total))
    logging.info(f"Rank {rank}/{world_size} will process {len(my_indices)} samples.")

    # ---- main loop ----
    for idx in my_indices:
        img_path, seed, hw, audio_path, prompt = samples[idx]
        if seed is None:
            seed = args.default_seed

        # Make per-sample id
        sample_id = f"{idx:04d}_{safe_stem(img_path)}"
        size_bucket = parse_resolution_bucket(hw)

        # preprocess audio (writes temp files)
        # input_data follows SkyReels pipeline conventions
        input_data = {
            "prompt": prompt,
            "cond_image": img_path,
            "cond_audio": {"person1": audio_path},
        }

        # Each rank preprocesses for its own samples (simple and robust)
        sample_audio_dir = ensure_dir(os.path.join(processed_audio_root, sample_id))
        input_data, _ = preprocess_audio(args.model_path, input_data, sample_audio_dir)

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

        logging.info(f"[{sample_id}] generate: bucket={size_bucket}, seed={seed}")
        t0 = time.time()

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logging.info(f"[{sample_id}] cuda mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB")
        logging.info(f"[{sample_id}] img_exists={os.path.exists(img_path)} audio_exists={os.path.exists(audio_path)}")

        video_out = None
        err_msg = None
        for attempt in [1, 2]:
            try:
                video_out = pipe.generate(**kwargs)
                if video_out is not None:
                    break
            except Exception as e:
                err_msg = f"attempt{attempt} {type(e).__name__}: {e}"
                logging.warning(f"[{sample_id}] generate failed: {err_msg}")

            # 清缓存再试一次
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



        dt = time.time() - t0

        if video_out is None:
            logging.error(f"[{sample_id}] generate returned None. time={dt:.2f}s. err={err_msg}")

            # 写meta，方便rank0汇总
            meta_path = os.path.join(meta_dir, f"{sample_id}_r{rank}.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"index: {idx}\n")
                f.write(f"sample_id: {sample_id}\n")
                f.write(f"image: {img_path}\n")
                f.write(f"audio: {audio_path}\n")
                f.write(f"seed: {seed}\n")
                f.write(f"bucket: {size_bucket}\n")
                f.write(f"rank: {rank}\n")
                f.write(f"status: failed\n")
                f.write(f"error: {err_msg}\n")
                f.write(f"prompt: {prompt}\n")

            # 可选：清一下缓存，避免后续连锁OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 直接跳过这条
            continue

        logging.info(f"[{sample_id}] generated frames={len(video_out)} in {dt:.2f}s")


        # save raw video (no audio) on each rank
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        raw_path = os.path.join(raw_video_dir, f"{sample_id}_seed{seed}_{current_time}_r{rank}.mp4")
        fps = 25  # talking_avatar
        imageio.mimwrite(
            raw_path,
            video_out,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )

        # mux with audio if available
        audio_for_mux = kwargs["input_data"].get("video_audio", None)
        final_path = os.path.join(final_video_dir, f"{sample_id}_seed{seed}_{current_time}_r{rank}.mp4")
        if audio_for_mux and os.path.exists(audio_for_mux):
            try:
                ffmpeg_mux(raw_path, audio_for_mux, final_path)
                os.remove(raw_path)  # keep only final
                saved_video = final_path
            except Exception as e:
                logging.warning(f"[{sample_id}] ffmpeg mux failed: {e}. Keep raw video.")
                saved_video = raw_path
        else:
            saved_video = raw_path

        # write meta
        meta_path = os.path.join(meta_dir, f"{sample_id}_r{rank}.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"index: {idx}\n")
            f.write(f"sample_id: {sample_id}\n")
            f.write(f"image: {img_path}\n")
            f.write(f"audio: {audio_path}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"bucket: {size_bucket}\n")
            f.write(f"rank: {rank}\n")
            f.write(f"saved_video: {saved_video}\n")
            f.write(f"prompt: {prompt}\n")

        # If single-process, log immediately; if multi-process, rank0 will log after barrier (see below)
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

    # ---- In dist mode, rank0 logs all videos after everyone finishes ----
    if use_wandb and is_dist:
        # rank0 logs everything found in final_video_dir (including other ranks)
        all_mp4 = sorted([str(p) for p in Path(final_video_dir).glob("*.mp4")])
        logging.info(f"[rank0] logging {len(all_mp4)} videos to wandb from {final_video_dir}")

        # Use a table for better browsing
        table = wandb.Table(columns=["idx", "sample_id", "seed", "bucket", "rank", "prompt", "video_path", "video"])
        # Also load meta files to recover prompt/idx info robustly
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

