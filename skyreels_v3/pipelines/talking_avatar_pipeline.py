import gc
import logging
import math
import os
import random
import types
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file
from torchao.quantization import float8_weight_only, quantize_
from tqdm import tqdm

from ..modules.clip import CLIPModel
from ..modules.t5 import T5EncoderModel
from ..modules.transformer_a2v import WanModel
from ..modules.vae import WanVAE
from ..utils.avatar_util import (
    ASPECT_RATIO_627,
    ASPECT_RATIO_960,
    match_and_blend_colors,
    process_video_samples,
)


def resize_and_centercrop(cond_image, target_size):
    """
    Resize image or tensor to the target size without padding.
    """

    # Get the original size
    if isinstance(cond_image, torch.Tensor):
        _, orig_h, orig_w = cond_image.shape
    else:
        orig_h, orig_w = cond_image.height, cond_image.width

    target_h, target_w = target_size

    # Calculate the scaling factor for resizing
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    # Compute the final size
    scale = max(scale_h, scale_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    # Resize
    if isinstance(cond_image, torch.Tensor):
        if len(cond_image.shape) == 3:
            cond_image = cond_image[None]
        resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode="nearest").contiguous()
        # crop
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor.squeeze(0)
    else:
        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_image = np.array(resized_image)
        # tensor and crop
        resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor[:, :, None, :, :]

    return cropped_tensor


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def _person_key_sort(k: str) -> int:
    # stable order: person1, person2, ...
    if k.startswith("person"):
        try:
            return int(k.replace("person", ""))
        except Exception:
            return 10**9
    return 10**9


def _build_masks_from_bbox_or_stripes(
    src_h: int,
    src_w: int,
    person_keys: List[str],
    bbox: Optional[Dict[str, List[float]]] = None,
) -> List[torch.Tensor]:
    """
    Return list of masks: [person1_mask, person2_mask, ..., background_mask]
    mask shape: [H,W], float32 0/1
    bbox format: {personK: [x_min, y_min, x_max, y_max]}  (x=width axis, y=height axis)
    """
    n = len(person_keys)
    human_masks: List[torch.Tensor] = []
    union = torch.zeros([src_h, src_w], dtype=torch.float32)

    if bbox is not None:
        for k in person_keys:
            if k not in bbox:
                raise ValueError(f"bbox missing key={k}, bbox keys={list(bbox.keys())}")
            x_min, y_min, x_max, y_max = bbox[k]
            x_min = int(max(0, min(src_w - 1, round(x_min))))
            x_max = int(max(0, min(src_w, round(x_max))))
            y_min = int(max(0, min(src_h - 1, round(y_min))))
            y_max = int(max(0, min(src_h, round(y_max))))

            m = torch.zeros([src_h, src_w], dtype=torch.float32)
            # IMPORTANT: y is height axis, x is width axis
            m[y_min:y_max, x_min:x_max] = 1.0
            human_masks.append(m)
            union += m
    else:
        # fallback: split into N vertical stripes
        stripe_w = max(1, src_w // n)
        for i in range(n):
            x0 = i * stripe_w
            x1 = src_w if i == n - 1 else (i + 1) * stripe_w
            m = torch.zeros([src_h, src_w], dtype=torch.float32)
            m[:, x0:x1] = 1.0
            human_masks.append(m)
            union += m

    background_mask = torch.where(union > 0, torch.tensor(0.0), torch.tensor(1.0))
    human_masks.append(background_mask)
    return human_masks


class TalkingAvatarPipeline:
    @classmethod
    def init_dit_model(
        cls,
        checkpoint_dir: str,
        quant: bool = False,
    ) -> Dict[str, WanModel]:
        print(f"load dit model from: {checkpoint_dir}")
        state_dict = {}
        with torch.device("cpu"):
            for file in os.listdir(checkpoint_dir):
                if file.endswith(".safetensors"):
                    state_dict.update(load_file(os.path.join(checkpoint_dir, file)))

        model = WanModel.from_config(os.path.join(checkpoint_dir, "config.json")).to(torch.bfloat16)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        model.eval().requires_grad_(False)
        model = model.to(torch.bfloat16)
        if quant:
            quantize_(model, float8_weight_only(), device="cuda")
            print(f"quantize dit model")

        return {"model": model}

    def __init__(
        self,
        config,
        model_path: str,
        device_id=0,
        rank=0,
        use_usp=False,
        num_timesteps=1000,
        use_timestep_transform=True,
        offload=False,
        low_vram=False,
    ):
        offload = offload or low_vram
        quant = low_vram

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.text_encoder = (
            T5EncoderModel(
                text_len=config.text_len,
                checkpoint_path=os.path.join(model_path, config.t5_checkpoint),
                tokenizer_path=os.path.join(model_path, config.t5_tokenizer),
                shard_fn=None,
            )
            .to(config.t5_dtype)
            .to("cpu")
        )

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(model_path, config.clip_checkpoint),
            tokenizer_path=os.path.join(model_path, config.clip_tokenizer),
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(model_path, config.vae_checkpoint),
        )

        # load dit model
        logging.info(f"Creating WanModel from {model_path}")
        self.model = self.init_dit_model(
            checkpoint_dir=model_path,
            quant=quant,
        )["model"]

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.context_parallel_for_avatar import (
                usp_attn_forward_avatar,
                usp_crossattn_multi_forward_avatar,
                usp_dit_forward_avatar,
            )

            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward_avatar, block.self_attn)
                block.audio_cross_attn.origin_forward = block.audio_cross_attn.forward
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_avatar, block.audio_cross_attn
                )
            self.model.forward = types.MethodType(usp_dit_forward_avatar, self.model)
            self.sp_size = get_sequence_parallel_world_size()
            local_rank = dist.get_rank() % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            self.device = f"cuda:{local_rank}"
        else:
            self.sp_size = 1
            self.device = "cuda"

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.offload = offload
        self.low_vram = low_vram
        if self.offload:
            self.model.to("cpu")
            self.text_encoder.to("cpu")
            self.clip.model.to("cpu")
        else:
            self.model.to(self.device)
            self.text_encoder.to(self.device)
            self.clip.model.to(self.device)
        self.vae.to(self.device)

        # NOTE: 不再在 init / generate 里无条件 barrier（容易 NCCL timeout）
        if self.use_usp and dist.is_initialized():
            dist.barrier()

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))
        return (1 - timesteps) * original_samples + timesteps * noise

    def generate(
        self,
        input_data,
        size_buckget="720P",
        motion_frame=25,
        drop_frame=12,
        frame_num=81,
        shift=5.0,
        sampling_steps=40,
        text_guide_scale=5.0,
        audio_guide_scale=4.0,
        n_prompt="",
        connection_prompt="a person is talking",
        seed=-1,
        max_frames_num=5000,
        progress=True,
    ):
        input_prompt = input_data["prompt"]
        cond_file_path = input_data["cond_image"]
        cond_image = Image.open(cond_file_path).convert("RGB")

        # decide a proper size
        if size_buckget == "480P":
            bucket_config = ASPECT_RATIO_627
        elif size_buckget == "720P":
            bucket_config = ASPECT_RATIO_960
        else:
            raise ValueError("Unsupported size bucket: {}".format(size_buckget))

        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        cond_image = resize_and_centercrop(cond_image, (target_h, target_w))

        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2  # normalization
        cond_image = cond_image.to(self.device)  # 1 C 1 H W

        original_color_reference = cond_image.clone()

        # -------------------------
        # read audio embeddings (NOW supports multi-person)
        # input_data["cond_audio"] should be dict: {"person1": "...pt", "person2": "...pt", ...}
        # -------------------------
        cond_audio = input_data.get("cond_audio", {})
        if not isinstance(cond_audio, dict) or len(cond_audio) == 0:
            raise ValueError("input_data['cond_audio'] must be a dict like {'person1': '...pt', 'person2': '...pt'}")

        person_keys = sorted(list(cond_audio.keys()), key=_person_key_sort)
        HUMAN_NUMBER = len(person_keys)
        audio_embedding_paths = [cond_audio[k] for k in person_keys]

        full_audio_embs = []
        for pkey, audio_embedding_path in zip(person_keys, audio_embedding_paths):
            if not os.path.exists(audio_embedding_path):
                raise FileNotFoundError(audio_embedding_path)
            full_audio_emb = torch.load(audio_embedding_path, map_location="cpu")
            if torch.isnan(full_audio_emb).any():
                raise ValueError(f"NaNs in audio embedding: {pkey} -> {audio_embedding_path}")
            full_audio_embs.append(full_audio_emb)

        assert len(full_audio_embs) == HUMAN_NUMBER, "audio embedding count mismatch"

        # preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if self.offload:
            self.text_encoder.to(self.device)
        context, context_null, connection_embedding = self.text_encoder.encode(
            [input_prompt, n_prompt, connection_prompt],
        )
        if self.offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []

        is_clip = False
        window_size = frame_num
        overlap = motion_frame + drop_frame
        video_length_real = min(max_frames_num, len(full_audio_embs[0]))
        if video_length_real < window_size:
            is_clip = True
            video_length = window_size
        else:
            remainder = (video_length_real - window_size) % (window_size - overlap)
            if remainder != 0:
                video_length = video_length_real + (window_size - overlap) - remainder
                add_audio_emb = torch.flip(
                    full_audio_embs[0][-1 * (video_length - video_length_real) :],
                    dims=[0],
                )
                full_audio_embs[0] = torch.cat([full_audio_embs[0], add_audio_emb], dim=0)
                assert len(full_audio_embs[0]) == video_length, f"audio_length not equals to video_length"
            else:
                video_length = video_length_real

        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # -------------------------
        # build initial audio_embs [HUMAN_NUMBER, ...]
        # -------------------------
        audio_embs = []
        for human_idx in range(HUMAN_NUMBER):
            center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0] - 1).cpu()
            audio_emb = full_audio_embs[human_idx][center_indices][None, ...].to(self.device)
            audio_embs.append(audio_emb)
        audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)

        bg_audio = torch.zeros_like(audio_embs[:1])
        audio_embs = torch.cat([audio_embs, bg_audio], dim=0)  # [HUMAN_NUMBER+1, T, K, D]
        
        h, w = cond_image.shape[-2], cond_image.shape[-1]
        lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
        max_seq_len = (
            ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        noise = torch.randn(
            16,
            (frame_num - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            device=self.device,
        )

        # get mask
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        msk[:, cur_motion_frames_num:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2).to(self.param_dtype)  # B 4 T H W

        with torch.no_grad():
            if self.offload:
                self.clip.model.to(self.device)
            clip_context = self.clip.visual(cond_image[:, :, :1, :, :]).to(self.param_dtype)
            if self.offload:
                self.clip.model.to("cpu")
                torch.cuda.empty_cache()

            video_frames = torch.zeros(
                1,
                cond_image.shape[1],
                frame_num - cond_image.shape[2],
                target_h,
                target_w,
            ).to(self.device)
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)

            y = self.vae.encode(padding_frames_pixels_values).to(self.param_dtype)
            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
            latent_motion_frames = y[:, :, :cur_motion_frames_latent_num][0]
            y = torch.concat([msk, y], dim=1)  # B 4+C T H W
            del video_frames, padding_frames_pixels_values

        # -------------------------
        # construct human masks (NOW supports multi-person)
        # prefer input_data["bbox"], else stripes
        # -------------------------
        bbox = input_data.get("bbox", None)
        human_masks = _build_masks_from_bbox_or_stripes(
            src_h=src_h,
            src_w=src_w,
            person_keys=person_keys,
            bbox=bbox,
        )
        ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)  # [HUMAN_NUMBER+1, H, W]

        # resize and centercrop for ref_target_masks
        ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

        _, _, _, lat_h, lat_w = y.shape
        ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode="nearest").squeeze(0)
        ref_target_masks = (ref_target_masks > 0).float().to(self.device)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # evaluation mode
        with torch.no_grad(), no_sync():
            # prepare timesteps
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.0)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

            latent = noise

            arg_c = {
                "context": [context],
                "clip_fea": clip_context,
                "seq_len": max_seq_len,
                "y": y,
                "audio": audio_embs,
                "ref_target_masks": ref_target_masks,
                "block_offload": self.low_vram,
            }

            arg_null_text = {
                "context": [context_null],
                "clip_fea": clip_context,
                "seq_len": max_seq_len,
                "y": y,
                "audio": audio_embs,
                "ref_target_masks": ref_target_masks,
                "block_offload": self.low_vram,
            }

            arg_null_audio = {
                "context": [context],
                "clip_fea": clip_context,
                "seq_len": max_seq_len,
                "y": y,
                "audio": torch.zeros_like(audio_embs)[-1:],
                "ref_target_masks": ref_target_masks,
                "block_offload": self.low_vram,
            }

            arg_null = {
                "context": [context_null],
                "clip_fea": clip_context,
                "seq_len": max_seq_len,
                "y": y,
                "audio": torch.zeros_like(audio_embs)[-1:],
                "ref_target_masks": ref_target_masks,
                "block_offload": self.low_vram,
            }

            if self.offload and not self.low_vram:
                self.model.to(self.device)

            progress_wrap = partial(tqdm, total=len(timesteps) - 1) if progress else (lambda x: x)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]
                latent_model_input = [latent.to(self.device)]

                noise_pred_drop_text, noise_pred_uncond, noise_pred_drop_audio = (None, None, None)

                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                if text_guide_scale > 1.0 and audio_guide_scale > 1.0:
                    noise_pred_drop_text = self.model(latent_model_input, t=timestep, **arg_null_text)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                elif text_guide_scale > 1.0:
                    noise_pred_drop_text = self.model(latent_model_input, t=timestep, **arg_null_text)[0]
                elif audio_guide_scale > 1.0:
                    noise_pred_drop_audio = self.model(latent_model_input, t=timestep, **arg_null_audio)[0]

                # vanilla CFG
                if text_guide_scale > 1.0 and audio_guide_scale > 1.0:
                    noise_pred = (
                        noise_pred_uncond
                        + text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                        + audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                    )
                elif text_guide_scale > 1.0:
                    noise_pred = noise_pred_drop_text + text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                elif audio_guide_scale > 1.0:
                    noise_pred = noise_pred_drop_audio + audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                else:
                    noise_pred = noise_pred_cond
                noise_pred = -noise_pred

                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                latent = latent + noise_pred * dt[:, None, None, None]

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if self.offload and not self.low_vram:
                self.model.to("cpu")
                torch.cuda.empty_cache()

            videos = self.vae.decode(x0[0])
            torch.cuda.empty_cache()

        generated_ref_videos = videos
        generated_ref_videos = match_and_blend_colors(generated_ref_videos, original_color_reference, 1.0)

        processed_generated_ref_videos = process_video_samples(generated_ref_videos)
        generated_ref_videos = generated_ref_videos.cpu()
        del videos

        if not is_clip:
            audio_length = min(max_frames_num, len(full_audio_embs[0]))
            if audio_length <= frame_num:
                generate_idx = [0, audio_length - 1]
            else:
                generate_idx = [0]
                current_idx = frame_num - 1
                segment_interval = frame_num - motion_frame - drop_frame
                while current_idx < audio_length - 1:
                    generate_idx.append(current_idx)
                    current_idx += segment_interval
                if generate_idx[-1] != audio_length - 1:
                    generate_idx.append(audio_length - 1)
            generate_idx = np.array(generate_idx, dtype=np.int16)
            original_max = generate_idx[-1]
            original_min = generate_idx[0]
            if original_max > original_min:
                generate_idx_float = (
                    (generate_idx.astype(np.float64) - original_min) * (frame_num - 1) / (original_max - original_min)
                )
                generate_idx = np.clip(np.round(generate_idx_float), 0, frame_num - 1).astype(np.int32)

            generate_idx = generate_idx[1:]
            generated_ref_videos_final = generated_ref_videos[:, :, generate_idx]

            tmp_indx = 0
            while True:
                if audio_end_idx == video_length:
                    arrive_last_frame = True

                # rebuild audio_embs for this window (multi-person)
                audio_embs = []
                for human_idx in range(HUMAN_NUMBER):
                    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
                    center_indices = torch.clamp(
                        center_indices, min=0, max=full_audio_embs[human_idx].shape[0] - 1
                    ).cpu()
                    audio_emb = full_audio_embs[human_idx][center_indices][None, ...].to(self.device)
                    audio_embs.append(audio_emb)
                audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)
                bg_audio = torch.zeros_like(audio_embs[:1])
                audio_embs = torch.cat([audio_embs, bg_audio], dim=0)

                h, w = cond_image.shape[-2], cond_image.shape[-1]
                lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
                max_seq_len = (
                    ((frame_num - 1) // self.vae_stride[0] + 1)
                    * lat_h
                    * lat_w
                    // (self.patch_size[1] * self.patch_size[2])
                )
                max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

                noise = torch.randn(
                    16,
                    (frame_num - 1) // 4 + 1,
                    lat_h,
                    lat_w,
                    dtype=torch.float32,
                    device=self.device,
                )

                tmp_indx = min(tmp_indx, generated_ref_videos_final.shape[2] - 1)
                pseudo_frames = (
                    generated_ref_videos_final[:, :, tmp_indx : tmp_indx + 1].repeat(1, 1, 5, 1, 1).to(self.device)
                )
                tmp_indx += 1

                msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
                msk[:, cur_motion_frames_num : -pseudo_frames.shape[2]] = 0
                msk = torch.concat(
                    [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                    dim=1,
                )
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2).to(self.param_dtype)

                with torch.no_grad():
                    if self.offload:
                        self.clip.model.to(self.device)
                    clip_context = self.clip.visual(cond_image[:, :, :1, :, :]).to(self.param_dtype)
                    if self.offload:
                        self.clip.model.to("cpu")
                        torch.cuda.empty_cache()

                    video_frames = torch.zeros(
                        1,
                        cond_image.shape[1],
                        frame_num - cond_image.shape[2] - pseudo_frames.shape[2],
                        target_h,
                        target_w,
                    ).to(self.device)
                    padding_frames_pixels_values = torch.concat([cond_image, video_frames, pseudo_frames], dim=2)

                    y = self.vae.encode(padding_frames_pixels_values).to(self.param_dtype)
                    cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
                    latent_motion_frames = y[:, :, :cur_motion_frames_latent_num][0]
                    y = torch.concat([msk, y], dim=1)
                    del video_frames, padding_frames_pixels_values

                # reuse the same masks (bbox is static); rebuild to match y's lat_h/lat_w
                ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
                ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))
                _, _, _, lat_h, lat_w = y.shape
                ref_target_masks = F.interpolate(
                    ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode="nearest"
                ).squeeze(0)
                ref_target_masks = (ref_target_masks > 0).float().to(self.device)

                @contextmanager
                def noop_no_sync():
                    yield

                no_sync = getattr(self.model, "no_sync", noop_no_sync)

                with torch.no_grad(), no_sync():
                    timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                    timesteps.append(0.0)
                    timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                    if self.use_timestep_transform:
                        timesteps = [
                            timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps
                        ]

                    latent = noise

                    arg_c = {
                        "context": [connection_embedding],
                        "clip_fea": clip_context,
                        "seq_len": max_seq_len,
                        "y": y,
                        "audio": audio_embs,
                        "ref_target_masks": ref_target_masks,
                        "block_offload": self.low_vram,
                    }
                    arg_null_text = {
                        "context": [context_null],
                        "clip_fea": clip_context,
                        "seq_len": max_seq_len,
                        "y": y,
                        "audio": audio_embs,
                        "ref_target_masks": ref_target_masks,
                        "block_offload": self.low_vram,
                    }
                    arg_null_audio = {
                        "context": [connection_embedding],
                        "clip_fea": clip_context,
                        "seq_len": max_seq_len,
                        "y": y,
                        "audio": torch.zeros_like(audio_embs)[-1:],
                        "ref_target_masks": ref_target_masks,
                        "block_offload": self.low_vram,
                    }
                    arg_null = {
                        "context": [context_null],
                        "clip_fea": clip_context,
                        "seq_len": max_seq_len,
                        "y": y,
                        "audio": torch.zeros_like(audio_embs)[-1:],
                        "ref_target_masks": ref_target_masks,
                        "block_offload": self.low_vram,
                    }

                    if self.offload and not self.low_vram:
                        self.model.to(self.device)

                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                        motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                        add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                        _, T_m, _, _ = add_latent.shape
                        latent[:, :T_m] = add_latent

                    progress_wrap = partial(tqdm, total=len(timesteps) - 1) if progress else (lambda x: x)
                    for i in progress_wrap(range(len(timesteps) - 1)):
                        timestep = timesteps[i]
                        latent_model_input = [latent.to(self.device)]

                        noise_pred_drop_text, noise_pred_uncond, noise_pred_drop_audio = (None, None, None)

                        noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]

                        if text_guide_scale > 1.0 and audio_guide_scale > 1.0:
                            noise_pred_drop_text = self.model(latent_model_input, t=timestep, **arg_null_text)[0]
                            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                        elif text_guide_scale > 1.0:
                            noise_pred_drop_text = self.model(latent_model_input, t=timestep, **arg_null_text)[0]
                        elif audio_guide_scale > 1.0:
                            noise_pred_drop_audio = self.model(latent_model_input, t=timestep, **arg_null_audio)[0]

                        if text_guide_scale > 1.0 and audio_guide_scale > 1.0:
                            noise_pred = (
                                noise_pred_uncond
                                + text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                                + audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                            )
                        elif text_guide_scale > 1.0:
                            noise_pred = noise_pred_drop_text + text_guide_scale * (
                                noise_pred_cond - noise_pred_drop_text
                            )
                        elif audio_guide_scale > 1.0:
                            noise_pred = noise_pred_drop_audio + audio_guide_scale * (
                                noise_pred_cond - noise_pred_drop_audio
                            )
                        else:
                            noise_pred = noise_pred_cond
                        noise_pred = -noise_pred

                        dt = timesteps[i] - timesteps[i + 1]
                        dt = dt / self.num_timesteps
                        latent = latent + noise_pred * dt[:, None, None, None]

                        if not is_first_clip:
                            latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                            motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                            add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[i + 1])
                            _, T_m, _, _ = add_latent.shape
                            latent[:, :T_m] = add_latent

                        x0 = [latent.to(self.device)]
                        del latent_model_input, timestep

                    if self.offload and not self.low_vram:
                        self.model.to("cpu")
                        torch.cuda.empty_cache()

                    videos = self.vae.decode(x0[0])
                    torch.cuda.empty_cache()

                if not arrive_last_frame:
                    videos = videos[:, :, :-drop_frame]

                videos = match_and_blend_colors(videos, original_color_reference, 1.0)
                processed_videos = process_video_samples(videos)
                videos = videos.cpu()

                if not is_first_clip:
                    gen_video_list.append(processed_videos[:, :, cur_motion_frames_num:])
                else:
                    gen_video_list.append(processed_videos)

                if arrive_last_frame:
                    break

                cur_motion_frames_num = motion_frame
                cond_image = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)

                audio_start_idx += frame_num - cur_motion_frames_num - drop_frame
                audio_end_idx = audio_start_idx + clip_length
                is_first_clip = False

                if max_frames_num <= frame_num:
                    break

        if is_clip:
            gen_video_list.append(processed_generated_ref_videos)

        gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, : int(max_frames_num)]
        gen_video_samples = gen_video_samples[:, :, :video_length_real]
        gen_video_samples = (
            gen_video_samples[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        )  # (T,H,W,C)

        del noise
        return gen_video_samples

