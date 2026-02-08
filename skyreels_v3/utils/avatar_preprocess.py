import copy
import os
import time

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from ..modules.wav2vec2 import Wav2Vec2Model


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def audio_prepare_multi_new(cond_audios, sample_rate=16000, min_duration_s=0.4):
    """
    cond_audios:
      - dict: {"person1": "/path/a.wav", "person2": "/path/b.wav", ...}
      - list/tuple: ["/path/a.wav", "/path/b.wav", ...]
    """
    human_speech_arrays = []

    # 1) 统一拿到“音频路径列表”，并保证 person1/person2... 顺序
    if isinstance(cond_audios, dict):
        items = sorted(
            cond_audios.items(),
            key=lambda kv: int(kv[0].replace("person", "")) if str(kv[0]).startswith("person") else 10**9
        )
        audio_paths = [v for _, v in items]
    elif isinstance(cond_audios, (list, tuple)):
        audio_paths = list(cond_audios)
    else:
        raise TypeError(f"cond_audios must be dict or list/tuple, got {type(cond_audios)}")

    # 2) 逐个加载+补齐
    min_len = int(min_duration_s * sample_rate)
    for p in audio_paths:
        if not isinstance(p, str):
            raise TypeError(f"audio path must be str, got {type(p)}: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio file not found: {p}")

        human_speech = audio_prepare_single(p, sample_rate=sample_rate)

        # 如果 audio_prepare_single 返回的是 numpy array（通常是）
        if len(human_speech) < min_len:
            pad = min_len - len(human_speech)
            human_speech = np.pad(human_speech, (0, pad), mode="constant")

        human_speech_arrays.append(human_speech)

    sum_human_speechs = np.concatenate(human_speech_arrays) if len(human_speech_arrays) else np.zeros((min_len,), dtype=np.float32)
    return human_speech_arrays, sum_human_speechs

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device="cpu"):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(np.ceil(video_length)), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def audio_prepare_single(audio_path, sample_rate=16000, min_duration_s=0.4):
    """
    Load a single wav and return a 1D numpy array.
    If too short, pad with zeros to min_duration_s instead of raising.
    """
    human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)

    # librosa 有时会返回空数组（坏文件/读失败），兜底成静音
    if human_speech_array is None or len(human_speech_array) == 0:
        sr = sample_rate
        human_speech_array = np.zeros(int(min_duration_s * sr), dtype=np.float32)

    # NaN/Inf 兜底
    if not np.isfinite(human_speech_array).all():
        human_speech_array = np.nan_to_num(human_speech_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # pad to minimum duration
    min_len = int(min_duration_s * sr)
    if len(human_speech_array) < min_len:
        pad = min_len - len(human_speech_array)
        human_speech_array = np.pad(human_speech_array, (0, pad), mode="constant")

    # loudness normalize AFTER padding (避免短片段 norm 过激/报错)
    human_speech_array = loudness_norm(human_speech_array, sr)
    return human_speech_array

def preprocess_audio(model_path, input_data, audio_save_dir):
    def custom_init(device, wav2vec):
        audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
        return wav2vec_feature_extractor, audio_encoder

    w2v_path = os.path.join(model_path, "chinese-wav2vec2-base")
    wav2vec_feature_extractor, audio_encoder = custom_init("cpu", w2v_path)
    os.makedirs(audio_save_dir, exist_ok=True)

    return _preprocess_audio(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir)


def _preprocess_audio(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir):
    input_data = copy.deepcopy(input_data)
    fps = 25
    sample_rate = 16000

    start = time.time()
    os.makedirs(audio_save_dir, exist_ok=True)
    max_frames_num = input_data.get("max_frames_num", 5000)
    max_duration = max_frames_num / fps
    _ext_info = {}

    # -------- helper: stable order person1, person2, ...
    def _person_key_sort(k: str) -> int:
        if k.startswith("person"):
            try:
                return int(k.replace("person", ""))
            except Exception:
                return 10**9
        return 10**9

    # -------- 1) prepare speech segments
    if "cond_audio" not in input_data or not isinstance(input_data["cond_audio"], dict) or len(input_data["cond_audio"]) == 0:
        raise ValueError("input_data['cond_audio'] must be a non-empty dict like {'person1': 'xxx.wav', 'person2': 'yyy.wav'}")

    speech_list, sum_human_speechs = audio_prepare_multi_new(input_data["cond_audio"])

    audio_duration = len(sum_human_speechs) / sample_rate
    if audio_duration > max_duration:
        raise ValueError(
            f"Sum of audio duration is too long: {audio_duration:.2f}s. Maximum allowed: {max_duration}s"
        )

    # -------- 2) compute embeddings for each speech segment
    audio_emb_list = []
    trans_list = []
    for speech in speech_list:
        audio_embedding = get_embedding(speech, wav2vec_feature_extractor, audio_encoder)
        audio_emb_list.append(audio_embedding)
        trans_list.append(len(torch.cat(audio_emb_list, dim=0)))

    if len(audio_emb_list) == 0:
        raise ValueError("audio_prepare_multi_new returned empty speech_list; cannot build any audio embeddings.")

    # -------- 3) save embeddings as 1.pt, 2.pt, ...
    audio_emb_path_list = []
    for i, audio_embedding in enumerate(audio_emb_list):
        emb_path = os.path.join(audio_save_dir, f"{i+1}.pt")
        torch.save(audio_embedding, emb_path)
        audio_emb_path_list.append(emb_path)

    # -------- 4) write sum.wav for mux
    sum_audio = os.path.join(audio_save_dir, "sum.wav")
    print("sum_human_speechs:", len(sum_human_speechs))
    print("sum_audio:", sum_audio)

    sf.write(sum_audio, sum_human_speechs, sample_rate)
    def _person_key_sort(k: str) -> int:
        if k.startswith("person"):
            try:
                return int(k.replace("person", ""))
            except Exception:
                return 10**9
        return 10**9

    person_keys = sorted(list(input_data["cond_audio"].keys()), key=_person_key_sort)

    # speech_list should align with person_keys (audio_prepare_multi_new should guarantee this).
    # But to be safe, handle mismatch.
    if len(speech_list) != len(person_keys):
        print(
            f"[WARN] speech_list length ({len(speech_list)}) != num persons ({len(person_keys)}). "
            "Will save by index and also fall back to generic names."
        )

    for i, speech in enumerate(speech_list):
        # choose a nice filename: person1.wav, person2.wav...
        if i < len(person_keys):
            name = person_keys[i]
        else:
            name = f"person{i+1}"

        out_wav = os.path.join(audio_save_dir, f"{name}.wav")
        # ensure float32 to avoid weird dtype issues
        speech = np.asarray(speech, dtype=np.float32)
        sf.write(out_wav, speech, sample_rate)
        print(f"saved {name} wav: {out_wav}  (len={speech.shape[0]})")

    input_data["video_audio"] = sum_audio
    input_data["audio_embs"] = audio_emb_path_list
    input_data["trans_points"] = trans_list

    # -------- 5) IMPORTANT FIX:
    # Map ALL personK in cond_audio -> corresponding K.pt.
    # If speech_list has fewer segments than persons, fill with silent.pt (recommended).
    person_keys = sorted(list(input_data["cond_audio"].keys()), key=_person_key_sort)

    silent_emb_path = None
    if len(person_keys) > 1:
        # build silent embedding once as fallback
        silent_speech = np.zeros(sum_human_speechs.shape[0], dtype=np.float32)
        silent_audio_embedding = get_embedding(silent_speech, wav2vec_feature_extractor, audio_encoder)
        silent_emb_path = os.path.join(audio_save_dir, "silent.pt")
        torch.save(silent_audio_embedding, silent_emb_path)
        input_data["silent_audio_embs"] = silent_emb_path
    else:
        input_data["silent_audio_embs"] = None

    for i, k in enumerate(person_keys):
        if i < len(audio_emb_path_list):
            input_data["cond_audio"][k] = audio_emb_path_list[i]
        else:
            # more persons than extracted segments: fallback
            if silent_emb_path is not None:
                input_data["cond_audio"][k] = silent_emb_path
            else:
                # single-person edge-case fallback: repeat last embedding
                input_data["cond_audio"][k] = audio_emb_path_list[-1]

    # -------- 6) bbox alignment check (KEEP AS DICT, do NOT convert to list)
    if "bbox" in input_data and input_data["bbox"] is not None:
        if not isinstance(input_data["bbox"], dict):
            raise ValueError(f"bbox must be a dict like {{'person1':[x1,y1,x2,y2], ...}}, got {type(input_data['bbox'])}")

        # 这里 person_keys 是你前面算出来的 ['person1','person2',...]
        missing = [k for k in person_keys if k not in input_data["bbox"]]
        if missing:
            raise ValueError(f"bbox missing keys={missing}, bbox keys={list(input_data['bbox'].keys())}")

        if len(input_data["bbox"]) != len(input_data["cond_audio"]):
            raise ValueError(f"bbox len != cond_audio len: {len(input_data['bbox'])} vs {len(input_data['cond_audio'])}")

    # -------- 7) ext info
    _ext_info["final_video_length"] = trans_list[-1]

    print("cond_audio(final):", input_data["cond_audio"])
    print(f"preprocess audio time: {time.time() - start:0.2f}s")
    return input_data, _ext_info

