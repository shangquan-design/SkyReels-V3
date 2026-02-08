# skyreels_v3/utils/multispeaker_helper.py

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MultiSpeakerMeta:
    """
    cond_audio_wav: {"person1": ".../person1.wav", "person2": ".../person2.wav", ...}
      - IMPORTANT: each personK.wav is time-aligned to the extracted full wav:
        same duration as full_audio.wav; non-speech regions are silent.
    bbox: {"person1": [x1,y1,x2,y2], ...}  # static bbox in pixels
    segments: {"person1": [[s,e],...], ...}  # diarization segments (seconds)
    """
    cond_audio_wav: Dict[str, str]
    bbox: Dict[str, List[float]]
    segments: Dict[str, List[List[float]]]
    media_path: str
    extracted_audio_wav: str
    fps: float = 25.0
    width: Optional[int] = None
    height: Optional[int] = None


# ---------------------------
# Basic ffmpeg helpers
# ---------------------------
def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _which(exe: str) -> str:
    p = shutil.which(exe)
    if p is None:
        raise FileNotFoundError(f"'{exe}' not found in PATH")
    return p


def _ffprobe_duration_seconds(media_path: str) -> float:
    _which("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8", errors="ignore").strip()
    try:
        dur = float(out)
    except Exception:
        dur = 0.0
    return max(0.0, dur)


def extract_audio_to_wav(media_path: str, wav_out: str, sample_rate: int = 16000) -> str:
    _which("ffmpeg")
    wav_out = str(Path(wav_out))
    cmd = [
        "ffmpeg", "-y",
        "-i", str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        wav_out,
        "-loglevel", "error",
    ]
    _run(cmd)
    return wav_out


def ffprobe_video_info(media_path: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    _which("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(media_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
    j = json.loads(out)
    if "streams" not in j or len(j["streams"]) == 0:
        return None, None, None
    s = j["streams"][0]
    w = int(s.get("width")) if s.get("width") is not None else None
    h = int(s.get("height")) if s.get("height") is not None else None
    r = s.get("r_frame_rate", None)
    fps = None
    if isinstance(r, str) and "/" in r:
        a, b = r.split("/")
        try:
            fps = float(a) / float(b)
        except Exception:
            fps = None
    return w, h, fps


# ---------------------------
# Diarization (pyannote REQUIRED)
# ---------------------------
def _load_waveform_in_memory(wav_path: str):
    """
    Load waveform WITHOUT torchcodec, so pyannote won't touch AudioDecoder/torchcodec path.
    Returns: {"waveform": torch.Tensor[1,T], "sample_rate": int}
    """
    try:
        import soundfile as sf
    except Exception as e:
        raise ImportError("soundfile is required to load wav in-memory. `pip install soundfile`.") from e

    try:
        import torch
    except Exception as e:
        raise ImportError("torch is required for pyannote waveform input.") from e

    x, sr = sf.read(wav_path, always_2d=False)
    # x: (T,) or (T,C)
    if x is None:
        raise RuntimeError(f"Failed to read wav: {wav_path}")
    if isinstance(x, np.ndarray) and x.ndim == 2:
        # (T,C) -> mono
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    wav = torch.from_numpy(x).unsqueeze(0)  # (1,T)
    return {"waveform": wav, "sample_rate": int(sr)}


def _find_annotation_with_itertracks(obj):
    """
    Robustly find an object (possibly nested inside DiarizeOutput/dict/dataclass)
    that has .itertracks (pyannote.core.Annotation-like).
    """
    if obj is None:
        return None

    # direct
    if hasattr(obj, "itertracks") and callable(getattr(obj, "itertracks")):
        return obj

    # dict-like
    if isinstance(obj, dict):
        # common keys first
        for k in ("diarization", "annotation", "speaker_diarization", "output", "result"):
            if k in obj:
                ann = _find_annotation_with_itertracks(obj[k])
                if ann is not None:
                    return ann
        # fallback: scan all values
        for v in obj.values():
            ann = _find_annotation_with_itertracks(v)
            if ann is not None:
                return ann

    # dataclass / object: scan attributes
    try:
        d = vars(obj)  # __dict__
    except Exception:
        d = None

    if isinstance(d, dict) and d:
        # common fields first
        for k in ("diarization", "annotation", "speaker_diarization"):
            if k in d:
                ann = _find_annotation_with_itertracks(d[k])
                if ann is not None:
                    return ann
        for v in d.values():
            ann = _find_annotation_with_itertracks(v)
            if ann is not None:
                return ann

    # last resort: getattr scan (avoid private huge stuff)
    for name in ("diarization", "annotation", "speaker_diarization"):
        if hasattr(obj, name):
            ann = _find_annotation_with_itertracks(getattr(obj, name))
            if ann is not None:
                return ann

    return None


def diarize_pyannote_required(
    wav_path: str,
    max_speakers: int,
    token: Optional[str],
    model_id: str = "pyannote/speaker-diarization-3.1",
) -> Dict[str, List[List[float]]]:
    """
    Required diarization using pyannote.
    - Always uses in-memory waveform input to avoid torchcodec/AudioDecoder issues.
    - Compatible with pyannote.audio 4.x where Pipeline(...) returns DiarizeOutput.

    Returns:
        {"person1": [[s,e],...], ..., "personN": [[s,e],...]}  (always includes all persons)
    """
    if not isinstance(max_speakers, int) or max_speakers <= 0:
        raise ValueError(f"max_speakers must be positive int, got {max_speakers}")
    if not token or not isinstance(token, str) or token.strip() == "":
        raise ValueError(
            "pyannote_hf_token is required (non-empty string). "
            "Set env PYANNOTE_HF_TOKEN and pass into prepare_multispeaker_assets()."
        )

    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise ImportError(
            "pyannote.audio is required but not installed/importable. Install it in your current env."
        ) from e

    # 1) load pipeline
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
        pipeline.instantiate({
            "clustering": {
                "min_cluster_size": 1,
                # 可选：更容易分开（值越小越“愿意分裂”，越大越“愿意合并”）
                # 先别瞎调，下面我给你一个调参命令
                "threshold": 0.15,
            }
        })
    except Exception as e:
        raise RuntimeError(
            f"Failed to load pyannote pipeline '{model_id}'. "
            "Likely token permission/terms not accepted or model unavailable. "
            f"Original error: {e}"
        ) from e

    # 2) load audio in-memory (avoid torchcodec)
    file_dict = _load_waveform_in_memory(wav_path)

    # 3) run diarization (API differs across versions)
    try:
        out = pipeline(file_dict, num_speakers=max_speakers)
    except TypeError:
        out = pipeline(file_dict, min_speakers=max_speakers, max_speakers=max_speakers)

    # 4) find Annotation-like object
    ann = _find_annotation_with_itertracks(out)
    if ann is None:
        # helpful debug hint (don’t print everything)
        keys = []
        try:
            keys = list(vars(out).keys())
        except Exception:
            pass
        raise RuntimeError(
            "Unexpected pyannote pipeline output type. "
            f"type(out)={type(out)}; cannot find an Annotation with .itertracks(). "
            f"vars(out).keys()={keys[:50]}"
        )

    # 5) collect speaker durations
    spk_dur: Dict[str, float] = {}
    for turn, _, speaker in ann.itertracks(yield_label=True):
        spk_dur[speaker] = spk_dur.get(speaker, 0.0) + float(turn.end - turn.start)

    spk_labels = sorted(spk_dur.keys(), key=lambda s: spk_dur[s], reverse=True)[:max_speakers]
    spk2person = {s: f"person{i+1}" for i, s in enumerate(spk_labels)}

    # Always include all persons
    segs: Dict[str, List[List[float]]] = {f"person{i+1}": [] for i in range(max_speakers)}

    for turn, _, speaker in ann.itertracks(yield_label=True):
        if speaker not in spk2person:
            continue
        p = spk2person[speaker]
        segs[p].append([float(turn.start), float(turn.end)])

    # 6) sort + merge overlaps per person
    for p in list(segs.keys()):
        xs = segs[p]
        if not xs:
            continue
        xs.sort(key=lambda t: (t[0], t[1]))
        merged = []
        cur_s, cur_e = xs[0]
        for s, e in xs[1:]:
            if s <= cur_e + 1e-3:
                cur_e = max(cur_e, e)
            else:
                merged.append([cur_s, cur_e])
                cur_s, cur_e = s, e
        merged.append([cur_s, cur_e])
        segs[p] = merged

    return segs


# ---------------------------
# Audio rendering: aligned per-speaker tracks (IMPORTANT)
# ---------------------------
def _sanitize_segments(
    segments: List[List[float]],
    total_duration_s: float,
    min_len_s: float = 0.03,
) -> List[List[float]]:
    out = []
    for s, e in segments:
        try:
            s = float(s)
            e = float(e)
        except Exception:
            continue
        s = max(0.0, min(total_duration_s, s))
        e = max(0.0, min(total_duration_s, e))
        if e - s >= min_len_s:
            out.append([s, e])
    return out


def render_aligned_speaker_track_ffmpeg(
    full_wav: str,
    segments: List[List[float]],
    out_wav: str,
    sample_rate: int = 16000,
    total_duration_s: Optional[float] = None,
) -> str:
    _which("ffmpeg")
    out_wav = str(Path(out_wav))

    if total_duration_s is None:
        total_duration_s = _ffprobe_duration_seconds(full_wav)
    total_duration_s = max(0.0, float(total_duration_s))

    segs = _sanitize_segments(segments, total_duration_s=total_duration_s)

    if total_duration_s <= 0:
        cmd = [
            "ffmpeg", "-y",
            "-i", full_wav,
            "-ac", "1",
            "-ar", str(sample_rate),
            out_wav,
            "-loglevel", "error",
        ]
        _run(cmd)
        return out_wav

    if len(segs) == 0:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(total_duration_s),
            "-ac", "1",
            "-ar", str(sample_rate),
            out_wav,
            "-loglevel", "error",
        ]
        _run(cmd)
        return out_wav

    filters = [f"anullsrc=r={sample_rate}:cl=mono:d={total_duration_s}[sil]"]
    delayed_labels = []

    for i, (s, e) in enumerate(segs):
        d_ms = int(round(s * 1000.0))
        lbl = f"seg{i}"
        filters.append(
            f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS,adelay={d_ms}|{d_ms}[{lbl}]"
        )
        delayed_labels.append(f"[{lbl}]")

    if len(delayed_labels) == 1:
        filters.append(f"{delayed_labels[0]}acopy[spk]")
    else:
        filters.append(f"{''.join(delayed_labels)}amix=inputs={len(delayed_labels)}:normalize=0[spk]")

    filters.append("[sil][spk]amix=inputs=2:normalize=0[outa]")
    fc = ";".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", full_wav,
        "-filter_complex", fc,
        "-map", "[outa]",
        "-ac", "1",
        "-ar", str(sample_rate),
        out_wav,
        "-loglevel", "error",
    ]
    _run(cmd)
    return out_wav


# ---------------------------
# BBox estimation (opencv optional else stripes)
# ---------------------------
def bbox_fallback_stripes(width: int, height: int, max_speakers: int, margin: float = 0.05) -> Dict[str, List[float]]:
    bbox = {}
    x_margin = int(width * margin)
    y_margin = int(height * margin)
    stripe_w = width / max_speakers
    for i in range(max_speakers):
        x1 = int(i * stripe_w + x_margin)
        x2 = int((i + 1) * stripe_w - x_margin) if i < max_speakers - 1 else int(width - x_margin)
        y1 = y_margin
        y2 = int(height - y_margin)
        bbox[f"person{i+1}"] = [float(x1), float(y1), float(x2), float(y2)]
    return bbox


def bbox_from_video_opencv_optional(
    media_path: str,
    max_speakers: int,
    sample_frames: int = 30,
) -> Optional[Tuple[Dict[str, List[float]], int, int, float]]:
    try:
        import cv2
    except Exception:
        return None

    w, h, fps = ffprobe_video_info(media_path)
    if w is None or h is None:
        return None
    if fps is None:
        fps = 25.0

    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return None

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        cap.release()
        return None

    idxs = np.linspace(0, max(0, total_frames - 1), num=min(sample_frames, total_frames), dtype=np.int32)

    boxes = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (x, y, fw, fh) in faces:
            x1, y1, x2, y2 = float(x), float(y), float(x + fw), float(y + fh)
            cx = (x1 + x2) / 2.0
            boxes.append([cx, x1, y1, x2, y2])

    cap.release()
    if len(boxes) == 0:
        return None

    arr = np.array(boxes, dtype=np.float32)  # [N,5] => cx,x1,y1,x2,y2

    cxs = np.sort(arr[:, 0])
    if max_speakers == 1:
        groups = [arr]
    else:
        qs = [np.quantile(cxs, (i + 1) / max_speakers) for i in range(max_speakers - 1)]
        tmp = [[] for _ in range(max_speakers)]
        for row in arr:
            cx = row[0]
            gi = 0
            while gi < len(qs) and cx > qs[gi]:
                gi += 1
            tmp[gi].append(row)
        groups = [np.stack(g, axis=0) if len(g) else None for g in tmp]

    bbox = {}
    stripes = bbox_fallback_stripes(w, h, max_speakers)
    for i in range(max_speakers):
        person = f"person{i+1}"
        g = groups[i]
        if g is None:
            bbox[person] = stripes[person]
            continue
        mean = g[:, 1:].mean(axis=0)  # x1,y1,x2,y2
        bbox[person] = [float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3])]

    items = []
    for p, b in bbox.items():
        cx = (b[0] + b[2]) / 2.0
        items.append((cx, p, b))
    items.sort(key=lambda x: x[0])
    bbox_sorted = {}
    for i, (_, _, b) in enumerate(items):
        bbox_sorted[f"person{i+1}"] = b

    return bbox_sorted, int(w), int(h), float(fps)


# ---------------------------
# Main API
# ---------------------------
def prepare_multispeaker_assets(
    media_path: str,
    out_dir: str,
    max_speakers: int = 2,
    sample_rate: int = 16000,
    target_fps: int = 25,
    pyannote_hf_token: Optional[str] = None,
    pyannote_model_id: str = "pyannote/speaker-diarization-3.1",
) -> MultiSpeakerMeta:
    """
    Only pyannote diarization. No fallback.

    Required:
      - pyannote.audio installed
      - token provided (env PYANNOTE_HF_TOKEN recommended)
      - HF model access granted
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    media_path = str(media_path)
    extracted_wav = str(out_dir / "full_audio.wav")
    extract_audio_to_wav(media_path, extracted_wav, sample_rate=sample_rate)

    total_dur = _ffprobe_duration_seconds(extracted_wav)

    # diarization (required)
    segments = diarize_pyannote_required(
        extracted_wav,
        max_speakers=max_speakers,
        token=pyannote_hf_token,
        model_id=pyannote_model_id,
    )

    # bbox
    bbox_info = bbox_from_video_opencv_optional(media_path, max_speakers=max_speakers)
    if bbox_info is None:
        w, h, fps = ffprobe_video_info(media_path)
        if w is None or h is None:
            w, h = 1, 1
        if fps is None:
            fps = float(target_fps)
        bbox = bbox_fallback_stripes(int(w), int(h), max_speakers)
    else:
        bbox, w, h, fps = bbox_info

    # per-person aligned wav
    cond_audio_wav = {}
    for i in range(max_speakers):
        person = f"person{i+1}"
        segs = segments.get(person, [])
        wav_out = str(out_dir / f"{person}.wav")
        render_aligned_speaker_track_ffmpeg(
            full_wav=extracted_wav,
            segments=segs,
            out_wav=wav_out,
            sample_rate=sample_rate,
            total_duration_s=total_dur,
        )
        cond_audio_wav[person] = wav_out

    meta = MultiSpeakerMeta(
        cond_audio_wav=cond_audio_wav,
        bbox=bbox,
        segments=segments,
        media_path=media_path,
        extracted_audio_wav=extracted_wav,
        fps=float(fps) if fps is not None else float(target_fps),
        width=int(w) if w is not None else None,
        height=int(h) if h is not None else None,
    )

    (out_dir / "multispeaker_meta.json").write_text(
        json.dumps(
            {
                "cond_audio_wav": meta.cond_audio_wav,
                "bbox": meta.bbox,
                "segments": meta.segments,
                "media_path": meta.media_path,
                "extracted_audio_wav": meta.extracted_audio_wav,
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta


def build_pipeline_input_data(
    prompt: str,
    cond_image_path: str,
    multispeaker_meta: MultiSpeakerMeta,
) -> Dict:
    input_data = {
        "prompt": prompt,
        "cond_image": cond_image_path,
        "cond_audio": dict(multispeaker_meta.cond_audio_wav),
        "bbox": dict(multispeaker_meta.bbox),
        "video_audio": multispeaker_meta.extracted_audio_wav,
    }
    return input_data

