import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class MultiSpeakerMeta:
    """
    cond_audio_wav: {"person1": ".../person1.wav", "person2": ".../person2.wav", ...}
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


def extract_audio_to_wav(media_path: str, wav_out: str, sample_rate: int = 16000) -> str:
    _which("ffmpeg")
    wav_out = str(Path(wav_out))
    cmd = [
        "ffmpeg", "-y",
        "-i", media_path,
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
        media_path,
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
# Diarization (runnable fallback)
# ---------------------------
def diarize_fallback_roundrobin(
    wav_path: str,
    max_speakers: int,
    chunk_s: float = 4.0,
    total_duration_s: Optional[float] = None,
) -> Dict[str, List[List[float]]]:
    """
    No external models: split into equal chunks and assign speakers round-robin.
    This is ONLY for "it runs" baseline.
    """
    if total_duration_s is None:
        # ask ffprobe for duration
        _which("ffprobe")
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ]
        dur = subprocess.check_output(cmd).decode("utf-8", errors="ignore").strip()
        total_duration_s = float(dur)

    persons = [f"person{i+1}" for i in range(max_speakers)]
    segs = {p: [] for p in persons}

    n = int(math.ceil(total_duration_s / chunk_s))
    t = 0.0
    for i in range(n):
        p = persons[i % max_speakers]
        s = t
        e = min(total_duration_s, t + chunk_s)
        if e > s + 0.05:
            segs[p].append([float(s), float(e)])
        t += chunk_s
        if t >= total_duration_s:
            break
    return segs


def diarize_pyannote_optional(
    wav_path: str,
    max_speakers: int,
    hf_token: Optional[str] = None,
) -> Optional[Dict[str, List[List[float]]]]:
    """
    Optional: if pyannote.audio is installed and HF token provided.
    Return None if not available.
    """
    try:
        from pyannote.audio import Pipeline
    except Exception:
        return None
    if not hf_token:
        return None

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline(wav_path, num_speakers=max_speakers)

    # map speaker labels to person1..N
    spk_labels = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in spk_labels:
            spk_labels.append(speaker)
    spk_labels = spk_labels[:max_speakers]
    spk2person = {s: f"person{i+1}" for i, s in enumerate(spk_labels)}
    segs = {spk2person[s]: [] for s in spk_labels}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in spk2person:
            continue
        segs[spk2person[speaker]].append([float(turn.start), float(turn.end)])

    return segs


# ---------------------------
# Audio segment cutting/merging
# ---------------------------
def merge_audio_segments_ffmpeg(
    full_wav: str,
    segments: List[List[float]],
    out_wav: str,
    sample_rate: int = 16000,
) -> str:
    """
    Concatenate multiple [start,end] segments into one wav using ffmpeg filter_complex.
    """
    _which("ffmpeg")
    out_wav = str(Path(out_wav))
    if len(segments) == 0:
        # generate empty-ish wav: just copy first 0.1s
        segments = [[0.0, 0.6]]

    # build atrim chains
    filters = []
    labels = []
    for i, (s, e) in enumerate(segments):
        s = max(0.0, float(s))
        e = max(s + 0.6, float(e))
        lbl = f"a{i}"
        filters.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[{lbl}]")
        labels.append(f"[{lbl}]")
    concat_lbl = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[outa]"
    fc = ";".join(filters + [concat_lbl])

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
# BBox estimation (runnable fallback + optional OpenCV)
# ---------------------------
def bbox_fallback_stripes(width: int, height: int, max_speakers: int, margin: float = 0.05) -> Dict[str, List[float]]:
    """
    Split screen into N vertical regions. Return bbox per person.
    """
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
    """
    Optional: use OpenCV Haar cascade to detect faces on sampled frames and produce coarse bbox.
    If OpenCV isn't available or no video stream, return None.

    Strategy:
      - sample frames uniformly
      - run Haar face detector
      - collect detected face boxes, cluster by x center into max_speakers groups
      - return mean bbox for each group, mapped left->right as person1..N
    """
    try:
        import cv2
    except Exception:
        return None

    w, h, fps = ffprobe_video_info(media_path)
    if w is None or h is None:
        return None
    if fps is None:
        fps = 25.0

    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return None

    # Haar cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        cap.release()
        return None

    # sample indices
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

    arr = np.array(boxes, dtype=np.float32)  # [N,5]
    # cluster by cx into K bins using quantiles (cheap + no sklearn)
    cxs = np.sort(arr[:, 0])
    if max_speakers == 1:
        groups = [arr]
    else:
        # quantile split points
        qs = [np.quantile(cxs, (i + 1) / max_speakers) for i in range(max_speakers - 1)]
        groups = [[] for _ in range(max_speakers)]
        for row in arr:
            cx = row[0]
            gi = 0
            while gi < len(qs) and cx > qs[gi]:
                gi += 1
            groups[gi].append(row)
        groups = [np.stack(g, axis=0) if len(g) else None for g in groups]

    # mean bbox per group; if empty, fallback stripes
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

    # ensure left->right ordering by bbox center x
    items = []
    for p, b in bbox.items():
        cx = (b[0] + b[2]) / 2.0
        items.append((cx, p, b))
    items.sort(key=lambda x: x[0])
    bbox_sorted = {}
    for i, (_, _, b) in enumerate(items):
        bbox_sorted[f"person{i+1}"] = b

    return bbox_sorted, w, h, fps


# ---------------------------
# Main API
# ---------------------------
def prepare_multispeaker_assets(
    media_path: str,
    out_dir: str,
    max_speakers: int = 2,
    sample_rate: int = 16000,
    target_fps: int = 25,
    use_pyannote: bool = False,
    pyannote_hf_token: Optional[str] = None,
    diar_chunk_s: float = 4.0,
) -> MultiSpeakerMeta:
    """
    Input: mp4 / wav / any ffmpeg-readable media with audio.
    Output: MultiSpeakerMeta including per-person wav and bbox.

    - diarization:
        * if use_pyannote & token & installed => pyannote diarization
        * else fallback round-robin chunk split (runnable baseline)
    - bbox:
        * if video stream and OpenCV available => Haar face detect -> coarse bbox
        * else fallback stripes bbox
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    media_path = str(media_path)
    extracted_wav = str(out_dir / "full_audio.wav")
    extract_audio_to_wav(media_path, extracted_wav, sample_rate=sample_rate)

    # diarization
    segments = None
    if use_pyannote:
        segments = diarize_pyannote_optional(extracted_wav, max_speakers=max_speakers, hf_token=pyannote_hf_token)
    if segments is None:
        segments = diarize_fallback_roundrobin(
            extracted_wav, max_speakers=max_speakers, chunk_s=diar_chunk_s, total_duration_s=None
        )

    # bbox
    bbox_info = bbox_from_video_opencv_optional(media_path, max_speakers=max_speakers)
    if bbox_info is None:
        # no cv2 or no video -> probe w/h if possible, else use fake 1x1 (pipeline will still run)
        w, h, fps = ffprobe_video_info(media_path)
        if w is None or h is None:
            w, h = 1, 1
        if fps is None:
            fps = float(target_fps)
        bbox = bbox_fallback_stripes(w, h, max_speakers)
    else:
        bbox, w, h, fps = bbox_info

    # per-person wav (merge segments)
    cond_audio_wav = {}
    for i in range(max_speakers):
        person = f"person{i+1}"
        segs = segments.get(person, [])
        wav_out = str(out_dir / f"{person}.wav")
        merge_audio_segments_ffmpeg(extracted_wav, segs, wav_out, sample_rate=sample_rate)
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

    (out_dir / "multispeaker_meta.json").write_text(json.dumps({
        "cond_audio_wav": meta.cond_audio_wav,
        "bbox": meta.bbox,
        "segments": meta.segments,
        "media_path": meta.media_path,
        "extracted_audio_wav": meta.extracted_audio_wav,
        "fps": meta.fps,
        "width": meta.width,
        "height": meta.height,
    }, indent=2), encoding="utf-8")
    return meta


def build_pipeline_input_data(
    prompt: str,
    cond_image_path: str,
    multispeaker_meta: MultiSpeakerMeta,
) -> Dict:
    """
    Build input_data for preprocess_audio + pipeline.
    Here cond_audio points to WAVs; preprocess_audio() will convert them to .pt embeddings.
    """
    input_data = {
        "prompt": prompt,
        "cond_image": cond_image_path,
        "cond_audio": dict(multispeaker_meta.cond_audio_wav),  # wavs
        "bbox": dict(multispeaker_meta.bbox),
        # optional: keep full audio for mux later if your preprocess_audio writes video_audio anyway
        "video_audio": multispeaker_meta.extracted_audio_wav,
    }
    return input_data

