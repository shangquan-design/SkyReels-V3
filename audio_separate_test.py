import os
import json
import math
import argparse
import numpy as np
import soundfile as sf
import torch


# ============================================================
# Basic utils
# ============================================================
def to_mono_1d(waveform: np.ndarray) -> np.ndarray:
    """Return mono waveform as shape (T,) float32."""
    w = np.asarray(waveform)
    if w.ndim == 1:
        return w.astype(np.float32).reshape(-1)
    if w.ndim != 2:
        return w.astype(np.float32).reshape(-1)

    # (T,1) or (1,T)
    if w.shape[1] == 1:
        return w[:, 0].astype(np.float32).reshape(-1)
    if w.shape[0] == 1:
        return w[0, :].astype(np.float32).reshape(-1)

    # (T,C) vs (C,T)
    if w.shape[0] > w.shape[1]:
        return w.mean(axis=1).astype(np.float32).reshape(-1)  # (T,C)
    else:
        return w.mean(axis=0).astype(np.float32).reshape(-1)  # (C,T)


def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


# ============================================================
# BGM reduction: stereo mid + spectral gate (numpy)
# ============================================================
def _stft_np(x: np.ndarray, n_fft: int, hop: int, win_length: int):
    x = x.astype(np.float32).reshape(-1)
    if win_length > n_fft:
        win_length = n_fft
    win = np.hanning(win_length).astype(np.float32)
    if win_length < n_fft:
        pad = n_fft - win_length
        win = np.pad(win, (0, pad), mode="constant")

    # pad so that we have at least one frame
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)), mode="constant")

    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frames[i, :] = x[s:s+n_fft]
    frames = frames * win[None, :]
    X = np.fft.rfft(frames, axis=1)  # (frames, freq)
    return X, win


def _istft_np(X: np.ndarray, hop: int, win: np.ndarray, length: int):
    n_frames, n_freq = X.shape
    n_fft = (n_freq - 1) * 2
    frames = np.fft.irfft(X, n=n_fft, axis=1).astype(np.float32)  # (frames, n_fft)
    frames = frames * win[None, :]

    out_len = n_fft + hop * (n_frames - 1)
    y = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)

    for i in range(n_frames):
        s = i * hop
        y[s:s+n_fft] += frames[i]
        wsum[s:s+n_fft] += (win * win)

    y = y / (wsum + 1e-8)
    return y[:length].astype(np.float32)


def reduce_bgm(
    waveform: np.ndarray,
    sr: int,
    mode: str = "auto",
    stereo_side_keep: float = 0.0,
    n_fft: int = 1024,
    hop: int = 256,
    win_length: int = 1024,
    noise_quantile: float = 0.20,
    strength: float = 1.0,
    mask_power: float = 1.0,
    mask_smooth: int = 3,
    debug: bool = False,
):
    """
    mode:
      - auto: if stereo -> stereo_mid then spectral_gate; else spectral_gate
      - stereo_mid: use mid/side (keep mid, attenuate side)
      - spectral_gate: estimate noise profile from low-energy frames, apply soft mask
      - off: do nothing
    """
    if mode == "off":
        return to_mono_1d(waveform).astype(np.float32)

    w = np.asarray(waveform)

    # 1) stereo mid/side
    x = None
    is_stereo = (w.ndim == 2 and (w.shape[1] >= 2 or w.shape[0] >= 2))
    if mode in ("auto", "stereo_mid") and is_stereo:
        # normalize to (T,C)
        if w.shape[0] < w.shape[1]:
            w_tc = w.T
        else:
            w_tc = w
        L = w_tc[:, 0].astype(np.float32)
        R = w_tc[:, 1].astype(np.float32)
        mid = 0.5 * (L + R)
        side = 0.5 * (L - R)
        x = (mid + float(stereo_side_keep) * side).astype(np.float32)
        if debug:
            print(f"[BGM] stereo_mid applied. side_keep={stereo_side_keep:.2f}")
    else:
        x = to_mono_1d(w).astype(np.float32)

    # 2) spectral gate (always helpful if BGM exists)
    if mode in ("auto", "spectral_gate", "stereo_mid"):
        X, win = _stft_np(x, n_fft=n_fft, hop=hop, win_length=win_length)
        mag = np.abs(X).astype(np.float32)
        phase = X / (mag + 1e-12)

        # frame energy for selecting "noise" frames
        frame_energy = np.sqrt(np.mean(mag * mag, axis=1) + 1e-12)  # (frames,)
        q = float(np.quantile(frame_energy, clip(noise_quantile, 0.05, 0.50)))
        noise_frames = frame_energy <= q
        if not np.any(noise_frames):
            # fallback: use lowest 10%
            k = max(1, int(0.10 * len(frame_energy)))
            idx = np.argsort(frame_energy)[:k]
            noise_frames = np.zeros_like(frame_energy, dtype=bool)
            noise_frames[idx] = True

        noise_profile = np.median(mag[noise_frames], axis=0)  # (freq,)
        # soft subtraction
        mag_d = np.maximum(mag - float(strength) * noise_profile[None, :], 0.0)
        mask = mag_d / (mag + 1e-12)
        if mask_power != 1.0:
            mask = np.power(mask, float(mask_power))

        # smooth mask over time (cheap)
        if mask_smooth > 1:
            k = int(mask_smooth)
            kernel = np.ones(k, dtype=np.float32) / k
            # convolve along time for each freq bin
            mask_s = np.zeros_like(mask)
            for f in range(mask.shape[1]):
                mask_s[:, f] = np.convolve(mask[:, f], kernel, mode="same")
            mask = mask_s

        Y = (mag * mask) * phase
        y = _istft_np(Y, hop=hop, win=win, length=len(x))
        y = np.clip(y, -1.0, 1.0).astype(np.float32)

        if debug:
            kept = float(np.mean(mask))
            print(f"[BGM] spectral_gate applied. noise_q={noise_quantile:.2f} strength={strength:.2f} "
                  f"mask_power={mask_power:.2f} avg_mask={kept:.3f}")

        return y

    return x.astype(np.float32)


def clip(x: float, lo: float, hi: float):
    return max(lo, min(hi, x))


# ============================================================
# Preprocess: resample/bandpass/rms norm (+ optional BGM reduce)
# ============================================================
def preprocess_audio(
    waveform: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    seed: int = 0,
    bgm_reduce_flag: bool = False,
    bgm_mode: str = "auto",
    bgm_side_keep: float = 0.0,
    bgm_n_fft: int = 1024,
    bgm_hop: int = 256,
    bgm_win: int = 1024,
    bgm_noise_q: float = 0.20,
    bgm_strength: float = 1.0,
    bgm_mask_power: float = 1.0,
    bgm_mask_smooth: int = 3,
    debug: bool = False,
):
    rng = np.random.default_rng(seed)

    # optional bgm reduce BEFORE mono/DC/rms, so stereo mid can help
    x = waveform
    if bgm_reduce_flag:
        x = reduce_bgm(
            waveform=x,
            sr=sr,
            mode=bgm_mode,
            stereo_side_keep=bgm_side_keep,
            n_fft=bgm_n_fft,
            hop=bgm_hop,
            win_length=bgm_win,
            noise_quantile=bgm_noise_q,
            strength=bgm_strength,
            mask_power=bgm_mask_power,
            mask_smooth=bgm_mask_smooth,
            debug=debug,
        )

    x = to_mono_1d(x)

    # DC remove
    x = x - float(np.mean(x))

    # tiny dither
    x = x + (1e-6 * rng.standard_normal(x.shape).astype(np.float32))

    # resample if needed
    if sr != target_sr:
        try:
            import torchaudio
            xt = torch.from_numpy(x)[None, :]
            xt = torchaudio.functional.resample(xt, sr, target_sr)
            x = xt.squeeze(0).cpu().numpy().astype(np.float32)
            sr = target_sr
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Need torchaudio for resampling (sr={sr} -> {target_sr}). "
                f"Install torchaudio or provide already-{target_sr}Hz audio. Error: {e}"
            )

    # bandpass (optional)
    try:
        import torchaudio
        xt = torch.from_numpy(x)[None, :]
        xt = torchaudio.functional.highpass_biquad(xt, sr, cutoff_freq=80.0)
        xt = torchaudio.functional.lowpass_biquad(xt, sr, cutoff_freq=7500.0)
        x = xt.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        pass

    # RMS normalize
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    target_rms = 0.10
    x = x * (target_rms / (rms + 1e-12))
    x = np.clip(x, -1.0, 1.0)

    return x.astype(np.float32).reshape(-1), sr


# ============================================================
# Energy VAD segmentation (robust)
# ============================================================
def segment_by_energy(
    x: np.ndarray,
    sr: int,
    frame_ms=20,
    hop_ms=10,
    min_seg_ms=180,
    max_silence_ms=200,
    pad_ms=20,
    thr_mad_k=3.5,
    thr_floor=1e-6,
    debug: bool = False,
):
    x = to_mono_1d(x)
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame:
        return []

    n = 1 + max(0, (len(x) - frame) // hop)
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = x[s : s + frame]
        rms[i] = np.sqrt(np.mean(w * w) + 1e-12)

    # smooth
    k = 5
    rms_s = np.convolve(rms, np.ones(k) / k, mode="same") if len(rms) >= k else rms

    # threshold
    med = float(np.median(rms_s))
    mad = float(np.median(np.abs(rms_s - med)) + 1e-12)
    thr1 = med + thr_mad_k * mad

    p10 = float(np.percentile(rms_s, 10))
    p90 = float(np.percentile(rms_s, 90))
    thr2 = p10 + 0.25 * max(0.0, (p90 - p10))

    thr = max(min(thr1, thr2), thr_floor)
    speech = rms_s > thr

    # relax progressively if empty
    used_thr = thr
    if not speech.any():
        for alpha in [0.15, 0.10, 0.07, 0.05, 0.03]:
            thr = max(p10 + alpha * max(0.0, (p90 - p10)), thr_floor)
            speech = rms_s > thr
            used_thr = thr
            if speech.any():
                break

    if debug:
        ratio = float(np.mean(speech)) if speech.size else 0.0
        print(f"[VAD] p10={p10:.6f} p90={p90:.6f} med={med:.6f} mad={mad:.6f} "
              f"thr1={thr1:.6f} thr2={thr2:.6f} thr_used={used_thr:.6f} speech_ratio={ratio:.3f}")

    if not speech.any():
        return []

    # closing: fill short gaps
    max_gap = int(max_silence_ms / hop_ms)
    if max_gap > 0:
        idx = np.where(speech)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 0 < (b - a) <= max_gap:
                speech[a : b + 1] = True

    segments = []
    min_len = int(min_seg_ms / hop_ms)
    i = 0
    while i < len(speech):
        if not speech[i]:
            i += 1
            continue
        j = i
        while j < len(speech) and speech[j]:
            j += 1
        if (j - i) >= min_len:
            start = i * hop
            end = min(len(x), j * hop + frame)
            pad = int(sr * pad_ms / 1000)
            start = max(0, start - pad)
            end = min(len(x), end + pad)
            segments.append((start / sr, end / sr))
        i = j

    return sorted(segments, key=lambda t: t[0])


def merge_by_gap(segments, gap_s=0.25):
    if not segments:
        return []
    segs = sorted(segments, key=lambda t: t[0])
    out = [segs[0]]
    for (s, e) in segs[1:]:
        ps, pe = out[-1]
        if s - pe <= gap_s:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def trim_leading_trailing_silence(x_seg: np.ndarray, sr: int, frame_ms=20, hop_ms=10, thr_mad_k=2.8):
    x_seg = to_mono_1d(x_seg)
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x_seg) < frame:
        return x_seg

    n = 1 + (len(x_seg) - frame) // hop
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = x_seg[s : s + frame]
        rms[i] = np.sqrt(np.mean(w * w) + 1e-12)

    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med)) + 1e-12)
    thr = max(med + thr_mad_k * mad, 1e-6)

    speech = rms > thr
    if not speech.any():
        return x_seg

    i0 = int(np.argmax(speech))
    i1 = int(len(speech) - 1 - np.argmax(speech[::-1]))
    start = max(0, i0 * hop)
    end = min(len(x_seg), i1 * hop + frame)
    return x_seg[start:end]


# ============================================================
# Speaker-ish features
# ============================================================
def spectral_features_numpy(x: np.ndarray, sr: int, frame_ms=25, hop_ms=10, rolloff=0.85):
    x = to_mono_1d(x)
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame:
        return 0.0, 0.0, 0.0

    n = 1 + (len(x) - frame) // hop
    win = np.hanning(frame).astype(np.float32)

    cents, rolls, zcrs = [], [], []
    freqs = np.fft.rfftfreq(frame, d=1.0 / sr).astype(np.float32)

    for i in range(n):
        s = i * hop
        w = x[s : s + frame] * win
        mag = np.abs(np.fft.rfft(w)).astype(np.float32) + 1e-12

        centroid = float((mag * freqs).sum() / (mag.sum() + 1e-12))
        cumsum = np.cumsum(mag)
        thr = rolloff * cumsum[-1]
        idx = int(np.searchsorted(cumsum, thr))
        idx = min(idx, len(freqs) - 1)
        roll = float(freqs[idx])

        zc = float(np.mean(np.abs(np.diff(np.sign(w))) > 0))
        cents.append(centroid)
        rolls.append(roll)
        zcrs.append(zc)

    return float(np.mean(cents)), float(np.mean(rolls)), float(np.mean(zcrs))


def estimate_pitch_autocorr(x: np.ndarray, sr: int, fmin=60, fmax=900):
    x = to_mono_1d(x)
    if len(x) < int(0.05 * sr):
        return 0.0
    mid = x[len(x) // 4 : 3 * len(x) // 4]
    mid = mid - float(np.mean(mid))
    mid = mid * np.hanning(len(mid)).astype(np.float32)

    ac = np.correlate(mid, mid, mode="full")[len(mid) - 1 :]
    if ac[0] <= 1e-9:
        return 0.0

    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, len(ac) - 1)
    if lag_max <= lag_min:
        return 0.0

    seg = ac[lag_min:lag_max]
    lag = int(np.argmax(seg) + lag_min)
    if lag <= 0:
        return 0.0
    return float(sr / lag)


def estimate_pitch(x: np.ndarray, sr: int):
    try:
        import torchaudio
        wav = torch.from_numpy(to_mono_1d(x)).float().unsqueeze(0)
        pitch = torchaudio.functional.detect_pitch_frequency(
            wav,
            sample_rate=sr,
            frame_time=0.02,
            win_length=int(0.02 * sr),
            hop_length=int(0.01 * sr),
        ).squeeze(0).cpu().numpy()
        pitch = pitch[(pitch > 50) & (pitch < 900)]
        if pitch.size > 0:
            return float(np.median(pitch))
    except Exception:
        pass

    try:
        import librosa
        f0 = librosa.yin(
            to_mono_1d(x).astype(np.float32),
            fmin=50,
            fmax=900,
            sr=sr,
            frame_length=int(0.04 * sr),
            hop_length=int(0.01 * sr),
        )
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 50) & (f0 < 900)]
        if f0.size > 0:
            return float(np.median(f0))
    except Exception:
        pass

    return estimate_pitch_autocorr(x, sr)


def extract_mfcc_if_available(x: np.ndarray, sr: int, n_mfcc=20):
    x = to_mono_1d(x)

    try:
        import torchaudio
        import torchaudio.transforms as T
        wav = torch.from_numpy(x).float().unsqueeze(0)
        mfcc = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 512, "hop_length": int(0.01 * sr), "n_mels": 40},
        )(wav)
        mfcc = mfcc.squeeze(0).transpose(0, 1).cpu().numpy()
        mfcc_mean = mfcc.mean(axis=0)
        mfcc_std = mfcc.std(axis=0)
        return np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)
    except Exception:
        pass

    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=x.astype(np.float32),
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=int(0.01 * sr),
        ).T
        mfcc_mean = mfcc.mean(axis=0)
        mfcc_std = mfcc.std(axis=0)
        return np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)
    except Exception:
        return None


def extract_features(x_seg: np.ndarray, sr: int, n_mfcc=20):
    x_seg = to_mono_1d(x_seg)
    mfcc = extract_mfcc_if_available(x_seg, sr, n_mfcc=n_mfcc)

    pitch = estimate_pitch(x_seg, sr)
    pitch_log = np.float32(math.log(pitch + 1.0))
    centroid, roll, zcr = spectral_features_numpy(x_seg, sr)

    extra = np.array([pitch_log, np.float32(centroid), np.float32(roll), np.float32(zcr)], dtype=np.float32)
    if mfcc is None:
        return extra
    return np.concatenate([mfcc, extra], axis=0).astype(np.float32)


def cosine_dist(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


# distance weights (set in main)
_DIST_W = {"cos": 1.0, "pitch": 2.5, "spec": 0.5}


def seg_distance(f1: np.ndarray, f2: np.ndarray):
    pitch1, pitch2 = float(f1[-4]), float(f2[-4])
    v1 = f1[:-4] if f1.shape[0] > 4 else np.zeros((1,), dtype=np.float32)
    v2 = f2[:-4] if f2.shape[0] > 4 else np.zeros((1,), dtype=np.float32)

    cos_d = cosine_dist(v1, v2) if (v1.size == v2.size and v1.size > 1) else 0.0
    pitch_d = abs(pitch1 - pitch2)
    spec_d = float(np.linalg.norm((f1[-3:] - f2[-3:]).astype(np.float32)))
    return _DIST_W["cos"] * cos_d + _DIST_W["pitch"] * pitch_d + _DIST_W["spec"] * spec_d


def zscore_features(F: np.ndarray):
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-6
    return (F - mu) / sd


# ============================================================
# Clustering: centroid agglomeration + auto threshold
# ============================================================
def cluster_by_threshold(F: np.ndarray, thr: float, min_cluster_size: int = 1):
    n = F.shape[0]
    clusters = [[i] for i in range(n)]
    centroids = [F[i].copy() for i in range(n)]

    def cdist(ci, cj):
        return seg_distance(centroids[ci], centroids[cj])

    while True:
        m = len(clusters)
        if m <= 1:
            break
        best = (1e18, -1, -1)
        for i in range(m):
            for j in range(i + 1, m):
                d = cdist(i, j)
                if d < best[0]:
                    best = (d, i, j)
        if best[0] >= thr or best[1] < 0:
            break

        _, i, j = best
        clusters[i].extend(clusters[j])
        centroids[i] = F[clusters[i]].mean(axis=0)
        del clusters[j]
        del centroids[j]

    labels = np.zeros(n, dtype=np.int32)
    for k, members in enumerate(clusters):
        for idx in members:
            labels[idx] = k

    # merge tiny clusters into nearest big cluster
    if min_cluster_size > 1:
        sizes = np.bincount(labels)
        big = [i for i, s in enumerate(sizes) if s >= min_cluster_size]
        small = [i for i, s in enumerate(sizes) if 0 < s < min_cluster_size]

        if big and small:
            big_cent = {c: F[labels == c].mean(axis=0) for c in big}
            for sc in small:
                sc_idx = np.where(labels == sc)[0]
                sc_cent = F[sc_idx].mean(axis=0)
                best_c, best_d = None, 1e18
                for bc in big:
                    d = seg_distance(sc_cent, big_cent[bc])
                    if d < best_d:
                        best_d = d
                        best_c = bc
                labels[sc_idx] = best_c

            uniq = sorted(list(set(labels.tolist())))
            remap = {u: i for i, u in enumerate(uniq)}
            labels = np.array([remap[int(x)] for x in labels], dtype=np.int32)

    return labels


def clustering_separation_score(F: np.ndarray, labels: np.ndarray):
    n = F.shape[0]
    if n <= 1:
        return 0.0
    uniq = sorted(list(set(labels.tolist())))
    if len(uniq) <= 1:
        return 0.0

    intra = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue
        cent = F[idx].mean(axis=0)
        d = [seg_distance(F[i], cent) for i in idx]
        intra.append(float(np.mean(d)))
    intra_mean = float(np.mean(intra)) if intra else 0.0

    cents = [F[labels == c].mean(axis=0) for c in uniq]
    inter = []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            inter.append(seg_distance(cents[i], cents[j]))
    inter_mean = float(np.mean(inter)) if inter else 0.0

    return (inter_mean - intra_mean) / (abs(inter_mean) + 1e-6)


def auto_cluster(F: np.ndarray, min_speakers=1, max_speakers=10, min_cluster_size=1, debug=False):
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32), {"chosen_thr": None, "num_speakers": 0}
    if n == 1:
        return np.array([0], dtype=np.int32), {"chosen_thr": None, "num_speakers": 1}

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(seg_distance(F[i], F[j]))
    dists = np.array(dists, dtype=np.float32)
    if dists.size == 0:
        return np.zeros(n, dtype=np.int32), {"chosen_thr": None, "num_speakers": 1}

    #percs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    percs = [55]
    cand = sorted(list(set([float(np.percentile(dists, p)) for p in percs])))
    if not cand:
        cand = [float(np.median(dists))]

    best = {"score": -1e18, "thr": cand[0], "labels": None, "k": 1}

    for thr in cand:
        labels = cluster_by_threshold(F, thr=thr, min_cluster_size=min_cluster_size)
        k = len(set(labels.tolist()))
        if k < min_speakers or k > max_speakers:
            continue
        score = clustering_separation_score(F, labels)

        if score > best["score"] + 1e-4 or (abs(score - best["score"]) <= 1e-4 and k < best["k"]):
            best = {"score": score, "thr": thr, "labels": labels, "k": k}

        if debug:
            print(f"[AUTO] thr={thr:.4f} -> K={k} score={score:.4f}")

    if best["labels"] is None:
        labels = np.zeros(n, dtype=np.int32)
        return labels, {"chosen_thr": None, "num_speakers": 1}

    return best["labels"], {"chosen_thr": best["thr"], "num_speakers": best["k"], "score": best["score"]}


# ============================================================
# Label remap for "role assignment" (speaker ordering)
# ============================================================
def remap_labels_by_order(segments, labels, feats, order_rule: str):
    """
    order_rule:
      - first: by first appearance
      - high_pitch: higher mean pitch -> speaker_00
      - low_pitch:  lower mean pitch  -> speaker_00
      - longest_duration: larger total duration -> speaker_00
    """
    labels = np.array(labels, dtype=np.int32)
    uniq = sorted(list(set(labels.tolist())))
    if len(uniq) <= 1:
        return labels

    # stats per cluster
    pitch = np.array([float(f[-4]) for f in feats], dtype=np.float32)  # log_pitch
    dur = np.array([float(te - ts) for (ts, te) in segments], dtype=np.float32)

    stats = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        p = float(np.mean(pitch[idx])) if idx.size else 0.0
        d = float(np.sum(dur[idx])) if idx.size else 0.0
        first_t = float(segments[int(idx[0])][0]) if idx.size else 1e18
        stats.append((c, p, d, first_t))

    if order_rule == "first":
        stats_sorted = sorted(stats, key=lambda x: x[3])  # first_t
    elif order_rule == "high_pitch":
        stats_sorted = sorted(stats, key=lambda x: -x[1])  # pitch desc
    elif order_rule == "low_pitch":
        stats_sorted = sorted(stats, key=lambda x: x[1])   # pitch asc
    else:  # longest_duration
        stats_sorted = sorted(stats, key=lambda x: -x[2])  # dur desc

    remap = {c: i for i, (c, _, _, _) in enumerate(stats_sorted)}
    new_labels = np.array([remap[int(x)] for x in labels], dtype=np.int32)
    return new_labels


# ============================================================
# Post-merge by predicted speaker labels
# ============================================================
def merge_same_speaker(segments, labels, max_gap_s=0.6):
    if not segments:
        return [], []
    segs = list(sorted(list(zip(segments, labels)), key=lambda x: x[0][0]))
    out_segs = [segs[0][0]]
    out_lab = [segs[0][1]]

    for (s, e), lab in segs[1:]:
        ps, pe = out_segs[-1]
        pl = out_lab[-1]
        if lab == pl and (s - pe) <= max_gap_s:
            out_segs[-1] = (ps, max(pe, e))
        else:
            out_segs.append((s, e))
            out_lab.append(lab)
    return out_segs, out_lab


# ============================================================
# Output writers
# ============================================================
def save_segment_wavs(x, sr, segments, speaker_names, out_dir):
    safe_makedirs(out_dir)
    for i, ((ts, te), spk) in enumerate(zip(segments, speaker_names)):
        s = int(ts * sr)
        e = int(te * sr)
        sf.write(os.path.join(out_dir, f"{i:03d}_{spk}_{ts:.2f}-{te:.2f}.wav"), x[s:e], sr)


def save_speaker_tracks(x, sr, segments, speaker_names, out_dir, silence_gap_ms=200):
    safe_makedirs(out_dir)
    x = to_mono_1d(x)

    uniq = sorted(list(set(speaker_names)))
    tracks = {u: np.zeros_like(x, dtype=np.float32) for u in uniq}
    for (ts, te), spk in zip(segments, speaker_names):
        s = int(ts * sr)
        e = int(te * sr)
        tracks[spk][s:e] = x[s:e]

    for spk in uniq:
        sf.write(os.path.join(out_dir, f"{spk}_timeline.wav"), tracks[spk], sr)

    gap = np.zeros(int(sr * silence_gap_ms / 1000), dtype=np.float32)
    for spk in uniq:
        parts = []
        for (ts, te), sname in zip(segments, speaker_names):
            if sname != spk:
                continue
            s = int(ts * sr)
            e = int(te * sr)
            parts.append(x[s:e])
            parts.append(gap)
        cat = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)
        sf.write(os.path.join(out_dir, f"{spk}_concat.wav"), cat, sr)


def write_rttm(segments, speaker_names, out_path, file_id="audio"):
    with open(out_path, "w", encoding="utf-8") as f:
        for (ts, te), spk in zip(segments, speaker_names):
            dur = max(0.0, te - ts)
            f.write(f"SPEAKER {file_id} 1 {ts:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")


def write_json(segments, speaker_names, out_path):
    items = [{"start": float(ts), "end": float(te), "speaker": spk} for (ts, te), spk in zip(segments, speaker_names)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"segments": items}, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--out", type=str, default="./diar_out")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)

    # BGM reduction
    ap.add_argument("--bgm_reduce", action="store_true")
    ap.add_argument("--bgm_mode", type=str, default="auto", choices=["auto", "stereo_mid", "spectral_gate", "off"])
    ap.add_argument("--bgm_side_keep", type=float, default=0.0)
    ap.add_argument("--bgm_n_fft", type=int, default=1024)
    ap.add_argument("--bgm_hop", type=int, default=256)
    ap.add_argument("--bgm_win", type=int, default=1024)
    ap.add_argument("--bgm_noise_q", type=float, default=0.20)
    ap.add_argument("--bgm_strength", type=float, default=1.0)
    ap.add_argument("--bgm_mask_power", type=float, default=1.0)
    ap.add_argument("--bgm_mask_smooth", type=int, default=3)

    # VAD params
    ap.add_argument("--vad_frame_ms", type=float, default=20.0)
    ap.add_argument("--vad_hop_ms", type=float, default=10.0)
    ap.add_argument("--vad_min_seg_ms", type=float, default=120.0)
    ap.add_argument("--vad_max_silence_ms", type=float, default=200.0)
    ap.add_argument("--vad_pad_ms", type=float, default=40.0)
    ap.add_argument("--vad_thr_mad_k", type=float, default=2.5)
    ap.add_argument("--vad_thr_floor", type=float, default=1e-6)

    # merging
    ap.add_argument("--pre_merge_gap_ms", type=float, default=250.0)
    ap.add_argument("--post_merge_gap_ms", type=float, default=600.0)

    # clustering controls
    ap.add_argument("--min_speakers", type=int, default=1)
    ap.add_argument("--max_speakers", type=int, default=2)
    ap.add_argument("--min_cluster_size", type=int, default=2)
    ap.add_argument("--num_speakers", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.000002)

    # distance weights (help fix over/under clustering)
    ap.add_argument("--w_cos", type=float, default=1.0)
    ap.add_argument("--w_pitch", type=float, default=2.5)
    ap.add_argument("--w_spec", type=float, default=0.5)

    # speaker naming / role assignment
    ap.add_argument("--speaker_order", type=str, default="first",
                    choices=["first", "high_pitch", "low_pitch", "longest_duration"])

    # outputs
    ap.add_argument("--write_rttm", action="store_true")
    ap.add_argument("--write_json", action="store_true")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    global _DIST_W
    _DIST_W = {"cos": float(args.w_cos), "pitch": float(args.w_pitch), "spec": float(args.w_spec)}

    safe_makedirs(args.out)

    wav, sr0 = sf.read(args.audio, always_2d=False, dtype="float32")

    x, sr = preprocess_audio(
        wav, sr0, target_sr=args.sr, seed=args.seed,
        bgm_reduce_flag=args.bgm_reduce,
        bgm_mode=args.bgm_mode,
        bgm_side_keep=args.bgm_side_keep,
        bgm_n_fft=args.bgm_n_fft,
        bgm_hop=args.bgm_hop,
        bgm_win=args.bgm_win,
        bgm_noise_q=args.bgm_noise_q,
        bgm_strength=args.bgm_strength,
        bgm_mask_power=args.bgm_mask_power,
        bgm_mask_smooth=args.bgm_mask_smooth,
        debug=args.debug,
    )

    if args.debug:
        print(f"[DEBUG] sr={sr}, len={len(x)}, sec={len(x)/sr:.2f}, "
              f"min={x.min():.3f}, max={x.max():.3f}, rms={np.sqrt(np.mean(x*x)+1e-12):.6f}")
        print(f"[DEBUG] dist_w: cos={_DIST_W['cos']}, pitch={_DIST_W['pitch']}, spec={_DIST_W['spec']}")

    # 1) VAD
    segments = segment_by_energy(
        x, sr,
        frame_ms=args.vad_frame_ms,
        hop_ms=args.vad_hop_ms,
        min_seg_ms=args.vad_min_seg_ms,
        max_silence_ms=args.vad_max_silence_ms,
        pad_ms=args.vad_pad_ms,
        thr_mad_k=args.vad_thr_mad_k,
        thr_floor=args.vad_thr_floor,
        debug=args.debug,
    )

    print("VAD segments (raw):")
    for i, (ts, te) in enumerate(segments):
        print(f"  [{i:03d}] {ts:.2f}s - {te:.2f}s  dur={te-ts:.2f}s")

    if not segments:
        print("[FAIL] No speech segments found.")
        print("Try: --bgm_reduce  and/or  lower --vad_thr_mad_k (e.g. 2.5)  and/or  lower --vad_min_seg_ms.")
        return

    # 2) pre-merge
    segments = merge_by_gap(segments, gap_s=args.pre_merge_gap_ms / 1000.0)
    if args.debug:
        print(f"\nAfter pre-merge (gap<={args.pre_merge_gap_ms}ms): {len(segments)} segments")
        for i, (ts, te) in enumerate(segments):
            print(f"  [{i:03d}] {ts:.2f}s - {te:.2f}s  dur={te-ts:.2f}s")

    # 3) features
    seg_wavs = []
    for (ts, te) in segments:
        s = int(ts * sr)
        e = int(te * sr)
        seg = trim_leading_trailing_silence(x[s:e], sr)
        seg_wavs.append(seg)

    feats = [extract_features(w, sr) for w in seg_wavs]
    F = np.stack(feats, axis=0).astype(np.float32)
    F = zscore_features(F)

    # 4) clustering
    if args.thr > 0:
        labels = cluster_by_threshold(F, thr=args.thr, min_cluster_size=args.min_cluster_size)
        meta = {"chosen_thr": args.thr, "num_speakers": len(set(labels.tolist())), "mode": "fixed_thr"}
    else:
        labels, meta = auto_cluster(
            F,
            min_speakers=max(1, args.min_speakers),
            max_speakers=max(max(1, args.min_speakers), args.max_speakers),
            min_cluster_size=args.min_cluster_size,
            debug=args.debug,
        )
        meta["mode"] = "auto"

    # optional force K
    if args.num_speakers and args.num_speakers > 0:
        target_k = int(args.num_speakers)
        dists = []
        n = F.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(seg_distance(F[i], F[j]))
        dists = np.array(dists, dtype=np.float32)
        cand = sorted(list(set([float(np.percentile(dists, p)) for p in [10,15,20,25,30,35,40,45,50,55,60,65,70]])))
        best_hit = None
        best_score = -1e18
        for thr in cand:
            lab = cluster_by_threshold(F, thr=thr, min_cluster_size=args.min_cluster_size)
            k = len(set(lab.tolist()))
            if k != target_k:
                continue
            score = clustering_separation_score(F, lab)
            if score > best_score:
                best_score = score
                best_hit = (thr, lab, k)
        if best_hit is not None:
            thr, labels, _ = best_hit
            meta = {"chosen_thr": thr, "num_speakers": target_k, "mode": "force_K", "score": best_score}

    # 4.5) role/order remap (fix "speaker swap" issues)
    labels = remap_labels_by_order(segments, labels.tolist(), feats, order_rule=args.speaker_order)

    # 5) post-merge by same speaker
    segments2, labels2 = merge_same_speaker(segments, labels.tolist(), max_gap_s=args.post_merge_gap_ms / 1000.0)
    segments, labels = segments2, np.array(labels2, dtype=np.int32)

    # 6) name speakers by label id after remap
    speaker_names = [f"speaker_{int(lab):02d}" for lab in labels.tolist()]
    K = len(set(speaker_names))

    print(f"\nClustering meta: {meta}")
    print(f"Detected speakers: {K}")
    print("Final diarization:")
    for (ts, te), spk in zip(segments, speaker_names):
        print(f"  {ts:.2f}s - {te:.2f}s  {spk}")

    # 7) save outputs
    save_segment_wavs(x, sr, segments, speaker_names, out_dir=os.path.join(args.out, "segments"))
    save_speaker_tracks(x, sr, segments, speaker_names, out_dir=os.path.join(args.out, "speakers"))

    if args.write_rttm:
        write_rttm(segments, speaker_names, os.path.join(args.out, "diarization.rttm"),
                   file_id=os.path.splitext(os.path.basename(args.audio))[0])
    if args.write_json:
        write_json(segments, speaker_names, os.path.join(args.out, "diarization.json"))

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"audio={args.audio}\n")
        f.write(f"sr={sr}\n")
        f.write(f"segments={len(segments)}\n")
        f.write(f"speakers={K}\n")
        f.write(f"meta={meta}\n")
        f.write(f"speaker_order={args.speaker_order}\n")
        f.write(f"dist_w={_DIST_W}\n")
        f.write(f"bgm_reduce={args.bgm_reduce} mode={args.bgm_mode}\n")

    print(f"\nSaved to: {args.out}")
    print(f"  - {args.out}/segments/*.wav")
    print(f"  - {args.out}/speakers/speaker_XX_timeline.wav + speaker_XX_concat.wav")
    if args.write_rttm:
        print(f"  - {args.out}/diarization.rttm")
    if args.write_json:
        print(f"  - {args.out}/diarization.json")


if __name__ == "__main__":
    main()

