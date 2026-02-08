import os
import argparse
import math
import numpy as np
import soundfile as sf
import torch


# -----------------------------
# Utils: force 1D mono
# -----------------------------
def to_mono_1d(waveform: np.ndarray) -> np.ndarray:
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
        return w.mean(axis=1).astype(np.float32).reshape(-1)
    else:
        return w.mean(axis=0).astype(np.float32).reshape(-1)


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_audio(waveform: np.ndarray, sr: int, target_sr: int = 16000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = to_mono_1d(waveform)

    # remove DC
    x = x - float(np.mean(x))

    # tiny dither
    x = x + (1e-6 * rng.standard_normal(x.shape).astype(np.float32))

    # resample
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
                f"Install torchaudio or provide already-16k audio. Error: {e}"
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


# -----------------------------
# Energy segmentation (robust)
# -----------------------------
def segment_by_energy(
    x: np.ndarray,
    sr: int,
    frame_ms=20,
    hop_ms=10,
    min_seg_ms=160,
    max_silence_ms=180,
    pad_ms=20,
    thr_mad_k=3.5,
    thr_floor=1e-6,
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
    if len(rms) >= k:
        rms_s = np.convolve(rms, np.ones(k) / k, mode="same")
    else:
        rms_s = rms

    # robust threshold: MAD + percentile fallback
    med = float(np.median(rms_s))
    mad = float(np.median(np.abs(rms_s - med)) + 1e-12)
    thr1 = med + thr_mad_k * mad

    p10 = float(np.percentile(rms_s, 10))
    p90 = float(np.percentile(rms_s, 90))
    thr2 = p10 + 0.25 * max(0.0, (p90 - p10))

    thr = min(thr1, thr2)
    thr = max(thr, thr_floor)

    speech = rms_s > thr

    # if empty, relax threshold progressively
    if not speech.any():
        for alpha in [0.15, 0.10, 0.07, 0.05]:
            thr = max(p10 + alpha * max(0.0, (p90 - p10)), thr_floor)
            speech = rms_s > thr
            if speech.any():
                break

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


# -----------------------------
# Merge segments to target turns
# -----------------------------
def merge_segments_to_n(segments, target_n=4, max_merge_gap_s=0.25, x=None, sr=None):
    """
    Merge adjacent segments until len == target_n.
    Priority:
      1) merge the adjacent pair with the smallest gap if gap <= max_merge_gap_s
      2) if all gaps larger, merge the shortest segment with its closest neighbor
    """
    segs = list(sorted(segments, key=lambda t: t[0]))
    if len(segs) <= target_n:
        return segs

    while len(segs) > target_n:
        gaps = []
        for i in range(len(segs) - 1):
            gap = segs[i + 1][0] - segs[i][1]
            gaps.append(gap)

        min_i = int(np.argmin(gaps))
        min_gap = gaps[min_i]

        if min_gap <= max_merge_gap_s:
            # merge by smallest gap
            a = segs[min_i]
            b = segs[min_i + 1]
            merged = (a[0], b[1])
            segs = segs[:min_i] + [merged] + segs[min_i + 2 :]
            continue

        # else: merge shortest seg with nearest neighbor
        lens = [s[1] - s[0] for s in segs]
        k = int(np.argmin(lens))
        if k == 0:
            merge_i = 0
        elif k == len(segs) - 1:
            merge_i = len(segs) - 2
        else:
            left_gap = segs[k][0] - segs[k - 1][1]
            right_gap = segs[k + 1][0] - segs[k][1]
            merge_i = (k - 1) if (left_gap <= right_gap) else k

        a = segs[merge_i]
        b = segs[merge_i + 1]
        merged = (a[0], b[1])
        segs = segs[:merge_i] + [merged] + segs[merge_i + 2 :]

    return segs


# -----------------------------
# Features for 4-turn pairing
# -----------------------------
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


def estimate_pitch_autocorr(x: np.ndarray, sr: int, fmin=60, fmax=700):
    x = to_mono_1d(x)
    if len(x) < int(0.05 * sr):
        return 0.0
    mid = x[len(x)//4 : 3*len(x)//4]
    mid = mid - float(np.mean(mid))
    mid = mid * np.hanning(len(mid)).astype(np.float32)

    ac = np.correlate(mid, mid, mode="full")[len(mid)-1:]
    if ac[0] <= 1e-9:
        return 0.0

    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, len(ac)-1)
    if lag_max <= lag_min:
        return 0.0

    seg = ac[lag_min:lag_max]
    lag = int(np.argmax(seg) + lag_min)
    if lag <= 0:
        return 0.0
    return float(sr / lag)


def estimate_pitch(x: np.ndarray, sr: int):
    # torchaudio
    try:
        import torchaudio
        wav = torch.from_numpy(to_mono_1d(x)).float().unsqueeze(0)
        pitch = torchaudio.functional.detect_pitch_frequency(
            wav, sample_rate=sr, frame_time=0.02,
            win_length=int(0.02 * sr), hop_length=int(0.01 * sr)
        ).squeeze(0).cpu().numpy()
        pitch = pitch[(pitch > 50) & (pitch < 900)]
        if pitch.size > 0:
            return float(np.median(pitch))
    except Exception:
        pass

    # librosa
    try:
        import librosa
        f0 = librosa.yin(
            to_mono_1d(x).astype(np.float32),
            fmin=50, fmax=900, sr=sr,
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


def extract_features(x_seg: np.ndarray, sr: int):
    x_seg = to_mono_1d(x_seg)
    mfcc = extract_mfcc_if_available(x_seg, sr, n_mfcc=20)

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


def distance(f1: np.ndarray, f2: np.ndarray):
    pitch1, pitch2 = float(f1[-4]), float(f2[-4])
    v1 = f1[:-4] if f1.shape[0] > 4 else np.zeros((1,), dtype=np.float32)
    v2 = f2[:-4] if f2.shape[0] > 4 else np.zeros((1,), dtype=np.float32)

    cos_d = cosine_dist(v1, v2) if (v1.size == v2.size and v1.size > 1) else 0.0
    pitch_d = abs(pitch1 - pitch2)
    spec_d = float(np.linalg.norm((f1[-3:] - f2[-3:]).astype(np.float32)))
    return cos_d + 2.5 * pitch_d + 0.5 * spec_d


def best_pairing_for_4(feats):
    pairings = [
        ((0, 3), (1, 2)),
        ((0, 2), (1, 3)),
        ((0, 1), (2, 3)),
    ]
    best = None
    best_score = 1e18
    for (a1, a2), (b1, b2) in pairings:
        score = distance(feats[a1], feats[a2]) + distance(feats[b1], feats[b2])
        if score < best_score:
            best_score = score
            best = ((a1, a2), (b1, b2))
    return set(best[0]), set(best[1]), best_score


# -----------------------------
# Writers
# -----------------------------
def save_segments(x, sr, segments, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, ((ts, te), lab) in enumerate(zip(segments, labels)):
        s = int(ts * sr)
        e = int(te * sr)
        sf.write(os.path.join(out_dir, f"{idx:02d}_{lab}_{ts:.2f}-{te:.2f}.wav"), x[s:e], sr)


def save_speaker_tracks(x, sr, segments, labels, out_dir, silence_gap_ms=200):
    os.makedirs(out_dir, exist_ok=True)
    x = to_mono_1d(x)

    p0 = np.zeros_like(x, dtype=np.float32)
    p1 = np.zeros_like(x, dtype=np.float32)
    for (ts, te), lab in zip(segments, labels):
        s = int(ts * sr)
        e = int(te * sr)
        if lab == "person0":
            p0[s:e] = x[s:e]
        else:
            p1[s:e] = x[s:e]
    sf.write(os.path.join(out_dir, "person0_timeline.wav"), p0, sr)
    sf.write(os.path.join(out_dir, "person1_timeline.wav"), p1, sr)

    gap = np.zeros(int(sr * silence_gap_ms / 1000), dtype=np.float32)
    p0_list, p1_list = [], []
    for (ts, te), lab in zip(segments, labels):
        s = int(ts * sr)
        e = int(te * sr)
        seg = x[s:e]
        if lab == "person0":
            p0_list += [seg, gap]
        else:
            p1_list += [seg, gap]
    p0c = np.concatenate(p0_list, axis=0) if p0_list else np.zeros(0, dtype=np.float32)
    p1c = np.concatenate(p1_list, axis=0) if p1_list else np.zeros(0, dtype=np.float32)
    sf.write(os.path.join(out_dir, "person0_concat.wav"), p0c, sr)
    sf.write(os.path.join(out_dir, "person1_concat.wav"), p1c, sr)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--out", type=str, default="./turns_energy")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--expect_turns", type=int, default=4)
    ap.add_argument("--merge_gap_ms", type=float, default=250.0)

    ap.add_argument("--person0_rule", type=str, default="first", choices=["first", "high_pitch", "low_pitch"])
    ap.add_argument("--ref0", type=str, default=None)
    ap.add_argument("--ref1", type=str, default=None)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    wav, sr0 = sf.read(args.audio, always_2d=False, dtype="float32")
    wav = to_mono_1d(wav)
    x, sr = preprocess_audio(wav, sr0, target_sr=args.sr, seed=args.seed)

    if args.debug:
        print(f"[DEBUG] sr={sr}, len={len(x)}, sec={len(x)/sr:.2f}, min={x.min():.3f}, max={x.max():.3f}, rms={np.sqrt(np.mean(x*x)+1e-12):.6f}")

    segments = segment_by_energy(x, sr)

    print("Energy-VAD segments (raw):")
    for i, (ts, te) in enumerate(segments):
        print(f"  [{i}] {ts:.2f}s - {te:.2f}s  dur={te-ts:.2f}s")

    # merge to expected turns
    merged = merge_segments_to_n(
        segments,
        target_n=args.expect_turns,
        max_merge_gap_s=args.merge_gap_ms / 1000.0,
    )

    if merged != segments:
        print(f"\nMerged segments to {len(merged)} turns (target={args.expect_turns}, merge_gap_ms={args.merge_gap_ms}):")
        for i, (ts, te) in enumerate(merged):
            print(f"  [{i}] {ts:.2f}s - {te:.2f}s  dur={te-ts:.2f}s")

    segments = merged

    if len(segments) != args.expect_turns:
        print(f"\n[FAIL] still got {len(segments)} segments (expected {args.expect_turns}).")
        print("=> 你可以把 --merge_gap_ms 调大一点（比如 350），或把 max_silence_ms 调大（代码里默认 180ms）。")
        return

    if args.expect_turns != 4:
        print("\nThis script focuses on 4-turn labeling. You can keep expect_turns=4 for your case.")
        return

    # cut + trim for features
    seg_wavs = []
    for (ts, te) in segments:
        s = int(ts * sr)
        e = int(te * sr)
        seg = trim_leading_trailing_silence(x[s:e], sr)
        seg_wavs.append(seg)

    feats = [extract_features(w, sr) for w in seg_wavs]
    A, B, score = best_pairing_for_4(feats)
    print(f"\nBest pairing score={score:.4f}  A={sorted(list(A))}  B={sorted(list(B))}")

    def group_pitch_mean(group):
        vals = [float(feats[i][-4]) for i in group]  # log pitch
        return float(np.mean(vals)) if vals else 0.0

    # reference mapping (best if you have it)
    if args.ref0 and args.ref1:
        r0, r0sr = sf.read(args.ref0, always_2d=False, dtype="float32")
        r1, r1sr = sf.read(args.ref1, always_2d=False, dtype="float32")
        r0 = to_mono_1d(r0); r1 = to_mono_1d(r1)
        r0, _ = preprocess_audio(r0, r0sr, target_sr=sr, seed=args.seed)
        r1, _ = preprocess_audio(r1, r1sr, target_sr=sr, seed=args.seed)
        r0f = extract_features(trim_leading_trailing_silence(r0, sr), sr)
        r1f = extract_features(trim_leading_trailing_silence(r1, sr), sr)

        gA = np.mean([feats[i] for i in A], axis=0)
        gB = np.mean([feats[i] for i in B], axis=0)

        dA0 = distance(gA, r0f); dA1 = distance(gA, r1f)
        dB0 = distance(gB, r0f); dB1 = distance(gB, r1f)

        if (dA0 + dB1) <= (dA1 + dB0):
            person0_group = A
        else:
            person0_group = B

        print(f"[REF MAP] dA0={dA0:.4f} dA1={dA1:.4f} dB0={dB0:.4f} dB1={dB1:.4f}")
    else:
        if args.person0_rule == "first":
            person0_group = A if (0 in A) else B
        elif args.person0_rule == "high_pitch":
            person0_group = A if group_pitch_mean(A) >= group_pitch_mean(B) else B
        else:  # low_pitch
            person0_group = A if group_pitch_mean(A) <= group_pitch_mean(B) else B

    labels = [("person0" if i in person0_group else "person1") for i in range(4)]
    print("Final labels:", labels)

    save_segments(x, sr, segments, labels, out_dir=args.out)
    save_speaker_tracks(x, sr, segments, labels, out_dir=args.out)
    print(f"Saved to: {args.out}")
    print("  - per-turn wavs: 00_personX_*.wav ...")
    print("  - person0_timeline.wav / person1_timeline.wav")
    print("  - person0_concat.wav / person1_concat.wav")


if __name__ == "__main__":
    main()

