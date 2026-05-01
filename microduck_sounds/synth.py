"""DSP primitives. Pure numpy, vectorised, Pi-Zero-2 friendly."""
from __future__ import annotations

import numpy as np

SR = 22050


def t_axis(duration_s: float, sr: int = SR) -> np.ndarray:
    n = max(1, int(round(duration_s * sr)))
    return np.arange(n, dtype=np.float32) / sr


def lerp(t: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear curve. `points` are (time_s, value)."""
    xs = np.asarray([p[0] for p in points], dtype=np.float32)
    ys = np.asarray([p[1] for p in points], dtype=np.float32)
    return np.interp(t, xs, ys).astype(np.float32)


def expdecay(t: np.ndarray, attack_s: float, decay_s: float) -> np.ndarray:
    """Fast attack, exponential decay envelope, peak ~1.0."""
    a = np.clip(t / max(attack_s, 1e-4), 0.0, 1.0)
    d = np.exp(-np.maximum(t - attack_s, 0.0) / max(decay_s, 1e-4))
    env = a * d
    return env.astype(np.float32)


def bell(t: np.ndarray, attack_s: float, hold_s: float, release_s: float) -> np.ndarray:
    """Soft attack, plateau, soft release. Useful for `coo` / `inquire`."""
    total = float(t[-1]) if len(t) else 0.0
    out = np.ones_like(t)
    rel_start = max(0.0, total - release_s)
    out = np.where(t < attack_s, t / max(attack_s, 1e-4), out)
    out = np.where(t > rel_start, np.maximum(0.0, 1.0 - (t - rel_start) / max(release_s, 1e-4)), out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def phase_from_freq(freq: np.ndarray, sr: int = SR) -> np.ndarray:
    """Integrate instantaneous frequency to phase (radians)."""
    return (2.0 * np.pi * np.cumsum(freq) / sr).astype(np.float32)


def harmonic_osc(phase: np.ndarray, weights) -> np.ndarray:
    """Sum sin(n*phase) * weight for each harmonic (n=1..len)."""
    out = np.zeros_like(phase)
    for n, w in enumerate(weights, start=1):
        if w == 0:
            continue
        out += float(w) * np.sin(n * phase)
    return out.astype(np.float32)


def vibrato(t: np.ndarray, rate_hz: float, depth_semitones: float, phase: float = 0.0) -> np.ndarray:
    """Pitch multiplier from a slow LFO. Multiply your freq curve by this."""
    if rate_hz <= 0 or depth_semitones <= 0:
        return np.ones_like(t)
    lfo = np.sin(2.0 * np.pi * rate_hz * t + phase)
    return (2.0 ** (depth_semitones * lfo / 12.0)).astype(np.float32)


def jitter(t: np.ndarray, depth_semitones: float, rng: np.random.Generator, smoothing: int = 64) -> np.ndarray:
    """Random pitch wobble — gives organic, alive feel. Multiply your freq curve."""
    if depth_semitones <= 0:
        return np.ones_like(t)
    raw = rng.standard_normal(len(t)).astype(np.float32)
    if smoothing > 1:
        k = np.ones(smoothing, dtype=np.float32) / smoothing
        raw = np.convolve(raw, k, mode="same")
    return (2.0 ** (depth_semitones * raw / 12.0)).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap pink-ish noise via cumulative sum + leak. Soft, breathy."""
    white = rng.standard_normal(n).astype(np.float32)
    pink = np.zeros(n, dtype=np.float32)
    leak = 0.0
    a = 0.985
    for i in range(n):
        leak = a * leak + white[i]
        pink[i] = leak
    pink /= np.max(np.abs(pink)) + 1e-9
    return pink


def click(n: int, rng: np.random.Generator, length: int = 80) -> np.ndarray:
    """Short transient click — for `peck` attack."""
    out = np.zeros(n, dtype=np.float32)
    L = min(length, n)
    out[:L] = rng.uniform(-1.0, 1.0, size=L).astype(np.float32)
    out[:L] *= np.linspace(1.0, 0.0, L).astype(np.float32) ** 2
    return out


def normalise(x: np.ndarray, peak_dbfs: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-9
    target = 10.0 ** (peak_dbfs / 20.0)
    return (x * (target / peak)).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
