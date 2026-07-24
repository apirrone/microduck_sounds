"""Parrot mode: an always-on ear that learns oft-repeated phrases.

Pipeline (pure numpy, Pi-Zero-2 friendly — no ML models, no cloud):

  mic (arecord) -> energy VAD -> utterance
      -> mel-cepstral features -> DTW match against known phrases
      -> heard `repeats` times? learned! squawk it back, parrot-voiced

"Learning a word" here just means clustering acoustically-similar
utterances, which is plenty for a toy parrot.

Analysis runs at 16 kHz mono S16LE — the same stream microduck_runtime's
pet_worker already pulls from the onboard mic (plughw:aic3104,0). Besides
spawning arecord itself, the loop can read raw samples from stdin (like the
pet_detect binary) so the runtime can tee its single mic stream into it:

  arecord -D plughw:aic3104,0 -f S16_LE -r 16000 -c 1 -t raw \
    | microduck-sounds parrot --stdin
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from . import api
from . import synth as S
from .personality import Personality

PSR = 16000          # analysis sample rate (matches the robot mic)
FRAME = 512          # 32 ms analysis window
HOP = 256            # 16 ms
N_MELS = 20
N_CEPS = 12

MIN_UTT_S = 0.30
MAX_UTT_S = 3.5
# DTW distance below which two utterances "match". Calibrated on synthetic
# speech (same phrase re-spoken with tempo/pitch/level variation lands at
# ~0.9-1.6, different phrases at ~1.7-2.3). Strict side on purpose: a missed
# repeat just costs one more repetition, a false merge learns garbage.
DEFAULT_THRESHOLD = 1.5
DEFAULT_MEMORY = Path.home() / ".microduck" / "parrot"
DEFAULT_OUT_RATE = 48000  # Radxa's I2S clock is pinned to the 48k family
ECHO_COOLDOWN_S = 8.0     # don't echo the same phrase back-to-back


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def _mel_filterbank(fmin: float = 120.0, fmax: float = 7400.0) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    freqs = np.linspace(0.0, PSR / 2.0, FRAME // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), N_MELS + 2))
    fb = np.zeros((N_MELS, len(freqs)), dtype=np.float32)
    for i in range(N_MELS):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0.0, 1.0)
    return fb


_FB = _mel_filterbank()
_DCT = np.cos(np.pi * np.arange(N_CEPS + 1)[:, None]
              * (2 * np.arange(N_MELS)[None, :] + 1) / (2 * N_MELS)).astype(np.float32)
_WINDOW = np.hanning(FRAME).astype(np.float32)


def features(x: np.ndarray) -> np.ndarray:
    """Mean/variance-normalised mel-cepstra, shape (n_frames, N_CEPS)."""
    x = x.astype(np.float32) - float(x.mean())
    if len(x) < FRAME + 3 * HOP:
        x = np.pad(x, (0, FRAME + 3 * HOP - len(x)))
    n = (len(x) - FRAME) // HOP + 1
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * _WINDOW
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = np.log(spec @ _FB.T + 1e-8)
    ceps = (mel @ _DCT.T)[:, 1:]          # drop c0 (overall loudness)
    ceps -= ceps.mean(axis=0)
    ceps /= ceps.std(axis=0) + 1e-6
    return ceps.astype(np.float32)


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Path-length-normalised DTW between two feature sequences.

    DP runs over anti-diagonals so each step is a vectorised numpy op —
    (i-1,j) and (i,j-1) live on the previous diagonal, (i-1,j-1) two back.
    """
    n, m = len(a), len(b)
    C = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).astype(np.float32)
    prev = np.empty(0, dtype=np.float32)
    prev2 = prev
    for k in range(n + m - 1):
        i0, i1 = max(0, k - m + 1), min(n - 1, k)
        ii = np.arange(i0, i1 + 1)
        jj = k - ii
        cur = C[ii, jj].copy()
        if k > 0:
            best = np.full(len(ii), np.inf, dtype=np.float32)
            p0 = max(0, k - m)                       # first i on diagonal k-1
            up = (ii - 1 >= p0) & (ii - 1 <= min(n - 1, k - 1))
            best[up] = prev[ii[up] - 1 - p0]
            left = (jj - 1 >= 0) & (ii <= min(n - 1, k - 1))
            best[left] = np.minimum(best[left], prev[ii[left] - p0])
            if k > 1:
                pp0 = max(0, k - 1 - m)              # first i on diagonal k-2
                diag = (ii - 1 >= pp0) & (ii - 1 <= min(n - 1, k - 2)) & (jj - 1 >= 0)
                best[diag] = np.minimum(best[diag], prev2[ii[diag] - 1 - pp0])
            cur += best
        prev2, prev = prev, cur
    return float(prev[0]) / (n + m)


# --------------------------------------------------------------------------
# parrot voice
# --------------------------------------------------------------------------

def parrotize(x: np.ndarray, p: Personality, in_sr: int = PSR,
              shift: float = 0.0) -> np.ndarray:
    """Pitch the clip up with a warble and a squawky buzz. `shift` adds
    semitones. Input at `in_sr`, output at synth rate (S.SR)."""
    rng = np.random.default_rng()
    x = S.normalise(x.astype(np.float32), peak_dbfs=-3.0)
    semi = 4.5 + 1.5 * p.register + shift + float(rng.uniform(-0.4, 0.4))
    ratio = 2.0 ** (semi / 12.0) * in_sr / S.SR

    # variable-rate resample: pitch-up + speed-up + slow warble in one pass
    t = np.arange(int(len(x) / ratio), dtype=np.float32) / S.SR
    wob = S.vibrato(t, rate_hz=6.5, depth_semitones=0.8, phase=float(rng.uniform(0, 6.28)))
    idx = np.cumsum(ratio * wob).astype(np.float64)
    idx = idx[idx < len(x) - 1]
    y = np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    # squawk buzz at the duck's own quack rate, then gritty soft-clip
    tt = np.arange(len(y), dtype=np.float32) / S.SR
    y *= 1.0 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * max(40.0, p.am_rate_hz) * tt))
    y = np.tanh(3.0 * y)
    return S.normalise(y, peak_dbfs=-3.0)


def squawk_buffer(audio: np.ndarray, p: Personality) -> np.ndarray:
    """Chirp + parrot-voiced phrase, sometimes said twice like a real parrot.
    `audio` at PSR; result at synth rate."""
    rng = np.random.default_rng()
    gap = np.zeros(int(0.12 * S.SR), dtype=np.float32)
    parts = [api.render("chirp", p, variant=int(rng.integers(0, 3))), gap,
             parrotize(audio, p)]
    if rng.random() < 0.4:
        parts += [gap, parrotize(audio, p, shift=1.0)]
    return np.concatenate(parts)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    pos = np.arange(int(len(x) * sr_out / sr_in)) * (sr_in / sr_out)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


# --------------------------------------------------------------------------
# VAD
# --------------------------------------------------------------------------

class VoiceGate:
    """Feed HOP-sized chunks at PSR; returns a full utterance when one ends."""

    START_RUN = 3                      # ~48 ms of voice to open
    END_RUN = 24                       # ~380 ms of quiet to close
    PREROLL = 9                        # ~140 ms kept from before the trigger
    KEEP_TAIL = 3                      # quiet chunks kept when trimming the end

    def __init__(self) -> None:
        self.floor = 0.004
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.run = 0
        self.quiet = 0
        self.buf: list[np.ndarray] = []
        self.preroll: list[np.ndarray] = []

    def feed(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        thresh = max(self.floor * 3.0, 0.010)
        voiced = rms > thresh

        if not self.in_speech:
            if not voiced:
                self.floor = 0.95 * self.floor + 0.05 * rms
            self.preroll.append(chunk)
            if len(self.preroll) > self.PREROLL:
                self.preroll.pop(0)
            self.run = self.run + 1 if voiced else 0
            if self.run >= self.START_RUN:
                self.in_speech = True
                self.buf = list(self.preroll)
                self.quiet = 0
            return None

        self.buf.append(chunk)
        self.quiet = 0 if voiced else self.quiet + 1
        dur = len(self.buf) * HOP / PSR
        if self.quiet >= self.END_RUN or dur >= MAX_UTT_S:
            # trim the quiet tail — silence matches silence, which would
            # drag every pair of utterances closer together in DTW
            keep = max(1, len(self.buf) - self.quiet + self.KEEP_TAIL)
            utt = np.concatenate(self.buf[:keep])
            self.reset()
            if len(utt) / PSR >= MIN_UTT_S:
                return utt
        return None


# --------------------------------------------------------------------------
# audio sources
# --------------------------------------------------------------------------

class Mic:
    """Raw S16LE stream at PSR from arecord. pause()/resume() around our own
    playback so the parrot doesn't learn its own squawks."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if shutil.which("arecord") is None:
            raise RuntimeError("arecord not found — install alsa-utils, "
                               "or pipe raw audio in with --stdin")
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(PSR), "-c", "1", "-t", "raw"]
        if self.device:
            cmd += ["-D", self.device]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def read(self, n: int) -> Optional[np.ndarray]:
        assert self.proc is not None and self.proc.stdout is not None
        want = n * 2
        data = b""
        while len(data) < want:
            part = self.proc.stdout.read(want - len(data))
            if not part:
                raise RuntimeError("arecord stream ended unexpectedly")
            data += part
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

    def spoke_for(self, seconds: float) -> None:
        """Mic was stopped during playback; nothing buffered to skip."""

    pause = stop
    resume = start


class StdinSource:
    """Raw S16LE at PSR on stdin — the pet_detect pattern, so the runtime can
    tee its single arecord stream into the parrot."""

    def __init__(self) -> None:
        self.skip = 0  # samples to discard (audio that arrived while we spoke)

    def start(self) -> None:
        pass

    def read(self, n: int) -> Optional[np.ndarray]:
        drop = min(self.skip, 8192)
        data = sys.stdin.buffer.read((n + drop) * 2)
        if data is None or len(data) < (n + drop) * 2:
            return None
        self.skip -= drop
        x = np.frombuffer(data, dtype=np.int16)[drop:]
        return x.astype(np.float32) / 32768.0

    def stop(self) -> None:
        pass

    def spoke_for(self, seconds: float) -> None:
        # stdin kept flowing while we played; drop that stretch (+ margin)
        self.skip += int((seconds + 0.25) * PSR)

    def pause(self) -> None:
        pass

    resume = start


class WavSource:
    """Feed a wav file through the same loop — testing without a microphone."""

    def __init__(self, path: Path) -> None:
        self.audio = load_wav(path)
        self.pos = 0

    def start(self) -> None:
        pass

    def read(self, n: int) -> Optional[np.ndarray]:
        if self.pos >= len(self.audio):
            return None
        chunk = self.audio[self.pos:self.pos + n]
        self.pos += n
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        return chunk

    def stop(self) -> None:
        pass

    def spoke_for(self, seconds: float) -> None:
        pass

    def pause(self) -> None:
        pass

    resume = start


def load_wav(path: Path) -> np.ndarray:
    """16-bit wav -> float32 mono at PSR."""
    with wave.open(str(path), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit wavs supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return _resample(x, sr, PSR)


def save_wav(x: np.ndarray, path: Path, sr: int = PSR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(S.to_int16(x).tobytes())
    return path


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------

class ParrotMemory:
    """Clusters of similar utterances. Pending ones live in RAM; once a
    cluster has been heard `repeats` times it's promoted to a learned phrase
    and persisted as a wav under `root`."""

    MAX_PENDING = 24
    MAX_LEARNED = 16

    def __init__(self, root: Path, repeats: int = 3,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self.root = Path(root)
        self.repeats = repeats
        self.threshold = threshold
        self.pending: list[dict] = []
        self.learned: list[dict] = []
        self._load()

    def _load(self) -> None:
        meta_path = self.root / "phrases.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text())
        for entry in meta:
            path = self.root / entry["file"]
            if not path.exists():
                continue
            audio = load_wav(path)
            self.learned.append({
                "id": path.stem, "path": path, "audio": audio,
                "feat": features(audio), "count": entry.get("count", self.repeats),
                "last_heard": 0.0, "last_spoken": 0.0,
            })

    def _save_meta(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = [{"file": ph["path"].name, "count": ph["count"]} for ph in self.learned]
        (self.root / "phrases.json").write_text(json.dumps(meta, indent=1))

    def _best(self, feat: np.ndarray, pool: list[dict]) -> tuple[Optional[dict], float]:
        best, best_d = None, float("inf")
        for item in pool:
            n, m = len(feat), len(item["feat"])
            if max(n, m) > 1.6 * min(n, m):      # too different in length to bother
                continue
            d = dtw_distance(feat, item["feat"])
            if d < best_d:
                best, best_d = item, d
        return best, best_d

    def observe(self, audio: np.ndarray) -> tuple[str, Optional[dict], float]:
        """Returns (event, phrase, distance) where event is one of
        'recognized' | 'learned' | 'again' | 'new'."""
        feat = features(audio)
        now = time.monotonic()

        ph, d = self._best(feat, self.learned)
        if ph is not None and d < self.threshold:
            ph["count"] += 1
            ph["last_heard"] = now
            self._save_meta()
            return "recognized", ph, d

        cand, d = self._best(feat, self.pending)
        if cand is not None and d < self.threshold:
            cand["count"] += 1
            cand["last_heard"] = now
            if cand["count"] >= self.repeats:
                self.pending.remove(cand)
                return "learned", self._promote(cand, audio, feat), d
            return "again", cand, d

        cand = {"id": f"candidate-{len(self.pending)}", "audio": audio,
                "feat": feat, "count": 1, "last_heard": now}
        self.pending.append(cand)
        if len(self.pending) > self.MAX_PENDING:
            self.pending.remove(min(self.pending, key=lambda c: (c["count"], c["last_heard"])))
        return "new", cand, float("inf")

    def _promote(self, cand: dict, audio: np.ndarray, feat: np.ndarray) -> dict:
        # keep the freshest take of the phrase, not the first (often cleaner)
        if len(self.learned) >= self.MAX_LEARNED:
            old = min(self.learned, key=lambda ph: ph["last_heard"])
            self.learned.remove(old)
            old["path"].unlink(missing_ok=True)
        taken = {ph["path"].name for ph in self.learned}
        i = 0
        while f"phrase_{i:03d}.wav" in taken:
            i += 1
        path = save_wav(audio, self.root / f"phrase_{i:03d}.wav")
        ph = {"id": path.stem, "path": path, "audio": audio, "feat": feat,
              "count": cand["count"], "last_heard": cand["last_heard"], "last_spoken": 0.0}
        self.learned.append(ph)
        self._save_meta()
        return ph


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def _safe_play(buf: np.ndarray, mute: bool, out_rate: int) -> float:
    """Play at out_rate; returns the clip duration in seconds."""
    dur = len(buf) / S.SR
    if not mute:
        try:
            api.play_buffer(_resample(buf, S.SR, out_rate), sr=out_rate)
        except RuntimeError as e:
            print(f"  (playback failed: {e})")
    return dur


def run(seed: int = 0, memory_dir: Path = DEFAULT_MEMORY, repeats: int = 3,
        threshold: float = DEFAULT_THRESHOLD, wav: Optional[Path] = None,
        use_stdin: bool = False, device: Optional[str] = None,
        babble_s: float = 240.0, out_rate: int = DEFAULT_OUT_RATE,
        mute: bool = False) -> int:
    p = Personality.from_seed(seed)
    mem = ParrotMemory(memory_dir, repeats=repeats, threshold=threshold)
    rng = np.random.default_rng()
    if wav is not None:
        src = WavSource(wav)
    elif use_stdin:
        src = StdinSource()
    else:
        src = Mic(device=device)
    gate = VoiceGate()

    print(f"parrot awake — {len(mem.learned)} learned phrase(s) in {mem.root}")
    _safe_play(api.render("chirp", p), mute, out_rate)

    def speak(buf: np.ndarray) -> None:
        src.pause()
        dur = _safe_play(buf, mute, out_rate)
        src.resume()
        src.spoke_for(dur)
        gate.reset()

    try:
        src.start()
    except RuntimeError as e:
        print(e)
        return 1
    last_activity = time.monotonic()
    next_babble = babble_s * float(rng.uniform(0.7, 1.3))
    live = wav is None  # babble only makes sense in real time
    try:
        while True:
            try:
                chunk = src.read(HOP)
            except RuntimeError as e:
                # arecord died (device hiccup) — same recovery as pet_worker
                print(f"mic error: {e} — restarting in 1s")
                src.stop()
                time.sleep(1.0)
                src.start()
                gate.reset()
                continue
            if chunk is None:
                break
            utt = gate.feed(chunk)
            now = time.monotonic()
            if utt is None:
                if live and mem.learned and now - last_activity > next_babble:
                    ph = mem.learned[int(rng.integers(len(mem.learned)))]
                    print(f"parrot babbles {ph['id']} to itself")
                    speak(squawk_buffer(ph["audio"], p))
                    last_activity = time.monotonic()
                    next_babble = babble_s * float(rng.uniform(0.7, 1.3))
                continue

            last_activity = now
            dur = len(utt) / PSR
            event, ph, d = mem.observe(utt)
            assert ph is not None
            if event == "new":
                print(f"heard something new ({dur:.1f}s) — watching for repeats")
            elif event == "again":
                print(f"heard that again! ({ph['count']}/{repeats}, d={d:.2f})")
            elif event == "learned":
                print(f"learned {ph['id']}! ({dur:.1f}s, d={d:.2f}) squawk!")
                speak(squawk_buffer(ph["audio"], p))
            elif event == "recognized":
                print(f"recognized {ph['id']} (heard {ph['count']}x, d={d:.2f})")
                if now - ph["last_spoken"] > ECHO_COOLDOWN_S:
                    ph["last_spoken"] = time.monotonic()
                    speak(squawk_buffer(ph["audio"], p))
    except KeyboardInterrupt:
        print("\nparrot goes to sleep")
    finally:
        src.stop()
    return 0


def say(wav: Optional[Path] = None, seed: int = 0,
        memory_dir: Path = DEFAULT_MEMORY,
        out_rate: int = DEFAULT_OUT_RATE) -> int:
    """Parrot-voice a wav file, or a random learned phrase if none given."""
    p = Personality.from_seed(seed)
    if wav is not None:
        audio = load_wav(wav)
    else:
        mem = ParrotMemory(memory_dir)
        if not mem.learned:
            print(f"no learned phrases in {mem.root} — run `microduck-sounds parrot` first")
            return 1
        rng = np.random.default_rng()
        ph = mem.learned[int(rng.integers(len(mem.learned)))]
        print(f"squawking {ph['id']}")
        audio = ph["audio"]
    buf = squawk_buffer(audio, p)
    api.play_buffer(_resample(buf, S.SR, out_rate), sr=out_rate)
    return 0
