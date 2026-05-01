"""Public API: render, save, play."""
from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Union

import numpy as np

from .personality import Personality
from .voices import RECIPES, VARIANT_COUNT
from . import synth as S

TAGS = list(RECIPES.keys())

SeedLike = Union[int, Personality]


def _resolve(seed: SeedLike) -> Personality:
    return seed if isinstance(seed, Personality) else Personality.from_seed(seed)


def render(tag: str, seed: SeedLike, variant: int = 0) -> np.ndarray:
    """Synthesize a single sound. Returns float32 mono at synth.SR."""
    if tag not in RECIPES:
        raise KeyError(f"unknown tag {tag!r}; known: {TAGS}")
    p = _resolve(seed)
    return RECIPES[tag](p, variant=variant)


def to_wav(buffer: np.ndarray, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = S.to_int16(buffer)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(S.SR)
        w.writeframes(pcm.tobytes())
    return path


def render_all(seed: SeedLike, out_dir: Union[str, Path]) -> list[Path]:
    """Render every (tag, variant) pair into <out_dir>/<tag>/<tag>_<letter>.wav.

    Matches the layout described in microduck_brain/docs/sounds.md.
    """
    p = _resolve(seed)
    root = Path(out_dir)
    paths = []
    for tag, n in VARIANT_COUNT.items():
        for v in range(n):
            buf = render(tag, p, variant=v)
            letter = chr(ord("a") + v)
            paths.append(to_wav(buf, root / tag / f"{tag}_{letter}.wav"))
    return paths


_PLAYERS = ("aplay", "paplay", "ffplay", "mpv")


def _wav_bytes(buffer: np.ndarray) -> bytes:
    pcm = S.to_int16(buffer).tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, S.SR, S.SR * 2, 2, 16)
    data = b"data" + struct.pack("<I", len(pcm)) + pcm
    return header + fmt + data


def play(tag: str, seed: SeedLike, variant: int = 0) -> None:
    """Synthesize and play immediately. Tries aplay/paplay/ffplay/mpv."""
    buf = render(tag, seed, variant=variant)
    payload = _wav_bytes(buf)
    for player in _PLAYERS:
        if shutil.which(player) is None:
            continue
        if player == "aplay":
            cmd = ["aplay", "-q", "-"]
        elif player == "paplay":
            cmd = ["paplay", "--raw=false"]
        elif player == "ffplay":
            cmd = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"]
        else:
            cmd = ["mpv", "--really-quiet", "-"]
        subprocess.run(cmd, input=payload, check=True)
        return
    raise RuntimeError(f"no audio player found (tried {_PLAYERS})")
