"""Recipes for each tag. Each function returns a float32 mono buffer at SR.

Recipes paint with the personality's traits — pitch center, register,
glide bias, harmonic tilt/formant, quackiness, warble — so the *same*
recipe on two different seeds gives two recognisably different ducks.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .personality import Personality
from . import synth as S


def _attack(p: Personality, dur: float, snappy: float = 1.0) -> float:
    """Attack time in seconds, modulated by personality.attack_sharpness.

    `snappy=1` for percussive recipes; lower for soft recipes that should
    still be soft on snappy ducks.
    """
    soft = 0.04 * dur
    sharp = 0.003 * dur
    a = soft + (sharp - soft) * p.attack_sharpness * snappy
    return float(max(a, 0.001))


def _voice(p: Personality, t: np.ndarray, freq: np.ndarray, rng: np.random.Generator,
           am_scale: float = 1.0, breath_scale: float = 1.0) -> np.ndarray:
    """Shared core: harmonic osc + vibrato + jitter + (optional) AM buzz + breath."""
    vib = S.vibrato(t, p.vibrato_rate_hz, p.vibrato_depth, phase=float(rng.uniform(0, 6.28)))
    jit = S.jitter(t, p.jitter_depth, rng)
    f = freq * vib * jit
    phase = S.phase_from_freq(f)
    body = S.harmonic_osc(phase, p.harmonics())

    # quackiness gates the AM buzz: pure-tone ducks have ~no AM
    am_d = p.am_depth * am_scale * p.quackiness
    if am_d > 0.01:
        am = 1.0 - am_d * (0.5 + 0.5 * np.sin(2 * np.pi * p.am_rate_hz * t))
        body *= am

    breath = p.breath * breath_scale
    if breath > 0:
        body += breath * S.pink_noise(len(t), rng)
    return body


def alarm(p: Personality, variant: int = 0) -> np.ndarray:
    rng = p.variant_rng("alarm", variant)
    dur = (0.20 + 0.12 * rng.random()) / p.speed
    t = S.t_axis(dur)
    # raised relative to this duck's center, but stays in honk range;
    # spread controls how high it climbs
    f0 = p.pitch_center_hz * (1.25 + 0.35 * p.pitch_spread)
    peak_mul = 1.15 + 0.25 * p.pitch_spread + 0.10 * rng.random()
    fall_mul = 0.75 + 0.20 * (1.0 - p.pitch_spread)
    freq = S.lerp(t, [(0.0, f0), (0.05 * dur, f0 * peak_mul), (dur, f0 * fall_mul)])
    env = S.expdecay(t, attack_s=_attack(p, dur, snappy=1.0), decay_s=dur * (0.40 + 0.20 * rng.random()))
    sig = _voice(p, t, freq, rng, am_scale=0.5) * env
    # crackle scales with brightness — bright ducks rasp, soft ducks just yelp
    sig += (0.04 + 0.10 * p.brightness) * rng.standard_normal(len(t)).astype(np.float32) * env
    return S.normalise(sig, peak_dbfs=-3.0)


def greet(p: Personality, variant: int = 0) -> np.ndarray:
    rng = p.variant_rng("greet", variant)
    dur = (0.32 + 0.25 * rng.random()) / p.speed
    t = S.t_axis(dur)
    f0 = p.pitch_center_hz * (0.9 + 0.15 * rng.random())
    # glide_bias flips contour: positive ducks bend up, negative down
    bias = p.glide_bias
    bend = 0.10 + 0.15 * p.pitch_spread
    start = f0 * (1.0 - bias * bend * 0.5)
    mid = f0 * (1.0 + bias * bend)
    end = f0 * (1.0 - bias * bend * 0.3) * (0.92 + 0.08 * rng.random())
    freq = S.lerp(t, [(0.0, start), (0.18 * dur, mid), (dur, end)])
    env = S.expdecay(t, attack_s=_attack(p, dur, snappy=0.5), decay_s=dur * 0.7)
    sig = _voice(p, t, freq, rng) * env
    return S.normalise(sig)


def inquire(p: Personality, variant: int = 0) -> np.ndarray:
    rng = p.variant_rng("inquire", variant)
    dur = (0.42 + 0.25 * rng.random()) / p.speed
    t = S.t_axis(dur)
    f0 = p.pitch_center_hz * (0.88 + 0.10 * rng.random())
    # always rises (it's a question), but how much depends on spread + bias
    rise = 1.15 + 0.50 * p.pitch_spread + 0.20 * max(0.0, p.glide_bias) + 0.10 * rng.random()
    freq = S.lerp(t, [(0.0, f0 * 0.92), (0.30 * dur, f0 * 0.95), (dur, f0 * rise)])
    env = S.bell(t, attack_s=dur * (0.06 + 0.10 * (1 - p.attack_sharpness)),
                 hold_s=dur * 0.50, release_s=dur * 0.32)
    sig = _voice(p, t, freq, rng, am_scale=0.6) * env
    return S.normalise(sig)


def peck(p: Personality, variant: int = 0) -> np.ndarray:
    rng = p.variant_rng("peck", variant)
    dur = (0.16 + 0.12 * rng.random()) / p.speed
    t = S.t_axis(dur)
    # always low for this duck; bigger ducks (low register) get even lower pecks
    f0 = p.pitch_center_hz * (0.45 + 0.20 * rng.random())
    freq = S.lerp(t, [(0.0, f0 * 1.5), (0.04 * dur, f0), (dur, f0 * 0.80)])
    env = S.expdecay(t, attack_s=_attack(p, dur, snappy=1.0), decay_s=dur * 0.35)
    body = _voice(p, t, freq, rng, am_scale=0.3, breath_scale=0.5) * env
    # click amount depends on attack_sharpness — snappy ducks have a sharper "tock"
    click_len = int((0.003 + 0.006 * p.attack_sharpness) * S.SR)
    body += (0.4 + 0.4 * p.attack_sharpness) * S.click(len(t), rng, length=click_len)
    return S.normalise(body)


def chirp(p: Personality, variant: int = 0) -> np.ndarray:
    rng = p.variant_rng("chirp", variant)
    dur = (0.10 + 0.13 * rng.random()) / p.speed
    t = S.t_axis(dur)
    f0 = p.pitch_center_hz * (1.2 + 0.4 * rng.random())
    # warble: rapid trill, depth & rate are personality-driven
    warble = S.vibrato(t, p.warble_hz, p.warble_depth, phase=float(rng.uniform(0, 6.28)))
    bend = S.lerp(t, [(0.0, 0.97), (0.4 * dur, 1.08 + 0.10 * rng.random()), (dur, 1.00)])
    freq = f0 * bend * warble
    env = S.expdecay(t, attack_s=_attack(p, dur, snappy=0.7), decay_s=dur * 0.55)
    # softer than greet: gut quackiness and brightness
    p_soft = replace(p, quackiness=p.quackiness * 0.2, brightness=p.brightness * 0.5,
                     formant_gain=p.formant_gain * 0.5)
    sig = _voice(p_soft, t, freq, rng, breath_scale=0.4) * env
    return S.normalise(sig, peak_dbfs=-6.0)


def coo(p: Personality, variant: int = 0) -> np.ndarray:
    rng = p.variant_rng("coo", variant)
    dur = (0.85 + 0.55 * rng.random()) / p.speed
    t = S.t_axis(dur)
    # well below center — drowsy ducks drop further
    f0 = p.pitch_center_hz * (0.42 + 0.15 * (1 - p.attack_sharpness))
    drift_a = 1.0 + 0.05 * rng.random() + 0.04 * p.glide_bias
    freq = S.lerp(t, [(0.0, f0 * 0.94), (dur * 0.5, f0 * drift_a), (dur, f0 * 0.90)])
    env = S.bell(t, attack_s=dur * (0.18 + 0.10 * (1 - p.attack_sharpness)),
                 hold_s=dur * 0.5, release_s=dur * 0.30)
    # breathier, much slower modulation, no buzz
    p_soft = replace(p,
                     breath=max(p.breath, 0.12) + 0.10,
                     quackiness=p.quackiness * 0.25,
                     am_rate_hz=p.am_rate_hz * 0.30,
                     vibrato_rate_hz=p.vibrato_rate_hz * 0.45,
                     vibrato_depth=p.vibrato_depth * 0.7)
    sig = _voice(p_soft, t, freq, rng) * env
    return S.normalise(sig, peak_dbfs=-5.0)


RECIPES = {
    "alarm": alarm,
    "greet": greet,
    "inquire": inquire,
    "peck": peck,
    "chirp": chirp,
    "coo": coo,
}

VARIANT_COUNT = {
    "alarm": 2,
    "greet": 3,
    "inquire": 2,
    "peck": 2,
    "chirp": 3,
    "coo": 2,
}
