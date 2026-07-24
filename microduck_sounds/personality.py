"""A `Personality` derives stable per-robot vocal traits from a single seed.

Two robots with different seeds sound recognisably different; the same
robot is consistent across runs. Variants within a tag re-roll a small
sub-seed so the duck doesn't sound like a stuck recording.

The trait set is deliberately wide: register (octave shift), harmonic
tilt, formant emphasis, glide bias, quackiness — each one alone is
enough to make two seeds feel like different creatures.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Personality:
    seed: int

    # --- pitch ---
    pitch_center_hz: float       # base voice pitch
    register: float              # -1..+1, extra octave-ish shift on top of center
    pitch_spread: float          # 0..1, how dramatic glides are
    glide_bias: float            # -1..+1, negative = falls, positive = rises

    # --- timbre ---
    brightness: float            # 0..1, harmonic rolloff (1 = bright/buzzy)
    tilt: float                  # 1.4..2.8, exponent on harmonic decay (higher = darker)
    nasal: float                 # 0..1, emphasis on 2nd/3rd harmonic
    harmonic_skew: float         # -1..+1, negative = odd-only (square-ish), positive = even-leaning
    formant_n: int               # 1..5, which harmonic the formant boosts
    formant_gain: float          # 0..1.5, formant strength

    # --- modulation ---
    vibrato_rate_hz: float       # 3..10
    vibrato_depth: float         # 0..0.7 semitones
    jitter_depth: float          # 0..0.4 semitones, random pitch wobble
    breath: float                # 0..0.35, noise mix
    quackiness: float            # 0.2..1, blends pure-tone vs am-buzz
    am_rate_hz: float            # 18..55, quack/croak-buzz rate (only matters with quackiness > 0)
    am_depth: float              # 0..0.7, modulation depth
    warble_hz: float             # 7..18, trill on chirp
    warble_depth: float          # 0..1.5 semitones

    # --- timing ---
    attack_sharpness: float      # 0..1, 0 = soft pad, 1 = snappy
    speed: float                 # 0.8..1.25, global tempo multiplier

    @classmethod
    def from_seed(cls, seed: int) -> "Personality":
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        u = rng.uniform

        # Bimodal-ish register: some ducks smaller & higher, some big & low —
        # but the whole population sits low, duck/toad territory.
        register = float(rng.choice([-1, 0, 0, 1])) + float(u(-0.4, 0.4))
        base = float(u(160.0, 380.0))
        pitch = float(np.clip(base * (2.0 ** (register * 0.45)), 110.0, 620.0))

        return cls(
            seed=int(seed),
            pitch_center_hz=pitch,
            register=register,
            pitch_spread=float(u(0.4, 1.2)),
            glide_bias=float(u(-1.0, 1.0)),

            brightness=float(u(0.05, 0.55)),
            tilt=float(u(1.4, 2.8)),
            nasal=float(u(0.1, 1.0)),
            harmonic_skew=float(u(-1.0, 1.0)),
            formant_n=int(rng.integers(1, 6)),
            formant_gain=float(u(0.0, 1.4)),

            vibrato_rate_hz=float(u(3.5, 9.5)),
            vibrato_depth=float(u(0.0, 0.7)),
            jitter_depth=float(u(0.03, 0.35)),
            breath=float(u(0.0, 0.30)),
            quackiness=float(u(0.2, 1.0)),
            am_rate_hz=float(u(18.0, 55.0)),
            am_depth=float(u(0.15, 0.70)),
            warble_hz=float(u(7.0, 18.0)),
            warble_depth=float(u(0.0, 1.4)),

            attack_sharpness=float(u(0.0, 1.0)),
            speed=float(u(0.82, 1.22)),
        )

    def variant_rng(self, tag: str, variant: int) -> np.random.Generator:
        """Stable per-(seed, tag, variant) RNG for sub-randomisation.

        crc32, not hash(): str hashing is salted per process, which would
        re-roll every variant on each regeneration of the bank.
        """
        h = (self.seed * 1_000_003) ^ zlib.crc32(tag.encode()) ^ (variant * 2654435761)
        return np.random.default_rng(h & 0xFFFFFFFF)

    def harmonics(self) -> list[float]:
        """Personality-shaped harmonic weights for the main oscillator.

        Combines four effects:
          - tilt: overall rolloff (darker → steeper)
          - brightness: lifts the high-end tail
          - nasal: lifts 2nd/3rd
          - harmonic_skew: even-vs-odd preference
          - formant_n: a bump at one chosen harmonic
        """
        n_harm = 7
        weights: list[float] = []
        for n in range(1, n_harm + 1):
            base = 1.0 / (n ** self.tilt)
            high_lift = self.brightness * (n / n_harm) ** 1.5
            nasal = self.nasal * (1.0 if n in (2, 3) else 0.0) * 0.6
            if self.harmonic_skew >= 0:
                skew = self.harmonic_skew * (0.4 if n % 2 == 0 else -0.2)
            else:
                skew = -self.harmonic_skew * (-0.3 if n % 2 == 0 else 0.4)
            formant = self.formant_gain if n == self.formant_n else 0.0
            weights.append(max(0.0, base + high_lift + nasal + skew + formant * base * 1.5))
        # keep f0 dominant so pitch is always perceptible
        weights[0] = max(weights[0], 0.7)
        return weights
