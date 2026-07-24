"""CLI: play one tag, render-all variants, or audition the whole repertoire."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import api
from . import parrot
from .personality import Personality
from .voices import VARIANT_COUNT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="microduck-sounds")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("play", help="synthesize and play one sound")
    pl.add_argument("tag", choices=api.TAGS)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--variant", type=int, default=0)

    rd = sub.add_parser("render", help="write one sound to a wav file")
    rd.add_argument("tag", choices=api.TAGS)
    rd.add_argument("out", type=Path)
    rd.add_argument("--seed", type=int, default=0)
    rd.add_argument("--variant", type=int, default=0)

    ra = sub.add_parser("render-all", help="render every tag x variant into a directory")
    ra.add_argument("out_dir", type=Path)
    ra.add_argument("--seed", type=int, default=0)

    au = sub.add_parser("audition", help="play every tag x variant in sequence")
    au.add_argument("--seed", type=int, default=0)

    sh = sub.add_parser("show", help="print the personality traits for a seed")
    sh.add_argument("--seed", type=int, default=0)

    pa = sub.add_parser("parrot", help="listen, learn repeated phrases, squawk them back")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--memory-dir", type=Path, default=parrot.DEFAULT_MEMORY)
    pa.add_argument("--repeats", type=int, default=3,
                    help="times a phrase must be heard before it's learned")
    pa.add_argument("--threshold", type=float, default=parrot.DEFAULT_THRESHOLD,
                    help="match strictness; lower = stricter (see d= in the logs)")
    pa.add_argument("--device", help="ALSA capture device, e.g. plughw:aic3104,0")
    pa.add_argument("--stdin", action="store_true",
                    help="read raw S16LE 16kHz mono from stdin instead of arecord")
    pa.add_argument("--wav", type=Path, help="process a wav file instead of the mic (testing)")
    pa.add_argument("--babble", type=float, default=240.0,
                    help="seconds of quiet before the parrot babbles a learned phrase")
    pa.add_argument("--out-rate", type=int, default=parrot.DEFAULT_OUT_RATE)
    pa.add_argument("--mute", action="store_true", help="log only, never play audio")

    ps = sub.add_parser("parrot-say", help="parrot-voice a wav (or a random learned phrase)")
    ps.add_argument("wav", nargs="?", type=Path)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--memory-dir", type=Path, default=parrot.DEFAULT_MEMORY)
    ps.add_argument("--out-rate", type=int, default=parrot.DEFAULT_OUT_RATE)

    args = parser.parse_args(argv)

    if args.cmd == "play":
        api.play(args.tag, args.seed, variant=args.variant)
    elif args.cmd == "render":
        buf = api.render(args.tag, args.seed, variant=args.variant)
        api.to_wav(buf, args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "render-all":
        paths = api.render_all(args.seed, args.out_dir)
        print(f"wrote {len(paths)} files under {args.out_dir}")
    elif args.cmd == "audition":
        for tag in api.TAGS:
            for v in range(VARIANT_COUNT[tag]):
                print(f"  {tag} v{v}")
                api.play(tag, args.seed, variant=v)
    elif args.cmd == "show":
        p = Personality.from_seed(args.seed)
        for k, v in p.__dict__.items():
            print(f"  {k:18s} {v}")
    elif args.cmd == "parrot":
        return parrot.run(seed=args.seed, memory_dir=args.memory_dir,
                          repeats=args.repeats, threshold=args.threshold,
                          wav=args.wav, use_stdin=args.stdin, device=args.device,
                          babble_s=args.babble, out_rate=args.out_rate,
                          mute=args.mute)
    elif args.cmd == "parrot-say":
        return parrot.say(wav=args.wav, seed=args.seed,
                          memory_dir=args.memory_dir, out_rate=args.out_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
