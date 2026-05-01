"""CLI: play one tag, render-all variants, or audition the whole repertoire."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import api
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
