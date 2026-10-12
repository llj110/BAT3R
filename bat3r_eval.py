#!/usr/bin/env python3
"""BAT3R evaluation entry point; forwards options to existing evaluators."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
TASKS = {
    "relpose": "eval/relpose/launch.py",
    "video_depth": "eval/video_depth/launch.py",
    "mv_recon": "eval/mv_recon/launch.py",
    "ply": "eval/mv_recon/launch_kitti_scannet_ply.py",
}


def build_command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--dry-run", action="store_true", help="Print arguments without loading a model")
    args, forwarded = parser.parse_known_args(argv)
    for option, default in (
        ("--model_update_type", "bat3r"),
        ("--output_dir", str(ROOT / "eval_results" / "bat3r" / args.task)),
    ):
        if not any(token == option or token.startswith(option + "=") for token in forwarded):
            forwarded += [option, default]
    return args, ROOT / TASKS[args.task], forwarded


def main(argv=None):
    args, launcher, forwarded = build_command(argv)
    if args.dry_run:
        import json
        print(json.dumps({"launcher": str(launcher), "arguments": forwarded}, indent=2))
        return
    os.chdir(ROOT)
    sys.path[:0] = [str(launcher.parent), str(ROOT), str(ROOT / "src")]
    sys.argv = [str(launcher), *forwarded]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
