"""
Stitches the epoch_*.png confusion-matrix frames (from ConfusionMatrixCallback)
into a single animated GIF.

Usage:
    python3 make_gif.py checkpoints/plots/confusion --output checkpoints/plots/confusion.gif
"""

import argparse
import glob
import os
import re

from PIL import Image


def natural_epoch_key(path: str) -> int:
    match = re.search(r"epoch_(\d+)\.png$", path)
    return int(match.group(1)) if match else 0


def make_gif(frame_dir: str, output_path: str, duration_ms: int, hold_last_ms: int) -> None:
    frames = sorted(glob.glob(os.path.join(frame_dir, "epoch_*.png")), key=natural_epoch_key)
    if not frames:
        raise FileNotFoundError(f"No epoch_*.png frames found in {frame_dir}")

    images = [Image.open(f) for f in frames]
    durations = [duration_ms] * (len(images) - 1) + [hold_last_ms]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    print(f"Wrote {output_path} from {len(images)} frames ({frames[0]} -> {frames[-1]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=str, help="Directory containing epoch_*.png frames.")
    parser.add_argument("--output", type=str, default=None, help="Output .gif path (default: <frame_dir>.gif).")
    parser.add_argument("--duration_ms", type=int, default=250, help="Milliseconds per frame.")
    parser.add_argument("--hold_last_ms", type=int, default=1500, help="Milliseconds to hold the final frame.")
    args = parser.parse_args()

    output = args.output or (args.frame_dir.rstrip("/") + ".gif")
    make_gif(args.frame_dir, output, args.duration_ms, args.hold_last_ms)
