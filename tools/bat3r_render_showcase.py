"""Render looping README previews from the project's saved BAT3R point clouds.

Run with Python, numpy and Pillow; no model inference or GPU is required.
The source positions and RGB values are preserved. Each preview uses a fixed
orthographic camera orbit, with a common palette across all animation frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://llj110.github.io/bat3r/assets/data/"
SCENES = ("7s-chess_seq-03", "7s-stairs_seq-01")
SIZE = (640, 420)
FRAMES = 120


def read_asset(path, source_dir):
    if source_dir:
        return (source_dir / path).read_bytes()
    with urlopen(BASE + path, timeout=30) as response:
        return response.read()


def render(scene_id, source_dir, output):
    meta = json.loads(read_asset(scene_id + "/scene.json", source_dir))
    method = meta["methods"]["bat3r"]
    raw = read_asset(method["file"], source_dir)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != method["sha256"] or len(raw) != method["points"] * 15:
        raise ValueError("Point-cloud checksum or size mismatch: " + scene_id)
    n = method["points"]
    xyz = np.frombuffer(raw, dtype="<f4", count=n * 3).reshape(n, 3).astype(float)
    rgb = np.frombuffer(raw, dtype="u1", offset=n * 12).reshape(n, 3)
    center = np.asarray(meta["center"])
    q = xyz - center
    q[:, 1:] *= -1  # Match the project viewer's world-coordinate convention.
    extent = np.percentile(np.linalg.norm(q, axis=1), 98.5)
    scale = 155 / max(extent, 1e-6)
    elevation = np.deg2rad(23)
    rendered = []
    for frame in range(FRAMES):
        angle = .65 + 2 * np.pi * frame / FRAMES
        right = np.array([np.cos(angle), 0, -np.sin(angle)])
        up = np.array([-np.sin(elevation) * np.sin(angle), np.cos(elevation),
                       -np.sin(elevation) * np.cos(angle)])
        towards = np.cross(right, up)
        # Render at 2x resolution for antialiasing, back-to-front occlusion.
        points = np.column_stack((q @ right, -(q @ up))) * scale * 2
        points += [SIZE[0], SIZE[1] + 24]
        pixels = np.rint(points).astype(int)
        order = np.argsort(q @ towards, kind="stable")
        pixels, colors = pixels[order], rgb[order]
        valid = ((pixels[:, 0] >= 2) & (pixels[:, 0] < SIZE[0] * 2 - 3)
                 & (pixels[:, 1] >= 90) & (pixels[:, 1] < SIZE[1] * 2 - 52))
        pixels, colors = pixels[valid], colors[valid]
        canvas = np.full((SIZE[1] * 2, SIZE[0] * 2, 3), [240, 242, 250], np.uint8)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
            canvas[pixels[:, 1] + dy, pixels[:, 0] + dx] = colors
        im = Image.fromarray(canvas).resize(SIZE, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(im)
        draw.rounded_rectangle((16, 14, 82, 42), radius=6, fill="white")
        draw.text((28, 22), "BAT3R", fill="#505d89")
        draw.text((100, 22), meta["name"], fill="#505d89")
        draw.text((18, SIZE[1] - 23), f'{meta["frames"]} input frames  /  saved reconstruction', fill="#65729a")
        rendered.append(im)
    # Shared palette avoids frame-to-frame color flicker.
    samples = Image.new("RGB", (SIZE[0] * 4, SIZE[1]))
    for i, index in enumerate((0, 30, 60, 90)):
        samples.paste(rendered[index], (i * SIZE[0], 0))
    palette = samples.quantize(colors=256)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in rendered]
    target = output / ("bat3r_" + scene_id + "_rotate.gif")
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=False, disposal=1)
    rendered[0].save(output / ("bat3r_" + scene_id + "_poster.png"))
    print(f"{scene_id}: {n:,} points, {FRAMES} frames, {target.stat().st_size / 1e6:.2f} MB", flush=True)
    return {"scene": scene_id, "dataset": meta["dataset"], "name": meta["name"],
            "input_frames": meta["frames"], "displayed_points": n,
            "source": BASE + method["file"], "source_sha256": digest,
            "metadata": BASE + scene_id + "/scene.json",
            "gif": target.name, "animation_frames": FRAMES, "duration_ms": 100,
            "loop": "infinite", "projection": "fixed orthographic camera orbit",
            "sampling": meta["sampling"], "alignment": meta["alignment"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Optional local project-site assets/data directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "media" / "showcase")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"project": "BAT3R", "source_note": "Saved benchmark exports, also displayed on the project website. These qualitative previews are not a new inference run or a claim of the best paper configuration.",
                "scenes": [render(s, args.source_dir, args.output_dir) for s in SCENES]}
    (args.output_dir / "bat3r_showcase.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
