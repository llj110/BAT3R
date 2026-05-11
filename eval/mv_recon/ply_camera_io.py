"""
在标准 PLY 头（end_header 前）插入 comment 行，嵌入 np.save 格式的 camera_poses (N,4,4) 的 base64。
Open3D / MeshLab 仍只读 vertex，忽略 comment；view_ply_local 等可解析。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

BEGIN = "comment TTT3R_CAMERAS_B64_BEGIN"
END = "comment TTT3R_CAMERAS_B64_END"
DATA_PREFIX = "comment TTT3R_CAM_DATA "


def inject_camera_poses_into_ply(ply_path: str | Path, camera_poses: np.ndarray) -> None:
    ply_path = Path(ply_path)
    buf = io.BytesIO()
    np.save(buf, np.asarray(camera_poses, dtype=np.float64))
    raw = buf.getvalue()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    lines = [BEGIN]
    for i in range(0, len(b64), 120):
        lines.append(f"{DATA_PREFIX}{b64[i : i + 120]}")
    lines.append(END)
    inserted = ("\n".join(lines) + "\n").encode("ascii")

    with open(ply_path, "rb") as f:
        data = f.read()
    idx = data.find(b"end_header")
    if idx < 0:
        return
    new_data = data[:idx] + inserted + data[idx:]
    with open(ply_path, "wb") as f:
        f.write(new_data)
