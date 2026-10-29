"""Frame extraction and COLMAP dense reconstruction."""
import shutil
import subprocess
from pathlib import Path

import cv2


def extract_frames(video_path, out_dir, every_n=5, blur_thresh=100.0, max_frames=300):
    """Keep every n-th frame whose Laplacian variance (sharpness) passes the threshold."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    kept, i = 0, 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_thresh:
                cv2.imwrite(str(out / f"frame_{kept:04d}.jpg"), frame)
                kept += 1
        i += 1
    cap.release()
    return kept


def run_colmap(image_dir, workspace, quality="medium"):
    """Run COLMAP automatic reconstruction. Returns path to the Poisson mesh."""
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP not found on PATH. Install it first.")
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["colmap", "automatic_reconstructor",
         "--workspace_path", str(ws), "--image_path", str(image_dir),
         "--quality", quality, "--single_camera", "1", "--dense", "1"],
        check=True,
    )
    mesh = ws / "dense" / "0" / "meshed-poisson.ply"
    if not mesh.exists():
        raise RuntimeError("COLMAP finished but no mesh was produced (dense step needs CUDA).")
    return mesh
