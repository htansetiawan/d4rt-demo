#!/usr/bin/env python3
"""Fetch pinned OpenD4RT source, an example, and optionally the 14 GB weights."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/Lijiaxin0111/Open-d4rt.git"
SOURCE_COMMIT = "403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3"
HF_REPO = "Lijiaxin0111/OpenD4RT"
HF_REVISION = "7099b1fc760475de3b7409acd6b63f801b015d07"
WEIGHT_DIR = "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG"
EXAMPLES = {"juggle_5": "13ZlJ-U1qNPafVmYyiLN9kGY8Lk7Zyy_p",
            "softball_25": "1ZQRdlE39l5BSaNgVh9eE6tVrEsl6sCj4"}


def setup(source=True, weights=False, example=None):
    if source:
        target = ROOT / ".cache/Open-d4rt"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(target)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", SOURCE_URL], check=True)
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", SOURCE_COMMIT], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
        found = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if found != SOURCE_COMMIT:
            raise RuntimeError(f"Source revision differs: {found}. Expected {SOURCE_COMMIT}.")
        print(f"OpenD4RT source: {found}")
    if example:
        import gdown
        target = ROOT / f"data/worldtrack/pstudio_mini/{example}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            temporary = target.with_suffix(".npz.part")
            result = gdown.download(id=EXAMPLES[example], output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError("Download failed. Use the WorldTrack folder linked in README.")
            temporary.replace(target)
        # Validate the download and decode RGB; do not access ground-truth tracks.
        sys.path.insert(0, str(ROOT))
        from d4rt_demo import load_video
        video = load_video(target)
        print(f"Example: {target} · {len(video)} RGB frames decoded")
    if weights:
        from huggingface_hub import hf_hub_download
        print("Downloading the released checkpoint (~14 GB) and its matching config.")
        for name in ("model.yaml", "opend4rt.ckpt"):
            path = hf_hub_download(HF_REPO, f"{WEIGHT_DIR}/{name}", revision=HF_REVISION, local_dir=ROOT)
            print(path)
    print("Ready. Open D4RT_demo.ipynb using this Python environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="Download the released ~14 GB checkpoint.")
    parser.add_argument("--example", nargs="?", const="juggle_5", choices=EXAMPLES)
    parser.add_argument("--no-source", action="store_true")
    args = parser.parse_args()
    setup(source=not args.no_source, weights=args.weights, example=args.example)
