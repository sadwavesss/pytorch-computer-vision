"""Save freely licensed demo images to assets/ (scikit-image sample data and the ultralytics bus image)."""

import argparse
import shutil
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from skimage import data

    for name in ("astronaut", "chelsea"):
        cv2.imwrite(str(args.out / f"{name}.png"), cv2.cvtColor(getattr(data, name)(), cv2.COLOR_RGB2BGR))

    try:
        from ultralytics.utils import ASSETS

        shutil.copy(ASSETS / "bus.jpg", args.out / "bus.jpg")
    except ImportError:
        print("ultralytics is not installed: skipping bus.jpg")
    print(f"Demo images saved to {args.out}")


if __name__ == "__main__":
    main()
