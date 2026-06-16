import argparse
import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
COLS = 5


def _natural_sort_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _collect_images(folder):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files, key=_natural_sort_key)


def main():
    parser = argparse.ArgumentParser(description="Visualize images in an output folder.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder containing images.")
    parser.add_argument("--alpha_val", type=float, default=0.75,
                        help="Overlay alpha value (default: 0.75).")
    args = parser.parse_args()

    images = _collect_images(args.output_folder)
    if not images:
        print(f"No images found in {args.output_folder}")
        return

    n = len(images)
    rows = math.ceil(n / COLS)
    col_width = 4

    fig, axes = plt.subplots(
        rows, COLS, figsize=(col_width * COLS, col_width * rows), layout="constrained"
    )
    axes = np.atleast_1d(axes).reshape(rows, COLS)

    for idx, path in enumerate(images):
        r, c = divmod(idx, COLS)
        ax = axes[r, c]
        img = np.array(Image.open(path).convert("RGB"))
        ax.imshow(img)
        ax.set_title(os.path.basename(path), fontsize=10)
        ax.axis("off")

    for idx in range(n, rows * COLS):
        r, c = divmod(idx, COLS)
        axes[r, c].axis("off")

    fig.suptitle("Results", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
