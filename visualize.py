import argparse
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from IPython.display import HTML, display as ipy_display

    def display_divider():
        ipy_display(
            HTML(
                "<hr style='border: none; border-top: 2px solid #cccccc; margin: 28px 0;'>"
            )
        )
except ImportError:
    def display_divider():
        print("\n" + "─" * 80 + "\n")


def _load_overlay_pair(img_path: str, msk_path: str, alpha_val: float):
    if not os.path.isfile(img_path):
        print(f"Error: Image not found at {img_path}")
        return None, None
    if not os.path.isfile(msk_path):
        print(f"Error: Mask not found at {msk_path}")
        return None, None

    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)

    mask = np.array(Image.open(msk_path).convert("L"))
    mask_resized = np.array(
        Image.fromarray(mask).resize(
            (img_array.shape[1], img_array.shape[0]), resample=Image.NEAREST
        )
    )

    cmap = plt.get_cmap("tab10")
    colored_mask = np.zeros_like(img_array, dtype=np.uint8)
    for label_id in range(1, int(mask_resized.max()) + 1):
        colored_mask[mask_resized == label_id] = (
            np.array(cmap(label_id - 1)[:3]) * 255
        ).astype(np.uint8)

    overlay = img_array.copy()
    mask_indices = mask_resized > 0
    overlay[mask_indices] = (
        alpha_val * colored_mask[mask_indices]
        + (1 - alpha_val) * img_array[mask_indices]
    ).astype(np.uint8)

    return img_array, overlay


def _find_support_pairs(input_folder: str):
    numbered_imgs = sorted(glob.glob(os.path.join(input_folder, "support_image_*")))
    numbered_msks = sorted(glob.glob(os.path.join(input_folder, "support_mask_*")))
    if numbered_imgs and numbered_msks and len(numbered_imgs) == len(numbered_msks):
        return list(zip(numbered_imgs, numbered_msks))

    imgs = sorted(glob.glob(os.path.join(input_folder, "support_image*")))
    msks = sorted(glob.glob(os.path.join(input_folder, "support_mask*")))
    if imgs and msks and len(imgs) == len(msks):
        return list(zip(imgs, msks))

    return []


def display_overlay(
    img_path,
    msk_path,
    alpha_val: float = 1.0,
    section_title: str = "Support Image & Mask",
):
    if isinstance(img_path, str) and isinstance(msk_path, str):
        pairs = [(img_path, msk_path)]
    else:
        pairs = list(zip(img_path, msk_path))

    loaded_pairs = []
    for image_path, mask_path in pairs:
        img_array, overlay = _load_overlay_pair(image_path, mask_path, alpha_val)
        if img_array is not None:
            loaded_pairs.append((image_path, img_array, overlay))

    if not loaded_pairs:
        print("No valid support image/mask pairs to display.")
        return

    num_pairs = len(loaded_pairs)
    width_ratios = []
    for _, img_array, _ in loaded_pairs:
        img_h, img_w = img_array.shape[:2]
        aspect = img_w / img_h
        width_ratios.append([aspect, aspect])

    fig, axes = plt.subplots(
        num_pairs,
        2,
        figsize=(20, 6),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": width_ratios[0]} if num_pairs == 1 else None,
    )
    fig.suptitle(section_title, fontsize=16, x=0.5)

    for row, (image_path, img_array, overlay) in enumerate(loaded_pairs):
        if num_pairs == 1:
            left_title = "Support Image"
            right_title = "Support Image + Mask"
        else:
            idx = row + 1
            left_title = f"Support Image {idx}"
            right_title = f"Support Image {idx} + Mask {idx}"

        axes[row, 0].imshow(img_array, aspect="equal")
        axes[row, 0].set_title(left_title, fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay, aspect="equal")
        axes[row, 1].set_title(right_title, fontsize=11)
        axes[row, 1].axis("off")

    plt.show()


def visualize_results(output_folder, title):
    output_images = sorted(glob.glob(os.path.join(output_folder, "*.png")))

    if not output_images:
        print(f"No output images found in {output_folder}")
        return

    num_images = len(output_images)
    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 6), squeeze=False, layout="constrained")
    fig.suptitle(title, fontsize=16, x=0.5)

    for i, image_path in enumerate(output_images):
        row, col = divmod(i, cols)
        image = cv2.imread(image_path)
        axes[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(os.path.basename(image_path), fontsize=10)
        axes[row, col].axis("off")

    for i in range(num_images, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis("off")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize trait segmentation results.")
    parser.add_argument("--input_folder", type=str, default=None,
                        help="Path to the input folder containing support image/mask pairs.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder containing segmentation results.")
    parser.add_argument("--alpha_val", type=float, default=0.75,
                        help="Overlay alpha value (default: 0.75).")
    parser.add_argument("--first_support_only", action="store_true",
                        help="Only visualize the first support pair instead of all.")
    args = parser.parse_args()

    if args.input_folder:
        pairs = _find_support_pairs(args.input_folder)
        if args.first_support_only:
            pairs = pairs[:1]
        if pairs:
            display_overlay(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                alpha_val=args.alpha_val,
                section_title="Support Image(s) & Mask(s)",
            )
        else:
            print(f"No support pairs found in {args.input_folder}")
        display_divider()

    visualize_results(args.output_folder, "Results")


if __name__ == "__main__":
    main()
