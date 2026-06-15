import argparse
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


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

    cmap = matplotlib.colormaps["tab10"]
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


def _find_support_pairs(support_folder: str):
    numbered_imgs = sorted(glob.glob(os.path.join(support_folder, "support_image_*")))
    numbered_msks = sorted(glob.glob(os.path.join(support_folder, "support_mask_*")))
    if numbered_imgs and numbered_msks and len(numbered_imgs) == len(numbered_msks):
        return list(zip(numbered_imgs, numbered_msks))

    imgs = sorted(glob.glob(os.path.join(support_folder, "support_image*")))
    msks = sorted(glob.glob(os.path.join(support_folder, "support_mask*")))
    if imgs and msks and len(imgs) == len(msks):
        return list(zip(imgs, msks))

    return []


def _collect_images(folder):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Visualize trait segmentation results.")
    parser.add_argument("--support_folder", type=str, default=None,
                        help="Path to the folder containing support image/mask pairs.")
    parser.add_argument("--input_folder", type=str, default=None,
                        help="Path to the folder containing the original input images "
                             "(correspond to output images by file order).")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output folder containing segmentation results.")
    parser.add_argument("--alpha_val", type=float, default=0.75,
                        help="Overlay alpha value (default: 0.75).")
    parser.add_argument("--first_support_only", action="store_true",
                        help="Only visualize the first support pair instead of all.")
    args = parser.parse_args()

    output_images = _collect_images(args.output_folder)
    if not output_images:
        print(f"No output images found in {args.output_folder}")
        return

    has_input = args.input_folder is not None
    input_images = _collect_images(args.input_folder) if has_input else []
    num_io = len(output_images)

    support_data = []
    if args.support_folder:
        pairs = _find_support_pairs(args.support_folder)
        if args.first_support_only:
            pairs = pairs[:1]
        for img_path, msk_path in pairs:
            img_array, overlay = _load_overlay_pair(img_path, msk_path, args.alpha_val)
            if img_array is not None:
                support_data.append((img_array, overlay))

    num_support = len(support_data)
    has_support = num_support > 0
    sep_cols = 1 if has_support else 0
    total_cols = num_support + sep_cols + num_io
    num_rows = 2 if (has_input or has_support) else 1

    width_ratios = [1.0] * num_support
    if has_support:
        width_ratios.append(0.02)
    width_ratios.extend([1.0] * num_io)

    col_width = 4
    fig_width = col_width * (num_support + num_io) + (0.3 if has_support else 0)
    fig_height = col_width * num_rows + 1

    plt.close("all")
    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    gs = gridspec.GridSpec(num_rows, total_cols, figure=fig, width_ratios=width_ratios)

    col = 0

    for i, (img_array, overlay) in enumerate(support_data):
        label = "Support Image" if num_support == 1 else f"Support Image {i + 1}"
        mask_label = "Support Mask" if num_support == 1 else f"Support Mask {i + 1}"

        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(img_array)
        ax_top.set_title(label, fontsize=10)
        ax_top.axis("off")

        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.imshow(overlay)
        ax_bot.set_title(mask_label, fontsize=10)
        ax_bot.axis("off")
        col += 1

    if has_support:
        ax_sep = fig.add_subplot(gs[:, col])
        ax_sep.axvline(x=0.5, color="black", linewidth=1.5)
        ax_sep.set_xlim(0, 1)
        ax_sep.set_ylim(0, 1)
        ax_sep.set_xticks([])
        ax_sep.set_yticks([])
        for spine in ax_sep.spines.values():
            spine.set_visible(False)
        col += 1

    for i in range(num_io):
        out_img = np.array(Image.open(output_images[i]).convert("RGB"))

        if has_input and i < len(input_images):
            inp_img = np.array(Image.open(input_images[i]).convert("RGB"))
            ax_top = fig.add_subplot(gs[0, col])
            ax_top.imshow(inp_img)
            ax_top.set_title(os.path.basename(input_images[i]), fontsize=10)
            ax_top.axis("off")
            out_row = 1
        elif num_rows == 2:
            ax_top = fig.add_subplot(gs[0, col])
            ax_top.axis("off")
            out_row = 1
        else:
            out_row = 0

        ax_bot = fig.add_subplot(gs[out_row, col])
        ax_bot.imshow(out_img)
        ax_bot.set_title(os.path.basename(output_images[i]), fontsize=10)
        ax_bot.axis("off")
        col += 1

    fig.suptitle("Results", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
