import os
import numpy as np
import matplotlib.cm as cm
import sam_utils
import cv2
import argparse
from PIL import Image


def overlay_masks(img, masks, obj_ids, alpha=0.75, borders=True):
    result = img.copy()
    cmap = cm.get_cmap("tab10")
    for j, mask in enumerate(masks):
        color = (np.array(cmap(obj_ids[j])[:3]) * 255).astype(np.uint8)
        m = mask.astype(bool)
        result[m] = (
            result[m].astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
        ).astype(np.uint8)
        if borders:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            border_color = tuple(int(c) for c in color)
            cv2.drawContours(result, contours, -1, border_color, thickness=2)
    return result

# parse the arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--support_image',
    type=str,
    nargs='+',
    help='Path(s) to support image(s). Pass one path (single-shot) or multiple paths (few-shot); order matches --support_mask.',
)
parser.add_argument(
    '--support_mask',
    type=str,
    nargs='+',
    help='Path(s) to support mask(s), same count as --support_image. Each mask uses labels 1..N per object (same as before).',
)
parser.add_argument('--target_images', type=str, help='Path to the target images folder.')
parser.add_argument('--output', type=str, help='Path to the output folder.')
parser.add_argument('--output_format', choices=["png", "gif"], default='gif', help='Output format (optional): gif, png.')

args = parser.parse_args()
support_image_paths = args.support_image
support_mask_paths = args.support_mask
target_images_folder = args.target_images
output_folder = args.output
output_format = args.output_format

if len(support_image_paths) != len(support_mask_paths):
    parser.error('--support_image and --support_mask must have the same number of paths.')

# load the support image(s) and mask(s)
print('Loading support image(s) and mask(s)...')
support_images = [cv2.imread(p)[..., ::-1] for p in support_image_paths]
support_masks_list = []
for p in support_mask_paths:
    support_mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    support_masks_list.append(
        [support_mask == i for i in range(1, int(support_mask.max()) + 1)]
    )
num_support = len(support_images)

# load the target images
target_images = sorted(os.listdir(target_images_folder))
target_images = [cv2.imread(os.path.join(target_images_folder, img))[..., ::-1] for img in target_images]

# build the predictor
video_predictor = sam_utils.build_sam2_predictor()

# load the support image and mask
print ("Inferring the masks...")
state = sam_utils.load_masks(
    video_predictor, target_images, support_images, support_masks_list, verbose=True
)
frames_info = sam_utils.propagate_masks(video_predictor, state, verbose=True)
frames_info_queries = frames_info[num_support:]

# visualize the results
output_imgs = []
print ("Visualizing the results...")
for i, frame in enumerate(frames_info_queries):
    out_masks = frame['segmentation']
    out_masks = [
        cv2.resize(mask.astype(np.uint8), (target_images[i].shape[1], target_images[i].shape[0]))
        for mask in out_masks
    ]
    obj_ids = frame['obj_ids']
    vis_img = overlay_masks(target_images[i], out_masks, obj_ids, alpha=0.75, borders=True)
    output_imgs.append(Image.fromarray(vis_img))

# save the output
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if output_format == 'gif':
    output_imgs[0].save(os.path.join(output_folder, "out.gif"), save_all=True, append_images=output_imgs[1:], loop=0, duration=1000)
else:
    for i, img in enumerate(output_imgs):
        img.save(os.path.join(output_folder, f"{i:06d}.png"))

print ("Done! The output is saved in", output_folder)

