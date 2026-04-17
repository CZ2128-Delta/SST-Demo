import os
import numpy as np
import matplotlib.pyplot as plt
import sam_utils
import cv2
import io
import argparse
from PIL import Image

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
parser.add_argument('--query_images', type=str, help='Path to the query images folder.')
parser.add_argument('--output', type=str, help='Path to the output folder.')
parser.add_argument('--output_format', choices=["png", "gif"], default='gif', help='Output format (optional): gif, png.')

args = parser.parse_args()
support_image_paths = args.support_image
support_mask_paths = args.support_mask
query_images_folder = args.query_images
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

# load the query images
query_images = sorted(os.listdir(query_images_folder))
query_images = [cv2.imread(os.path.join(query_images_folder, img))[..., ::-1] for img in query_images]

# build the predictor
video_predictor = sam_utils.build_sam2_predictor()

# load the support image and mask
print ("Inferring the masks...")
state = sam_utils.load_masks(
    video_predictor, query_images, support_images, support_masks_list, verbose=True
)
frames_info = sam_utils.propagate_masks(video_predictor, state, verbose=True)
frames_info_queries = frames_info[num_support:]

# visualize the results
output_imgs = []
print ("Visualizing the results...")
for i, frame in enumerate(frames_info_queries):
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(query_images[i])
    out_masks = frame['segmentation']
    out_masks = [cv2.resize(mask.astype(np.uint8), (query_images[i].shape[1], query_images[i].shape[0])) for mask in out_masks]
    obj_ids = frame['obj_ids']
    for j, mask in enumerate(out_masks):
        sam_utils.show_mask(mask, plt.gca(), obj_ids[j], borders=True, alpha=0.75)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    output_imgs.append(img)

# save the output
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if output_format == 'gif':
    output_imgs[0].save(os.path.join(output_folder, "out.gif"), save_all=True, append_images=output_imgs[1:], loop=0, duration=1000)
else:
    for i, img in enumerate(output_imgs):
        img.save(os.path.join(output_folder, f"{i:06d}.png"))

print ("Done! The output is saved in", output_folder)

