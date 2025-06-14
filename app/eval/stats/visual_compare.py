import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

from app.eval.stats.stat_helpers import gather_dataset, normalize_map, calc_seg_stats
from app.models.u2net.u2_wrapper import U2NetWrapper, U2NetPWrapper
from app.models.samnet.samnet_wrapper import SAMNetWrapper
from app.models.custom_models import BMS, IttiKoch
import cv2.saliency as saliency
import pysaliency as pys

# Setup detectors
DETECTORS = {
    "U^2-Net": U2NetWrapper(weights_path="app/models/u2net/u2net.pth"),
    "U^2-NetP": U2NetPWrapper(weights_path="app/models/u2net/u2netp.pth"),
    "SAM-Net": SAMNetWrapper(weights_path="app/models/samnet/SAMNet_with_ImageNet_pretrain.pth"),
    "AIM": pys.AIM(location="app/models/pysal"),
    "SUN": pys.SUN(location="app/models/pysal"),
    "BMS": BMS(),
    "IKN": IttiKoch(),
    "SpectralRes": saliency.StaticSaliencySpectralResidual.create(),
    "FineGrained": saliency.StaticSaliencyFineGrained.create(),
}

def compute_saliency(detector, img):
    if hasattr(detector, "computeSaliency"):
        success, sal_map = detector.computeSaliency(img)
    # Use saliency_map for pysal models
    else:
        sal_map = detector.saliency_map(img)
        success = sal_map is not None
    return success, normalize_map(sal_map) if success else None

def visualize_single_sample(dataset, detectors, out_path="output_preview.png", max_cols=4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fn, img, gt = random.choice(dataset)
    pred_images = []
    titles = []

    # Original and GT
    pred_images.append(img)
    titles.append("Original")

    gt_rgb = np.stack([gt * 255]*3, axis=-1).astype(np.uint8)
    pred_images.append(gt_rgb)
    titles.append("Ground Truth")

    for name, detector in detectors.items():
        success, sal = compute_saliency(detector, img)
        if not success or sal is None:
            print(f"Skipping {name} — failed inference.")
            continue

        pred_mask = (sal >= 0.5).astype(np.uint8)
        stats = calc_seg_stats(pred_mask, gt)
        iou = stats["iou"]
        dice = stats["dice"]

        # Format as 3-channel binary mask
        seg_vis = np.stack([pred_mask * 255]*3, axis=-1).astype(np.uint8)
        caption = f"{name}: IoU={iou:.2f}, Dice={dice:.2f}"

        pred_images.append(seg_vis)
        titles.append(caption)

    # Plot layout
    n_images = len(pred_images)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_images > 1 else [axes]

    for ax, img, title in zip(axes, pred_images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Hide any unused axes
    for ax in axes[len(pred_images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    dataset = gather_dataset("data/COCO/annotations/instances_val2017.json", "data/COCO/val2017")
    visualize_single_sample(dataset, DETECTORS, out_path="output/sample_tile_grid.png")
