import numpy as np
import cv2
from tqdm import tqdm
import os
from pycocotools.coco import COCO

import pandas as pd
from scipy import stats
import numpy as np
from itertools import combinations

def gather_dataset(coco_annotation_file: str, img_dir: str):
    # Load validation images & binary masks using COCO API
    coco = COCO(coco_annotation_file)
    img_ids = coco.getImgIds()  # all validation images
    print(f"Found {len(img_ids)} images in COCO val set.")

    dataset = []
    for img_id in tqdm(img_ids, desc="Building COCO dataset", unit="img"):
        meta = coco.loadImgs(img_id)[0]
        fn = meta['file_name']
        path = os.path.join(img_dir, fn)
        # read image
        img = cv2.imread(path)
        if img is None:
            # skip missing files
            continue
        h, w = meta['height'], meta['width']

        # load all instance annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # build a single binary mask: union of all object masks
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            # annToMask returns a HxW uint8 array (1 inside segment)
            m = coco.annToMask(ann)
            mask = np.logical_or(mask, m)

        dataset.append((fn, img, mask))

    print(f"→ Loaded {len(dataset)} image/mask pairs.\n")
    return dataset


def normalize_map(sal_map):
    map = sal_map.astype(np.float32)
    # Compute minimum and maximum only once
    m_min = map.min()
    m_max = map.max()
    # Make 0 new min, scale range of map to be 1
    return (map - m_min) / (m_max - m_min + 1e-12)

def calc_seg_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Calculate segmentation metrics between binary prediction and ground truth.
    Returns dict with 'precision', 'recall', 'iou', 'dice'.
    """
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    # avoid division by zero
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    iou  = tp / (tp + fp + fn + 1e-12)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    time = 0
    return {'dice': dice, 'iou': iou,  'time': time,
             'precision': prec, 'recall': rec,
             'f1-score': f1, 'accuracy': acc, }

def compare_models(model_metrics, alpha=0.05, d_thresh=0.2):
    # Compare models pairwise using paired t-tests and Cohen's d
    comparisons = []
    
    # Group models by sample size
    groups = {}
    for model, metrics in model_metrics.items():
        n = metrics['n']
        groups.setdefault(n, []).append(model)
    
    # Compare within each sample group
    for n, models in groups.items():
        for model_a, model_b in combinations(models, 2):
            for metric in ['dice', 'iou']:
                a_scores = model_metrics[model_a][metric]
                b_scores = model_metrics[model_b][metric]
                
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
                
                # Cohen's d calculation
                diff = np.array(a_scores) - np.array(b_scores)
                d = np.mean(diff) / np.std(diff, ddof=1)
                
                comparisons.append({
                    'Model A': model_a,
                    'Model B': model_b,
                    'Metric': metric,
                    'T-stat': t_stat,
                    'p-value': p_val,
                    "Cohen's d": d,
                    'Significant': (p_val < alpha) and (abs(d) > d_thresh),
                    'Sample Size': n
                })
    
    return pd.DataFrame(comparisons)
