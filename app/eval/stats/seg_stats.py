# Required libraries
import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import pysaliency as pys
from time import time

# Required classes & methods
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List

# Custom modules
from app.models.custom_models import BMS, BMSOptimized, IttiKoch
from app.models.u2net.u2_wrapper import U2NetWrapper, U2NetPWrapper
from app.models.samnet.samnet_wrapper import SAMNetWrapper
from app.config.exp_config import ExperimentConfig
from app.eval.stats.stat_helpers import (
    gather_dataset,
    normalize_map,
    compare_models,
    calc_seg_stats)

# Using dataclass avoids storing stats in nested dicts
@dataclass
class ModelStats:
    model: str
    n_images: int
    dice: float
    iou: float
    time: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float

FULL_VAL_SET = 5000
# Paths for pretrained weights & model files
PYSAL_ROOT = "app/models/pysal"
U2_WEIGHTS = "app/models/u2net/u2net.pth"
U2P_WEIGHTS = "app/models/u2net/u2netp.pth"
SAM_WEIGHTS = "app/models/samnet/SAMNet_with_ImageNet_pretrain.pth"

# List detectors to calculate stats for
DETECTORS = {
    "U2Net": U2NetWrapper(weights_path=U2_WEIGHTS),
    "U2NetP": U2NetPWrapper(weights_path=U2P_WEIGHTS),
    "SAMNet": SAMNetWrapper(weights_path=SAM_WEIGHTS),
    "AIM": pys.AIM(location=PYSAL_ROOT),
    # "SUN": pys.SUN(location=PYSAL_ROOT),
    # "Finegrain": cv2.saliency.StaticSaliencyFineGrained.create(),
    # "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual.create(),
    # "BMS": BMSOptimized,
    # "IKN": IttiKoch,
}

# Compute segmentation stats for one image using the given detector
def compute_stats(detector, img, gt_mask):
    # Use computeSaliency for OpenCV & custom models
    if hasattr(detector, "computeSaliency"):
        success, sal_map = detector.computeSaliency(img)
    # Use saliency_map for pysal models
    else:
        sal_map = detector.saliency_map(img)
        success = sal_map is not None
    if not success:
        return None
    # Normalize saliency map to [0,1]
    sal = normalize_map(sal_map)
    # Binarize vals to create segmentation mask
    pred_mask = sal >= 0.5
    return calc_seg_stats(pred_mask, gt_mask)

def print_results(stats_list: List[ModelStats]):
    for ms in stats_list:
        print(f"{ms.model} (n={ms.n_images}):")
        print(f"  dice:      {ms.dice:.4f}")
        print(f"  iou:       {ms.iou:.4f}")
        print(f"  precision: {ms.precision:.4f}")
        print(f"  recall:    {ms.recall:.4f}")
        print(f"  f1-score:  {ms.f1_score:.4f}")
        print(f"  accuracy:  {ms.accuracy:.4f}")
        print(f"  time (s):  {ms.time:.3f}")
        print()

def results_to_csv(stats_list: List[ModelStats], output_dir: str, fname: str="model_stats.csv"):
    rows = [asdict(ms) for ms in stats_list]
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    df.to_csv(out_path, index=False)
    print(f"Wrote CSV to {out_path}")


def evaluate(cfg: ExperimentConfig):
    # 1) Load COCO dataset once
    print(">>> Gathering dataset…")
    dataset = gather_dataset(cfg.masks_json, cfg.input_dir)

    # 2) Prepare to record results
    model_metrics = {}
    stats_objs: list[ModelStats] = []
    random.seed(42) # Set seed for replicability in other experiments
    # Separate detectors into gpu/cpu for parallel processing loop
    gpu_detectors = ["U2Net", "U2NetP", "SAMNet"]
    cpu_detectors = [name for name in DETECTORS.keys() if name not in gpu_detectors]

    # 3) Aggregate stats for each detector
    for name, detector in DETECTORS.items():
        start_ts = time()

        # 3a) Pick sample size & worker count
        if name in cfg.slow_models:
            # Excessive copies of SUN & AIM overallocate mem
            max_workers = 1
            sample_data = random.sample(dataset, cfg.slow_model_n)
        else:
            # Throttle threads to leave some cores free
            max_workers = max(1, (os.cpu_count() or 1) - cfg.leave_free_cores)
            sample_data = random.sample(dataset, cfg.fast_model_n)
        
        stats_list = []

        # 4) Pool saliency computations for a given detector
        # Process CPU detectors in parallel
        if name in cpu_detectors:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {
                    exe.submit(compute_stats, detector, img, gt_mask): fn
                    for fn, img, gt_mask in sample_data
                }
                # Progress bar for total computations per detector
                pbar = tqdm(total=len(futures), desc=f"{name}", unit="img")
                for future in tqdm(as_completed(futures)):
                    # 5) Append results for each input image to list
                    result = future.result()
                    if result is not None:
                        stats_list.append(result)
                    pbar.update(1)
                pbar.close()
        # Process GPU comps sequentially
        elif name in gpu_detectors:
            pbar = tqdm(total=len(sample_data), desc=f"{name}", unit="img")
            for fn, img, gt_mask in sample_data:
                result = compute_stats(detector, img, gt_mask)
                if result is not None:
                    stats_list.append(result)
                pbar.update(1)
            pbar.close()
        else:
            print("Key Error: cannot process detector with key {name}.")

        # 6) Aggregate stats across validation set
        if stats_list:
            metrics = [m for m in stats_list[0].keys()]
            # Compute average for each metric
            summary = {m: np.mean([s[m] for s in stats_list]).astype(np.float64)
                       for m in metrics}
            
            end_ts = time()
            summary['time'] = np.float64(end_ts - start_ts)

            # Build ModelStats instance
            ms = ModelStats(
                model = name, 
                n_images = len(stats_list), 
                dice = summary['dice'],
                iou = summary['iou'],
                time  = summary['time'],
                precision = summary['precision'], 
                recall = summary['recall'],
                f1_score = summary['f1-score'],
                accuracy = summary['accuracy'],
            )
            # Append to final stats list
            stats_objs.append(ms)
        
        # 7) NEW: Collect per-image statistics for significance comparisons
        if stats_list:
            # Store per-image metrics
            model_metrics[name] = {
                'dice': [s['dice'] for s in stats_list],
                'iou': [s['iou'] for s in stats_list],
                'n': len(stats_list)
            }
    # 8) Print aggregated results
    print_results(stats_objs)
    # Output to csv if enabled
    if (cfg.csv_out):
        results_to_csv(stats_objs, cfg.output_dir, cfg.output_file)

    # 9) NEW: Verify that diff between model metrics is significant
    comparison_df = compare_models(model_metrics)
    print("\n=== Statistical Comparisons ===")
    print(comparison_df.round(4))

    # Save comparisons to CSV
    if cfg.csv_out:
        comp_path = os.path.join(cfg.output_dir, "model_comparisons.csv")
        comparison_df.to_csv(comp_path, index=False)

def main():
    cfg = ExperimentConfig(
        input_dir = "data/COCO/val2017", 
        output_dir = "app/eval/stats/results",
        output_file = "results.csv",
        masks_json = "data/COCO/annotations/instances_val2017.json", 
        slow_models = {"AIM", "SUN"},
        slow_model_n = 500,
        fast_model_n = 500, 
        leave_free_cores = 2,
        csv_out = True,
    )

    # Calculate aggregate metrics for each detector
    evaluate(cfg)

if __name__ == '__main__':
    main()