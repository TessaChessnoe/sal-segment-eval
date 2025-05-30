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
from app.config.exp_config import ExperimentConfig
from app.stats.stat_helpers import (
    gather_dataset,
    normalize_map,
    calc_seg_stats)

# Using dataclass avoids storing stats in nested dicts
@dataclass
class ModelStats:
    model: str
    n_images: int
    precision: float
    recall: float
    f1_score: float
    iou: float
    dice: float
    accuracy: float
    time: float

# Download built-in models into your model_loc folder
model_root = "app/models/pysal"
# List detectors to calculate stats for
DETECTORS = {
    # "AIM": pys.AIM(location=model_root),
    # "SUN": pys.SUN(location=model_root),
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
        print(f"  precision: {ms.precision:.4f}")
        print(f"  recall:    {ms.recall:.4f}")
        print(f"  f1-score:  {ms.f1_score:.4f}")
        print(f"  iou:       {ms.iou:.4f}")
        print(f"  dice:      {ms.dice:.4f}")
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
    print(f">>> Dataset size: {len(dataset)} samples\n")

    # 2) Prepare to record results
    stats_objs: list[ModelStats] = []
    i = 0 # Tracks which model we're on
    random.seed(42) # Set seed for replicability in other experiments

    # 3) Aggregate stats for each detector
    for name, detector in DETECTORS.items():
        i += 1
        stats_list = [] # Collect stats for each image's computation
        start_ts = time()

        # Run slow models with less samples
        if name in cfg.slow_models:
            # Excessive copies of SUN & AIM overallocate mem
            max_workers = 2
            sample_data = random.sample(dataset, cfg.slow_model_n)
        else:
            # Throttle threads to leave some cores free
            max_workers = max(1, (os.cpu_count() or 1) - cfg.leave_free_cores)
            sample_data = random.sample(dataset, cfg.fast_model_n)
        
        # 4) Pool saliency computations for a given detector
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_stats, detector, img, gt_mask): fn
                for fn, img, gt_mask in sample_data
            }
            # Per-model progress bar for remaining computations
            for future in tqdm(as_completed(futures),
                                total=len(futures),
                                desc=f"[{i}/{len(DETECTORS)}] {name}",
                                unit="img"):
                # 5) Append results for each input image to list
                img_stats = future.result()
                if img_stats is not None:
                    stats_list.append(img_stats)

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
                precision = summary['precision'], 
                recall = summary['recall'],
                f1_score = summary['f1-score'],
                iou = summary['iou'],
                dice = summary['dice'],
                accuracy = summary['accuracy'],
                time  = summary['time']
            )
            # Append to final stats list
            stats_objs.append(ms)
    # 7) Print aggregated results
    print_results(stats_objs)
    # Output to csv if enabled
    if (cfg.csv_out):
        results_to_csv(stats_objs, cfg.output_dir, cfg.output_file)

def main():
    cfg = ExperimentConfig(
        input_dir = "data/COCO/val2017", 
        output_dir = "app/stats/results",
        output_file = "results.csv",
        masks_json = "data/COCO/annotations/instances_val2017.json", 
        slow_models = {"AIM", "SUN"},
        slow_model_n = 200,
        fast_model_n = 2000, 
        leave_free_cores = 2,
        csv_out = False,
    )

    # Calculate aggregate metrics for each detector
    evaluate(cfg) # write result to csv

if __name__ == '__main__':
    main()