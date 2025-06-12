import pandas as pd
import matplotlib.pyplot as plt
from app.stats.seg_stats import ModelStats
import itertools

stats_list = [
    ModelStats("U^2-Net-Small", 2000, dice=0.4407, iou=0.3417,
               precision=0.7563, recall=0.3878, f1_score=0.4407, 
               accuracy=0.7722, time=118.806),
    ModelStats("U^2-Net", 2000, dice=0.4984, iou=0.3938,
               precision=0.7420, recall=0.4600, f1_score=0.4984,
               accuracy=0.7864, time=154.990),
    ModelStats("SAM-Net", 2000, dice=0.5710, iou=0.4519, 
               precision=0.6896, recall=0.5948, f1_score=0.5710,
               accuracy=0.7944, time=226.317),
    # ModelStats("SUN", 400, dice=0.1383, iou=0.0811, 
    #            precision=0.4476, recall=0.1221, f1_score=0.1383,
    #            accuracy=0.6804, time=3156.894),
    # ModelStats("AIM", 400, dice=0.4602, iou=0.3374, 
    #            precision=0.3702, recall=0.8935, f1_score=0.4602,
    #            accuracy=0.5124, time=9736.886),
    ModelStats("BMS", 2000, dice=0.2033, iou=0.1297, 
               precision=0.3372, recall=0.2140, f1_score=0.2033,
               accuracy=0.6074, time=78.636),
    ModelStats("IttiKoch", 2000, dice=0.1152, iou=0.0643, 
               precision=0.5062, recall=0.0800, f1_score=0.1152,
               accuracy=0.7051, time=48.352),
    ModelStats("SpectralRes", 2000, dice=0.1158, iou=0.0667, 
               precision=0.4903, recall=0.0836, f1_score=0.1158,
               accuracy=0.7062, time=7.853),
    ModelStats("FineGrained", 2000, dice=0.1531, iou=0.0880, 
               precision=0.4597, recall=0.1162, f1_score=0.1531,
               accuracy=0.7009, time=16.735),
]

def acc_time_pareto(stats_list):
    # 1) Get x-vals, y-vals, and model labels
    times = [round(ms.time / ms.n_images, 4) for ms in stats_list] # Get FPS from total time
    accs = [ms.accuracy for ms in stats_list]
    names = [ms.model for ms in stats_list]

    # 2) Glue values to model name & sort points (ASC)
    points = list(zip(times, accs, names))
    points.sort()

    pareto = []
    max_acc = -1
    for t, acc, name in points:
        if acc > max_acc:
            pareto.append((t, acc, name))
            max_acc = acc

    # Separate for plotting
    ptimes = [t for t, a, n in pareto]
    paccs = [a for t, a, n in pareto]
    pnames = [n for t, a, n in pareto]

    plt.figure(figsize=(7,5))
    plt.scatter(times, accs, label="All models", alpha=0.5)
    plt.plot(ptimes, paccs, color="red", label="Pareto frontier", linewidth=2, marker="o")

    # Optional: annotate graph with model names
    for t, a, n in zip(times, accs, names):
        # 1. Label all points (default = non-bold, grayish)
        if n not in pnames:
            plt.text(t * 1.01, a, n, fontsize=8, alpha=0.6, color='gray')
        # 2. Highlight Pareto frontier pts in bold
        else:
            plt.text(t * 1.01, a, n, fontsize=8, weight='bold', color='black')

    plt.xlabel("Time/image (fps)")
    plt.ylabel("Accuracy")
    plt.title("Pareto Frontier: Accuracy vs. Inference Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def dice_iou(stats_list):
    # Sort by (iou + dice) descending
    sorted_stats = sorted(stats_list, key=lambda ms: ms.iou + ms.dice, reverse=True)
    models = [ms.model for ms in sorted_stats]
    composite_scores = [ms.iou + ms.dice for ms in sorted_stats]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, composite_scores)
    
    # Add value labels above bars
    for bar, score in zip(bars, composite_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f"{score:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.title("Methods Ranked by IOU + DICE")
    plt.xlabel('Method')
    plt.ylabel('Composite Score (IoU + Dice)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# acc_time_pareto(stats_list)
dice_iou(stats_list)