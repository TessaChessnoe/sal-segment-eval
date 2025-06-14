import pandas as pd
import matplotlib.pyplot as plt
from app.eval.stats.seg_stats import ModelStats
import itertools
import numpy as np
import os

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
    # Fit long models by disp. diagonally
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def radar_plot(stats_list, selected_models=None):
    # Choose which metrics to show
    metrics = ["precision", "recall", "f1_score", "iou", "dice", "accuracy"]
    n_metrics = len(metrics)

    # Angles for radar chart axes
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # to close the loop

    # Set up plot
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    # Optional: filter which models to plot
    if selected_models:
        stats_list = [ms for ms in stats_list if ms.model in selected_models]

    data = []
    # Plot each model & get data means
    for ms in stats_list:
        values = [getattr(ms, m) for m in metrics]
        values += values[:1]  # loop to starting angle
        mean_score = np.mean(values)

        ax.plot(angles, values, label=ms.model)
        ax.fill(angles, values, alpha=0.1)

        data.append({
            "Model": ms.model,
            **{m.capitalize(): getattr(ms, m) for m in metrics},
            "Mean Score": mean_score
        })

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_ylim(0, 1.0)

    plt.title("Model Comparison Across Segmentation Metrics", size=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
    plt.tight_layout()
    plt.show()

    # Export mean score data to df
    df = pd.DataFrame(data)
    sorted_df = df.sort_values(by="Mean Score", ascending=False)
    return sorted_df

def res_thru_bars(df):
    models = df['Model']
    mp_per_s = df['MP/s']

    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, mp_per_s)
    
    # Add value labels above bars
    for bar, score in zip(bars, mp_per_s):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f"{score:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.title("Methods Ranked by MP/s")
    plt.xlabel('Method')
    plt.ylabel('MP/s (Megapixels/second)')
    # Fit long models by disp. diagonally
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def quality_per_time(stats_list):
    models = [ms.model for ms in stats_list]
    q_per_time = [(ms.iou + ms.dice) / (ms.time/ms.n_images)
                        for ms in stats_list]
    results = pd.DataFrame(list(zip(models, q_per_time)), columns=['Models', 'Quality/Time'])
    plt.figure(figsize=(9, 5))
    bars = plt.bar(models, q_per_time)
    
    # Add value labels above bars
    for bar, score in zip(bars, q_per_time):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 f"{score:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.title("Segmentation Quality per Second of Inference")
    plt.xlabel('Method')
    plt.ylabel('Composite Score (IoU + Dice / fps)')
    # Fit long models by disp. diagonally
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return results

def flop_efficiency(stats_list):
    return

# 1) Pareto graph of accuracy vs. time/image
acc_time_pareto(stats_list)
# 2) Measure segmentation performance
dice_iou(stats_list)

# 3) Radar plot of all model metrics
# Prepare to read in radar plot data
radar_output = "app/eval/graphs/csv/radar_means.csv"
os.makedirs("results", exist_ok=True)

# Group 1: Deep models
df1 = radar_plot(stats_list, selected_models=["SAM-Net", "U^2-Net", "U^2-Net-Small"])
# Group 2: Classical
df2 = radar_plot(stats_list, selected_models=["BMS", "IttiKoch"])
# Group 3: OpenCV
df3 = radar_plot(stats_list, selected_models=["FineGrained", "SpectralRes"])
# Group 4: Outliers or slow models
df4 = radar_plot(stats_list, selected_models=["SUN", "AIM"])
radar_means = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
# Sort by highest mean score 1st
s_radar_means = radar_means.sort_values(by="Mean Score", ascending=False)
radar_means.to_csv(radar_output, index=False)

# 4) Test raw pixel throughput 
res_thru_df = pd.read_csv("app/eval/graphs/results/res_thru.csv")
res_thru_bars(res_thru_df)

# 5) Eval quality per second of inference
mp_res_output = "app/eval/graphs/csv/quality_time.csv"
os.makedirs("results", exist_ok=True)

results = quality_per_time(stats_list)
results.to_csv(mp_res_output, index=False)
