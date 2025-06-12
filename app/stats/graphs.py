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

def acc_vs_time(stats_list):
    # Set up the figure
    plt.figure(figsize=(6, 4))
    
    # Use a color cycle to assign colors automatically
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for ms, color in zip(stats_list, color_cycle):
        plt.scatter(ms.time, ms.accuracy, color=color, label=ms.model)

    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Inference Time')
    plt.legend(title='Model')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

times = [ms.time for ms in stats_list]
accuracies = [ms.accuracy for ms in stats_list]
names = [ms.model for ms in stats_list]

acc_vs_time(stats_list)