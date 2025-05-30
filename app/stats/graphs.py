import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('input.csv')
def acc_vs_time(time, acc):
    x = time
    y = acc
    plt.figure(4,3)
    plt.scatter(x,y)
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy (%)')
