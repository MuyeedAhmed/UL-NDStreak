import pandas as pd
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AffinityPropagation
import time
import csv




def plot_abs_time(df, output_path):
    plt.figure(figsize=(6, 4))

    
    plt.plot(df['n_samples'], df["A2"], marker='o', color="green", label="Discrete")
    # plt.plot(df['n_samples'], df["A5"], marker='s', label=r"$\{-\epsilon, -\frac{\epsilon}{2}, 0, \frac{\epsilon}{2}, +\epsilon\}$")
    plt.plot(df['n_samples'], df["All"], marker='^', color="red", label="Continuous")

    plt.xlabel('n_samples')
    plt.ylabel('Time (seconds)')

    plt.legend()
    # plt.grid(True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    input_csv = 'Abstraction.csv'
    output_image = 'AbstractionTime.pdf'

    df = pd.read_csv(input_csv)
    plot_abs_time(df, output_image)