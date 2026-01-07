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




if __name__ == "__main__":
    AP = pd.read_csv('Output/AP.csv')
    KMeans = pd.read_csv('Output/KMeans.csv')
    IF = pd.read_csv('Output/IF.csv')

    AP_filenames = set(AP['Dataset'].tolist())
    KMeans_filenames = set(KMeans['Dataset'].tolist())
    IF_filenames = set(IF['Dataset'].tolist())

    common_filenames = AP_filenames.intersection(KMeans_filenames).intersection(IF_filenames)

    print(len(common_filenames), "common files found among all three methods.")

    AP_filtered = AP[AP['Dataset'].isin(common_filenames)]
    KMeans_filtered = KMeans[KMeans['Dataset'].isin(common_filenames)]
    IF_filtered = IF[IF['Dataset'].isin(common_filenames)]

    AP_streak_values = AP_filtered['Streak'].values
    KMeans_streak_values = KMeans_filtered['Streak'].values
    IF_streak_values = IF_filtered['Streak'].values

    AP_stats = {
        "1" : np.sum((AP_streak_values == 0)),
        "2-10" : np.sum((AP_streak_values >= 1) & (AP_streak_values <= 10)),
        "11-20" : np.sum((AP_streak_values >= 11) & (AP_streak_values <= 20)),
        "21-30" : np.sum((AP_streak_values >= 21) & (AP_streak_values <= 30)),
        "31-50" : np.sum((AP_streak_values >= 31) & (AP_streak_values <= 50)),
        "51+" : np.sum((AP_streak_values >= 51) & (AP_streak_values < 100)),
        "Deterministic" : np.sum(AP_streak_values == 100)
    }
    print("AP_stats:", AP_stats)

    KMeans_stats = {
        "1" : np.sum((KMeans_streak_values == 0)),
        "2-10" : np.sum((KMeans_streak_values >= 1) & (KMeans_streak_values <= 10)),
        "11-20" : np.sum((KMeans_streak_values >= 11)   & (KMeans_streak_values <= 20)),
        "21-30" : np.sum((KMeans_streak_values >= 21)   & (KMeans_streak_values <= 30)),
        "31-50" : np.sum((KMeans_streak_values >= 31)   & (KMeans_streak_values <= 50)),
        "51+" : np.sum((KMeans_streak_values >= 51) & (KMeans_streak_values < 100)),
        "Deterministic" : np.sum(KMeans_streak_values == 100)
    }
    print("KMeans_stats:", KMeans_stats)

    IF_stats = {
        "1" : np.sum((IF_streak_values == 0)),
        "2-10" : np.sum((IF_streak_values >= 1) & (IF_streak_values <= 10)),
        "11-20" : np.sum((IF_streak_values >= 11)   & (IF_streak_values <= 20)),
        "21-30" : np.sum((IF_streak_values >= 21)   & (IF_streak_values <= 30)),
        "31-50" : np.sum((IF_streak_values >= 31)   & (IF_streak_values <= 50)),
        "51+" : np.sum((IF_streak_values >= 51) & (IF_streak_values < 100)),
        "Deterministic" : np.sum(IF_streak_values == 100)
    }
    print("IF_stats:", IF_stats)

