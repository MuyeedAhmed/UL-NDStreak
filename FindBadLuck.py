


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

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_and_prepare_data(file_path, target_column):
    df = pd.read_csv(file_path)
    unique_values = df[target_column].nunique()
    data = df.drop(columns=[target_column])
    return data, unique_values

def process_datasets(algo, input_folder, target_column, restarts = 100):
    if os.path.exists(f"{algo}.csv"):
        done_files = set()
        with open(f"{algo}.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                done_files.add(row[0])
    else:
        done_files = set()
        with open(f"{algo}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset'] + [f'Restart_{i+1}' for i in range(restarts)] + ['Streak'])

    for file_name in os.listdir(input_folder):
        print(file_name)
        file_path = os.path.join(input_folder, file_name)

        if not file_name.endswith('.csv'):
            continue
        if file_name in done_files:
            continue

        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb > 100:
            continue

        max_streak = 0
        streak = 0
        try:
            data, unique_values = load_and_prepare_data(file_path, target_column)
            aris = []
            
            t0 = time.time()
            if algo == "AP":
                clustering = AffinityPropagation().fit(data)            
                l1 = clustering.labels_
            elif algo == "KMeans":
                kmeans = KMeans(n_clusters=unique_values).fit(data)
                l1 = kmeans.labels_
            elif algo == "IF":
                iso_forest = IsolationForest().fit(data)
                l1 = iso_forest.predict(data)

            print("Time", time.time()-t0)
            if time.time()-t0 > 1:
                print("Skipping due to time limit")
                continue
            for i in range(restarts):
                if algo == "AP":
                    clustering = AffinityPropagation().fit(data)
                    l2 = clustering.labels_
                elif algo == "KMeans":
                    kmeans = KMeans(n_clusters=unique_values).fit(data)
                    l2 = kmeans.labels_
                elif algo == "IF":
                    iso_forest = IsolationForest().fit(data)
                    l2 = iso_forest.predict(data)
                    
                ari = adjusted_rand_score(l1, l2)
                if ari == 1:
                    streak += 1
                    if streak >= max_streak:
                        max_streak = streak
                else:
                    if streak >= max_streak:
                        max_streak = streak
                    streak = 0
                aris.append(ari)
                l1 = l2
            # print(file_name, aris)
            with open(f"{algo}.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([file_name] + aris + [max_streak])
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def main(algo):
    input_folder = '../Dataset/'
    target_column = 'target'

    process_datasets(algo, input_folder, target_column)

if __name__ == '__main__':
    algo = sys.argv[1]
    main(algo)
