from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd
import os


X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0], [5,5], [5,6], [6,5], [6,6]])

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# Run1 = AffinityPropagation(random_state=5).fit(X, noise_scale=0.1)

Diff_A_all = []
Diff_R_all = []

for s  in seeds:
    Diff_A = []
    Diff_R = []

    for noise in noises:
        
        Run1 = AffinityPropagation(random_state=s).fit(X, noise_scale=noise)
        Run1_A = pd.read_csv("A.csv", header=None)
        Run1_R = pd.read_csv("R.csv", header=None)
        iterations1 = Run1_A.shape[0]
        os.remove("A.csv")
        os.remove("R.csv")
        Run2 = AffinityPropagation(random_state=s+1).fit(X, noise_scale=noise)
        Run2_A = pd.read_csv("A.csv", header=None)
        Run2_R = pd.read_csv("R.csv", header=None)
        iterations2 = Run2_A.shape[0]
        os.remove("A.csv")
        os.remove("R.csv")

        iterations = min(iterations1, iterations2)
        sums_A = []
        sums_R = []
        for i in range(iterations):
            diag1 = Run1_A.iloc[i].values
            diag2 = Run2_A.iloc[i].values
            diff = np.abs(diag1 - diag2)
            sums_A.append(np.sum(diff))

            diag1 = Run1_R.iloc[i].values
            diag2 = Run2_R.iloc[i].values
            diff = np.abs(diag1 - diag2)
            sums_R.append(np.sum(diff))


        Diff_A.append(sums_A)
        Diff_R.append(sums_R)

    Diff_A_all.append(Diff_A)
    Diff_R_all.append(Diff_R)


Diff_A = np.mean(np.array(Diff_A_all), axis=0)
Diff_R = np.mean(np.array(Diff_R_all), axis=0)


# with open("Diff_A.csv", "w") as f:
#     for i in range(len(Diff_A[0])):
#         f.write(",".join([str(Diff_A[j][i]) for j in range(len(Diff_A))]) + "\n")

# with open("Diff_R.csv", "w") as f:
#     for i in range(len(Diff_R[0])):
#         f.write(",".join([str(Diff_R[j][i]) for j in range(len(Diff_R))]) + "\n")

import matplotlib.pyplot as plt

Diff_A = np.array(Diff_A)
Diff_R = np.array(Diff_R)

for i in range(11):
    iteration_i_a = Diff_A[:, i]
    iteration_i_r = Diff_R[:, i]

    plt.figure(figsize=(5, 5))

    plt.plot(noises, iteration_i_a, label=r"$\sum_{i=1}^{N} |A_{(i,i)}^{Run1} - A_{(i,i)}^{Run2}|$")
    plt.plot(noises, iteration_i_r, label=r"$\sum_{i=1}^{N} |R_{(i,i)}^{Run1} - R_{(i,i)}^{Run2}|$")
    plt.xlabel(r"Noise Level ($\epsilon$)", fontsize=20)
    plt.ylabel("Difference", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(f"Figures/Difference_Iter{i}.pdf", dpi=300, bbox_inches='tight')