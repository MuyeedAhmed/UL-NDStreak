from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

def RunAndSaveDiff(noises):
    X = np.array([[1, 2], [1, 4], [1, 0],
                [4, 2], [4, 4], [4, 0], [5,5], [5,6], [6,5], [6,6]])

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    

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

    data = {
        "Diff_A": Diff_A,
        "Diff_R": Diff_R
    }
    with open("Data/Diff_Data.pkl", "wb") as f:
        pickle.dump(data, f)
    

def plot_differences(Diff_A, Diff_R, noises):
    Diff_A = np.array(Diff_A)
    Diff_R = np.array(Diff_R)

    for i in range(11):
        iteration_i_a = Diff_A[:, i]
        iteration_i_r = Diff_R[:, i]

        plt.figure(figsize=(7, 3))

        # plt.plot(noises, iteration_i_a, label=r"$\sum_{i=1}^{N} |A_{(i,i)}^{Run1} - A_{(i,i)}^{Run2}|$")
        # plt.plot(noises, iteration_i_r, label=r"$\sum_{i=1}^{N} |R_{(i,i)}^{Run1} - R_{(i,i)}^{Run2}|$")
        # plt.plot(noises, iteration_i_a, label="Availability (A)")
        # plt.plot(noises, iteration_i_r, label="Responsibility (R)")
        plt.plot(noises, iteration_i_a, label=r"$\Delta_A$",lw=4)
        plt.plot(noises, iteration_i_r, label=r"$\Delta_R$",lw=4)
        plt.xlabel(r"Noise Level ($\epsilon$)", fontsize=28)
        plt.ylabel("Difference", fontsize=28)
        plt.xticks(fontsize=22)
        if i == 2:
            plt.yticks([0e-14, 1e-14, 2e-14, 3e-14], fontsize=22)
        else:
            plt.yticks(fontsize=22)
        plt.legend(fontsize=28)
        ax = plt.gca()
        ax.yaxis.get_offset_text().set_fontsize(18)
        plt.savefig(f"Figures/Difference_Iter{i}.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if not os.path.exists("Data"):
        os.makedirs("Data")
    if not os.path.exists("Data/Diff_Data.pkl"):
        RunAndSaveDiff(noises)
    with open("Data/Diff_Data.pkl", "rb") as f:
        data = pickle.load(f)
    Diff_A = data["Diff_A"]
    Diff_R = data["Diff_R"]
    plot_differences(Diff_A, Diff_R, noises)