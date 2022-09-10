import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def plot_histograms(df_AUs, cols2select, title="Regression AUs", ylabel="Average"):

    plt.subplots(figsize=(12, 8))

    data = df_AUs.melt('emotion', var_name='AUs', value_name='Mean')
    ax = sn.barplot(x="AUs", y="Mean", hue="emotion", data=data)
    # Make the plot
    plt.xlabel('Unidades de Acción', fontweight='bold', fontsize=15)
    plt.ylabel(ylabel, fontweight='bold', fontsize=15)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_covMatrixes(covMatrix_df, title =""):
    x_axis_labels = covMatrix_df.columns
    plt.figure(figsize=(12,8))
    sn.heatmap(covMatrix_df, annot=True, fmt='.1f', yticklabels=x_axis_labels)
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    path_annotations = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_mean.csv"
    df_AUs = pd.read_csv(path_annotations, sep=";", header=0)
    df_AUs['emotion']=['Neutra', 'Feliz', 'Triste', 'Sorpresa', 'Miedo', 'Asco', 'Ira']
    cols2select_r = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
                   "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

    plot_histograms(df_AUs, cols2select_r, title="Media de AU por emoción de AffectNet", ylabel="Media de AU")

    path_annotations = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_std.csv"
    df_AUs = pd.read_csv(path_annotations, sep=";", header=0)
    cols2select_r = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
                   "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

    plot_histograms(df_AUs, cols2select_r, title="Desviación típica de AU por emoción de AffectNet", ylabel="Desviación típica de AU")

    emotion = "Happy"
    path_covMatrixes = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_covM_"+emotion+".csv"
    df_covMatrix = pd.read_csv(path_covMatrixes, sep=";", header=0)
    #df_covMatrix=df_covMatrix.loc[0:(len(cols2select_r)-1),cols2select_r]
    plot_covMatrixes(df_covMatrix, title="Matriz de covarianza de emoción 'feliz' de AffectNet")

