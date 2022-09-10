import os
import random
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

if __name__ == '__main__':
    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    #pathCSV = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsTrain8/Posteriorq25Fold0.csv"
    pathCSV ="/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/PosteriorTodoFold0.csv"
    pathCSV="/home/enriquediez/Guided-EMO-SpatialTransformer-main/extraccionRegq35Th07Test/fold0/df_embs_fc50.csv"
    #/home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modeloThresholdReg35/20220216_221409th07/trained_models
    emot_dict = {1:"Neutra",
                 2:"Calma",
                 3:"Feliz",
                 4:"Triste",
                 5:"Ira",
                 6:"Miedo",
                 7:"Asco",
                 8:"Sorpresa"}

    #Lee el CSV con todos los embeddings
    df_Posteriors = pd.read_csv(pathCSV, sep=";", header=0)
    """
        emot_dict = {0:"Neutral",
                 1:"Calm",
                 2:"happy",
                 3:"sad",
                 4:"angry",
                 5:"fear",
                 6:"disgust",
                 7:"surprise"}
    con el codigo de creador posteriors
    y_pred=df_Posteriors.iloc[:,7:15]

    emotion=df_Posteriors['emotion']
    """
    # Los ordena según idx, mismo que el csv de datasets_distribution
    df_Posteriors = df_Posteriors.sort_values('idx')
    df_Posteriors = df_Posteriors.reset_index(drop=True)
    # separo el nombre y el nombre con el frame
    splitted_video_name = df_Posteriors["name"].str.split("/")
    name, nameComp = map(list, zip(*splitted_video_name))
    emotion = []
    actor = []
    for i in name:
        emotion.append(int(i[6:8]))  # Estos dos numeros indican la emocion de RAVDES
        actor.append(int(i[18:20]))  # Estos dos numeros indican el actor de RAVDES

    y_pred=df_Posteriors.iloc[:,0:50]

    # t-SNE ANALYSIS
    #preprocessed_train = preprocessing.normalize(preprocessing.scale(y_pred))
    scaler = preprocessing.StandardScaler()
    preprocessed_train=scaler.fit_transform(y_pred)


    # Create tsne and fit with preprocessed data
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    tsn=1
    if(tsn):
        tsne = TSNE(n_components=2, init='pca', verbose=1, random_state=0)
        results = tsne.fit_transform(preprocessed_train)
    else:
        pca = PCA(n_components=2)
        pca.fit(preprocessed_train)
        results=pca.transform(preprocessed_train)

    # t-SNE components
    comp_x = results[:, 0]
    comp_y = results[:, 1]

    # Plot components with labels
    fig, ax = plt.subplots(figsize=(15, 10))
    for g in np.unique(emotion):
      ix = np.where(emotion == g)
      ax.scatter(comp_x[ix], comp_y[ix], label = emot_dict[g])
      ax.set(xlabel="Primera componente", ylabel="Segunda componente")
    ax.legend()
    plt.title("Visualización de T-SNE por emociones")
    plt.show()

    # Plot components with labels
    fig, ax = plt.subplots(figsize=(15, 10))
    for count, g in enumerate(np.unique(actor)):
      ix = np.where(actor == g)
      ax.scatter(comp_x[ix], comp_y[ix], label = g)
      ax.set(xlabel="Primera componente", ylabel="Segunda componente")
    ax.legend()
    plt.title("Visualización de T-SNE por actores")
    plt.show()
