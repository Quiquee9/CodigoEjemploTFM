import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter

if __name__ == '__main__':
    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    pathCSV = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/posteriorsRADVESS3/df_embs_posteriors.csv"
    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dict = {1:"Neutral",
                 2:"Calm",
                 3:"happy",
                 4:"sad",
                 5:"angry",
                 6:"fear",
                 7:"disgust",
                 8:"surprise"}
    emot_to_valence = {1: 0,
                       2: 0,
                       3: 1,
                       4: 2,
                       5: 2,
                       6: 2,
                       7: 2,
                       8: 2}
    valence_to_text = {0: "Neutral",
                       1: "Positive",
                       2: "Negative"}
    #Lee el CSV con todos los posteriors
    df_Posteriors = pd.read_csv(pathCSV, sep=";", header=0)
    #Los ordena según idx, mismo que el csv de datasets_distribution
    df_Posteriors = df_Posteriors.sort_values('idx')
    df_Posteriors = df_Posteriors.reset_index(drop=True)
    #separo el nombre y el nombre con el frame
    splitted_video_name = df_Posteriors["name"].str.split("/")
    name, nameComp = map(list, zip(*splitted_video_name))
    emotion=[]
    actor=[]
    for i in name:
        emotion.append(int(i[6:8])) #Estos dos numeros indican la emocion de RAVDES
        actor.append(int(i[18:20]))#Estos dos numeros indican el actor de RAVDES

    df_Posteriors["emotion"]=emotion
    df_Posteriors["actor"]=actor
    df_Posteriors["idVideo"]=name
    namesUniques=df_Posteriors["idVideo"].unique()

    df_Posteriors=df_Posteriors.replace({"emotion": emot_to_valence})

    y_true = list(df_Posteriors["emotion"]) #de 0 a 2 cada frame

    #Da la columna maxima de embs
    y_predText = df_Posteriors[["embs_0", "embs_1", "embs_2"]].idxmax(axis=1)
    y_pred=[]
    #Pasar de texto a numero
    for pred in y_predText:
        y_pred.append(int(pred[5]))
    target_names = ['Neutral', 'Positive', 'Negative']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print(accuracy_score(y_true, y_pred))

    y_trueVid=[]
    y_predVid=[]
    for idVideo in namesUniques:
        dfVid = df_Posteriors.loc[df_Posteriors.loc[:, 'idVideo'] == idVideo]
        nameVideo = idVideo
        y_trueVid.append(dfVid["emotion"].unique()[0])
        y_predText = dfVid[["embs_0", "embs_1", "embs_2"]].idxmax(
            axis=1)
        aux=[]
        for pred in y_predText:
            aux.append(int(pred[5]))
        counter = Counter(aux)
        y_predVid.append(counter.most_common()[0][0])
    print(classification_report(y_trueVid, y_predVid, target_names=target_names, digits=4))
    print(accuracy_score(y_trueVid, y_predVid))


