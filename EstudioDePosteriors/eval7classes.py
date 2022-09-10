import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter

if __name__ == '__main__':
    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    pathCSV = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsRADVESS7/df_embs_posteriors.csv"
    plt.rcParams['figure.figsize'] = (10, 5)
    #RAVDESS
    emot_dict = {1:"Neutral",
                 2:"Calm",
                 3:"Happy",
                 4:"Sad",
                 5:"Angry",
                 6:"Fear",
                 7:"Disgust",
                 8:"Surprise"}

    #STN 7 clases classes = ('Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger')

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

    #Cambio los neutrales 1 por 2 (calmados), al restar 2 se corresponde a las prediciones de embs
    df_Posteriors["emotion"] = df_Posteriors["emotion"].replace(1, 2)
    df_Posteriors["emotion"]=df_Posteriors["emotion"]-2
    #Cambio el posterior de angry con el surprise
    embSurprise = list(df_Posteriors["embs_3"])
    embAngry = list(df_Posteriors["embs_6"])
    df_Posteriors["embs_3"] = embAngry
    df_Posteriors["embs_6"] = embSurprise

    y_true = list(df_Posteriors["emotion"]) #de 0 a 6 cada frame

    #Da la columna maxima de embs
    y_predText = df_Posteriors[["embs_0", "embs_1", "embs_2", "embs_3", "embs_4", "embs_5", "embs_6"]].idxmax(axis=1)
    y_pred=[]
    #Pasar de texto a numero
    for pred in y_predText:
        y_pred.append(int(pred[5]))
    target_names = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print(accuracy_score(y_true, y_pred))

    y_trueVid=[]
    y_predVid=[]
    AUC_vot=[]

    for idVideo in namesUniques:
        dfVid = df_Posteriors.loc[df_Posteriors.loc[:, 'idVideo'] == idVideo]
        nameVideo = idVideo
        y_trueVid.append(dfVid["emotion"].unique()[0])
        y_predText = dfVid[["embs_0", "embs_1", "embs_2", "embs_3", "embs_4", "embs_5", "embs_6"]].idxmax(
            axis=1)
        aux=[]
        for pred in y_predText:
            aux.append(int(pred[5]))
        counter = Counter(aux)
        y_predVid.append(counter.most_common()[0][0])

        AUC_list = []
        AUC_list.append(sum(dfVid["embs_0"]))
        AUC_list.append(sum(dfVid["embs_1"]))
        AUC_list.append(sum(dfVid["embs_2"]))
        AUC_list.append(sum(dfVid["embs_3"]))
        AUC_list.append(sum(dfVid["embs_4"]))
        AUC_list.append(sum(dfVid["embs_5"]))
        AUC_list.append(sum(dfVid["embs_6"]))
        AUC_vot.append(AUC_list.index(max(AUC_list)))

    print(classification_report(y_trueVid, y_predVid, target_names=target_names, digits=4))
    print(accuracy_score(y_trueVid, y_predVid))
    print(classification_report(y_trueVid, AUC_vot, target_names=target_names, digits=4))
    print(accuracy_score(y_trueVid, AUC_vot))

