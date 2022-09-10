import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter
from os import listdir
from os.path import isfile, join
from scipy.spatial import distance
if __name__ == '__main__':

    pathCSV_comp = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution" \
                   "/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv"

    aufilesRAVDESS= "/mnt/RESOURCES/enriquediez/RAVDESS/AUs_frames_720x720_processed/"

    path_means = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_mean.csv"


    df_Comp = pd.read_csv(pathCSV_comp, sep=";", header=0)
    df_AUsmeans = pd.read_csv(path_means, sep=";", header=0)


    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dictRAVD = {1: "Neutral",
                     2:"Calm",
                     3:"happy",
                     4:"sad",
                     5:"angry",
                     6:"fear",
                     7:"disgust",
                     8:"surprise"}

    emotions_AffectNet = {
        0:"Neutral", #75374
        1:"Happy", #134914
        2:"Sad", #25958
        3:"Surprise", # 14590
        4:"Fear", #6878
        5:"Disgust", #4303
        6:"Anger", #25382
    }


    listcsvAU_RAVD = [f for f in listdir(aufilesRAVDESS) if isfile(join(aufilesRAVDESS, f))]
    listcsvAU_RAVD.sort()

    listAUsALL = []
    cols2select = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
                   "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
    df_auComp = pd.DataFrame([], columns=cols2select+["emotion"])

    filtInd=[]
    for csvfile in listcsvAU_RAVD:
        df_au = pd.read_csv(aufilesRAVDESS + csvfile, sep=";", header=0)

        emotion = csvfile[6:8]
        df_au['emotion'] = int(emotion)
        df_auComp=df_auComp.append(df_au[cols2select+["emotion"]])

    df_auComp = df_auComp.reset_index(drop=True)
    #La dos de calma la junto con la de 1.
    lista_distancias_emocion = {1: [],
                          3: [],
                          4: [],
                          5: [],
                          6: [],
                          7: [],
                          8: []}
    threshold_distancias_emocion = {1: 0,
                          3: 0,
                          4: 0,
                          5: 0,
                          6: 0,
                          7: 0,
                          8: 0}

    emot_dictRAVD2 = {1: "Neutra",
                     2:"Calma",
                     3:"Feliz",
                     4:"Triste",
                     5:"Ira",
                     6:"Miedo",
                     7:"Asco",
                     8:"Sorpresa"}

    emotions_AffectNet2 = {
        0:"Neutra", #75374
        1:"Feliz", #134914
        2:"Triste", #25958
        3:"Sorpresa", # 14590
        4:"Miedo", #6878
        5:"Asco", #4303
        6:"Ira", #25382
    }
    for emotionNumber,emotion in emotions_AffectNet.items():

        if emotion=="Neutral":
            emotionRAVDESS = 1
        elif emotion=="Happy":
            emotionRAVDESS = 3
        elif emotion=="Sad":
            emotionRAVDESS = 4
        elif emotion=="Surprise":
            emotionRAVDESS = 8
        elif emotion=="Fear":
            emotionRAVDESS = 6
        elif emotion=="Disgust":
            emotionRAVDESS = 7
        elif emotion=="Anger":
            emotionRAVDESS = 5

        medias=df_AUsmeans.loc[emotionNumber, cols2select].values
        path_covMatrixes = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_covM_" + emotion + ".csv"
        df_covMatrix = pd.read_csv(path_covMatrixes, sep=";", header=0)
        matrizCov=df_covMatrix.values
        if(emotion=="Neutral"):
            df_aux = df_auComp.loc[(df_auComp["emotion"] == 1) | (df_auComp["emotion"] == 2), cols2select]
        else:
            df_aux=df_auComp.loc[df_auComp["emotion"]==emotionRAVDESS,cols2select]
        print(emotion)
        IV=np.linalg.inv(matrizCov)
        for i in df_aux.index:
            dist=distance.mahalanobis(medias, df_aux.loc[i], IV)
            lista_distancias_emocion[emotionRAVDESS].append(dist)

    quartile=60
    for number,listaDist in lista_distancias_emocion.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(listaDist, bins=100)
        ax.set(xlabel="Distancia", ylabel="Número de fotogramas")
        plt.title("Histograma de distancia de emoción '"+emot_dictRAVD2[number]+"'")
        threshold = np.percentile(listaDist, quartile)
        threshold_distancias_emocion[number]=threshold
        plt.axvline(x=threshold,color="red")
        plt.show()

    filtInd=[]
    for emotionNumber,emotion in emotions_AffectNet.items():

        if emotion=="Neutral":
            emotionRAVDESS = 1
        elif emotion=="Happy":
            emotionRAVDESS = 3
        elif emotion=="Sad":
            emotionRAVDESS = 4
        elif emotion=="Surprise":
            emotionRAVDESS = 8
        elif emotion=="Fear":
            emotionRAVDESS = 6
        elif emotion=="Disgust":
            emotionRAVDESS = 7
        elif emotion=="Anger":
            emotionRAVDESS = 5

        medias=df_AUsmeans.loc[emotionNumber, cols2select].values
        path_covMatrixes = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_covM_" + emotion + ".csv"
        df_covMatrix = pd.read_csv(path_covMatrixes, sep=";", header=0)
        matrizCov=df_covMatrix.values
        if(emotion=="Neutral"):
            df_aux = df_auComp.loc[(df_auComp["emotion"] == 1) | (df_auComp["emotion"] == 2), cols2select]
        else:
            df_aux=df_auComp.loc[df_auComp["emotion"]==emotionRAVDESS,cols2select]
        print(emotion)
        IV=np.linalg.inv(matrizCov)
        for i in df_aux.index:
            dist=distance.mahalanobis(medias, df_aux.loc[i], IV)
            if(dist<=threshold_distancias_emocion[emotionRAVDESS]):
                filtInd.append(i)


    filtInd.sort()
    df_CompFiltrado = df_Comp.iloc[filtInd]
    path = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/GMM"
    #df_CompFiltrado.to_csv(os.path.join(path, "labelsEmotion_GMM_AU_q_" + str(quartile) + ".csv"),
    #                       sep=";", header=True, index=False)
    print("De 1440 videos quedan ",len(df_CompFiltrado["video_name"].unique()))
    #set(df_Comp["video_name"].unique()).difference(set(df_CompFiltrado["video_name"].unique()))

    fig, ax = plt.subplots()
    ax.plot(range(1,96),lista_distancias_emocion[1][0:95])
    ax.set(xlabel="Fotogramas", ylabel="Distancia")
    plt.title("Distancia de Mahalanobis para cada fotograma del vídeo 01-01-01-01-01-01-01")
    plt.show()
