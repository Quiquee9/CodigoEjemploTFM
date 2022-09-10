import os
from typing import List

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
    df_auComp["video_name"]=list(df_Comp["video_name"])
    namesUniques=df_auComp["video_name"].unique()
    df_auComp["dist"] = None

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
            df_auComp.loc[i,"dist"]=dist
            #lista_distancias_emocion[emotionRAVDESS].append(dist)

    quartile=90
    """
    dfThreshTrain = pd.DataFrame(columns=list(df_auComp.columns))
    for idVideo in namesUniques:
        dfAUVid = df_auComp.loc[df_auComp.loc[:, 'video_name'] == idVideo]
        threshold = np.percentile(dfAUVid.loc[:, "dist"], quartile)
        dfThreshTrain=dfThreshTrain.append(dfAUVid.loc[dfAUVid.loc[:, "dist"] <= threshold])

    df_CompFiltrado = df_Comp.iloc[dfThreshTrain.index]
    """
    videoEmotionAU=[]

    dfThreshTest = pd.DataFrame(columns=list(df_auComp.columns))
    for idVideo in namesUniques:
        dfAUVid = df_auComp.loc[df_auComp.loc[:, 'video_name'] == idVideo]
        for emotionNumber, emotion in emotions_AffectNet.items():
            medias = df_AUsmeans.loc[emotionNumber, cols2select].values
            path_covMatrixes = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs/GMM_covM_" + emotion + ".csv"
            df_covMatrix = pd.read_csv(path_covMatrixes, sep=";", header=0)
            matrizCov = df_covMatrix.values
            IV = np.linalg.inv(matrizCov)
            for i in dfAUVid.index:
                dist = distance.mahalanobis(medias, dfAUVid.loc[i,cols2select], IV)
                dfAUVid.loc[i, "dist"+str(emotionNumber)] = dist
        distancias=[np.mean(dfAUVid["dist0"]),np.mean(dfAUVid["dist1"]),np.mean(dfAUVid["dist2"]),np.mean(dfAUVid["dist3"]),np.mean(dfAUVid["dist4"]),np.mean(dfAUVid["dist5"]),np.mean(dfAUVid["dist6"])]
        minimo = min(distancias)
        i_min = distancias.index(minimo)
        videoEmotionAU.append(i_min)
        threshold = np.percentile(dfAUVid.loc[:, "dist"+str(i_min)], quartile)
        dfThreshTest=dfThreshTest.append(dfAUVid.loc[dfAUVid.loc[:, "dist"+str(i_min)] <= threshold])

    df_CompFiltradoTest = df_Comp.iloc[dfThreshTest.index]
    """

    path = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/GMM_perVideo"
    df_CompFiltrado.to_csv(os.path.join(path, "labelsEmotion_GMM_AU_q_" + str(quartile) + ".csv"),
                            sep=";", header=True, index=False)
    print("De 1440 videos quedan ",len(df_CompFiltrado["video_name"].unique()))

    """

    path = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/GMM_perVideoTest"
    df_CompFiltradoTest.to_csv(os.path.join(path, "labelsEmotion_GMM_AU_q_" + str(quartile) + ".csv"),
                           sep=";", header=True, index=False)


