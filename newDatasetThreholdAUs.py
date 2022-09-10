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
if __name__ == '__main__':

    pathCSV_comp = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution" \
                   "/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv"
    aufiles="/mnt/RESOURCES/enriquediez/RAVDESS/AUs_frames_720x720_processed/"
    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dict = {1:"Neutral",
                 2:"Calm",
                 3:"happy",
                 4:"sad",
                 5:"angry",
                 6:"fear",
                 7:"disgust",
                 8:"surprise"}

    df_Comp = pd.read_csv(pathCSV_comp, sep=";", header=0)
    listcsv = [f for f in listdir(aufiles) if isfile(join(aufiles, f))]
    listcsv.sort()
    listNumberAus=[]
    listNumberFrames=[]
    filtInd=[]
    listAUsALL=[]
    th=0.75
    for csvfile in listcsv:
        df_au = pd.read_csv(aufiles+csvfile, sep=";", header=0)
        video_name=csvfile[:-4]
        listAUsALL.extend(df_au['AU45_r'])
        df_vid=df_Comp.loc[df_Comp.loc[:,'video_name']==video_name]
        listNumberAus.append(len(df_au))
        listNumberFrames.append(len(df_vid))

        df_vid['AU45_r']=list(df_au['AU45_r'])

        var=df_vid.loc[df_vid.loc[:, "AU45_r"] <= th]
        filtInd.extend(list(var.index))

    df_CompFiltrado = df_Comp.iloc[filtInd]
    path = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AUth"
    df_CompFiltrado.to_csv(os.path.join(path, "labelsEmotion_AUth_" + str(th) + ".csv"),
                            sep=";", header=True, index=False)

    df_describe = pd.DataFrame(listAUsALL)
    print(df_describe.describe())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(listAUsALL,bins=75)
    ax.set(xlabel="Valor de AU45", ylabel="Número de Frames")
    plt.title("Histograma de todos los AU45")
    plt.show()


    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(listAUsALL,bins=75)
    ax.set(xlabel="Valor de AU45", ylabel="Número de Frames")
    ax.set(ylim=[0,10000])
    plt.title("Histograma de todos los AU45 [recortado eje y a 10000]")
    plt.show()

    #AUs_frames=np.subtract(listNumberAus,listNumberFrames)
    #fig, ax = plt.subplots(figsize=(7, 5))
    #a=sns.histplot(AUs_frames,ax=ax)
    #ax.set(xlabel="Diferencia entre Frames y AUs por vídeo", ylabel="Número de vídeos")
    #plt.show()
    #df_describe = pd.DataFrame(AUs_frames)
    #print(df_describe.describe())
    #print(df_describe.value_counts())

    print("Th: ",th,"% datos: ",len(filtInd)/154108*100)
