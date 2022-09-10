import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    pathCSV = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/posteriorsRADVESS7/df_embs_posteriors.csv"
    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dict = {1:"Neutral",
                 2:"Calm",
                 3:"happy",
                 4:"sad",
                 5:"angry",
                 6:"fear",
                 7:"disgust",
                 8:"surprise"}
    #Lee el CSV con todos los posteriors
    df_Posteriors = pd.read_csv(pathCSV, sep=";", header=0)
    #Los ordena según idx, mismo que el csv de datasets_distribution
    df_Posteriors=df_Posteriors.sort_values('idx')
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
    for idVideo in namesUniques:

        dfVid = df_Posteriors.loc[df_Posteriors.loc[:, 'idVideo'] == idVideo]
        nameVideo = idVideo
        emotionVideo = emot_dict[dfVid["emotion"].unique()[0]]
        actorVideo = dfVid["actor"].unique()[0]
        plt.title(idVideo+' Emotion:'+emotionVideo)
        x = range(len(dfVid))
        plt.plot(x, dfVid["embs_0"], color='green', label='Neutral')
        plt.plot(x, dfVid["embs_1"], color='red', label='Happy')
        plt.plot(x, dfVid["embs_2"], color='skyblue', label='Sad')
        plt.plot(x, dfVid["embs_3"], color='black', label='Surprise')
        plt.plot(x, dfVid["embs_4"], color='purple', label='Fear')
        plt.plot(x, dfVid["embs_5"], color='pink', label='Disgust')
        plt.plot(x, dfVid["embs_6"], color='yellow', label='Anger')

        plt.legend()  # Mostrar leyenda

        plt.xlabel('Video Frames')
        plt.ylabel('Probability')

        os.makedirs(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/posteriorsPlots7/EMOTIONS", emotionVideo), exist_ok=True)
        os.makedirs(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/posteriorsPlots7/ACTOR", str(actorVideo)), exist_ok=True)

        plt.savefig(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/posteriorsPlots7/EMOTIONS",emotionVideo, idVideo+".png"))
        plt.savefig(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/posteriorsPlots7/ACTOR", str(actorVideo),idVideo+".png"))

        plt.close('all')



