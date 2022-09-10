import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

if __name__ == '__main__':

    pathCSV ="/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv"
    emot_dict = {0:"Neutra",
                 1:"Calma",
                 2:"Feliz",
                 3:"Triste",
                 4:"Ira",
                 5:"Miedo",
                 6:"Asco",
                 7:"Sorpresa"}
    #Lee el CSV con todos los posteriors
    df_Frames = pd.read_csv(pathCSV, sep=";", header=0)
    a=df_Frames["emotion"].copy()
    a.replace(emot_dict, inplace=True)
    df_Frames["emotion_label"]=a

    print("Número total de frames:",len(df_Frames))


    sns.countplot(df_Frames['emotion_label'])
    plt.title("Número de frames por emoción")
    plt.xlabel("Emociones")
    plt.ylabel("Número de frames")
    plt.show()

    framesPorEmocion=df_Frames["emotion_label"].value_counts()
    print("Frames por emoción: ")
    print(framesPorEmocion)
    fig1, ax1 = plt.subplots()
    ax1.pie(framesPorEmocion.values, labels=framesPorEmocion.index, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title("Gráfico circular de frames por emoción")
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

    df_videos = pd.DataFrame(columns=list(df_Frames.columns))  # mismas columnas que posteriors
    namesUniques = df_Frames["video_name"].unique()
    for idVideo in namesUniques:
        dfVid = df_Frames.loc[df_Frames.loc[:, 'video_name'] == idVideo]
        nameVideo = idVideo
        emotionVideo = emot_dict[dfVid["emotion"].unique()[0]]
        actorVideo = dfVid["actor"].unique()[0]
        maxFrame = max(dfVid["frame_i"])
        df_videos = df_videos.append(dfVid.loc[dfVid.loc[:, "frame_i"] == maxFrame])
    df_videos = df_videos.reset_index(drop=True)
    df_videos=df_videos.drop(["path"],axis=1)
    df_videos["duracion"]=df_videos["frame_i"]*(1/30)

    sns.countplot(df_videos['emotion_label'])
    plt.title("Número de vídeos por emoción")
    plt.xlabel("Emociones")
    plt.ylabel("Número de vídeos")
    plt.show()

    sns.countplot(df_Frames['actor'])
    plt.title("Número de frames por actor")
    plt.xlabel("Actor")
    plt.ylabel("Número de frames")
    plt.show()

    videosPorEmocion=df_videos["emotion_label"].value_counts()
    print("Vídeos por emoción: ")
    print(videosPorEmocion)
    fig1, ax1 = plt.subplots()
    ax1.pie(videosPorEmocion.values, labels=videosPorEmocion.index, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title("Gráfico circular de videos por emoción")
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

    sns.boxplot(x="emotion_label", y="duracion", data=df_videos)
    plt.title("Diagrama de caja de duración de vídeo por emoción")
    plt.xlabel("Emociones")
    plt.ylabel("Duración vídeo (s)")
    plt.show()

    sns.boxplot(x="actor", y="duracion", data=df_videos)
    plt.title("Diagrama de caja de duración de vídeo por actor")
    plt.xlabel("Actores")
    plt.ylabel("Duración vídeo (s)")
    plt.show()
    frase=[]
    name=df_videos['video_name']
    for i in name:
        frase.append(int(i[12:14])) #Estos dos numeros indican la emocion de RAVDES

    df_videos["Frase"]=frase
    sns.boxplot(x="emotion_label", y="duracion", data=df_videos, hue='Frase')
    plt.title("Diagrama de caja de duración de vídeo por emoción")
    plt.xlabel("Emociones")
    plt.ylabel("Duración vídeo (s)")
    plt.show()

    sns.boxplot(x="actor", y="duracion", data=df_videos, hue='Frase')
    plt.title("Diagrama de caja de duración de vídeo por actor")
    plt.xlabel("Actores")
    plt.ylabel("Duración vídeo (s)")
    plt.show()



