import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter

if __name__ == '__main__':

    pathCSV_post = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsRADVESS7" \
                   "/df_embs_posteriors.csv"
    pathCSV_comp = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution" \
                   "/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv"
    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dict = {1:"Neutral",
                 2:"Calm",
                 3:"happy",
                 4:"sad",
                 5:"angry",
                 6:"fear",
                 7:"disgust",
                 8:"surprise"}
    #Lee el CSV con todos los posteriors y el completo
    df_Posteriors = pd.read_csv(pathCSV_post, sep=";", header=0)
    df_Comp = pd.read_csv(pathCSV_comp, sep=";", header=0)
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
    # Cambio el posterior de angry con el surprise
    embSurprise = list(df_Posteriors["embs_3"])
    embAngry = list(df_Posteriors["embs_6"])
    df_Posteriors["embs_3"] = embAngry
    df_Posteriors["embs_6"] = embSurprise

    #Da la columna maxima de embs
    #y_predText = df_Posteriors[["embs_0", "embs_1", "embs_2", "embs_3", "embs_4", "embs_5", "embs_6"]].idxmax(axis=1)
    #y_pred=[]
    #Pasar de texto a numero
    #for pred in y_predText:
    #    y_pred.append(int(pred[5]))

    emot_dict2 = {0: "neutral and calm",
                 1: "happy",
                 2: "sad",
                 3: "angry",
                 4: "fear",
                 5: "disgust",
                 6: "surprise"}

    emot_dict = {0: "neutral",
                 1: "calm",
                 2: "happy",
                 3: "sad",
                 4: "angry",
                 5: "fear",
                 6: "disgust",
                 7: "surprise"}
    numVideosPerEmotion = {0: '288',
                            1: '192',
                            2: '192',
                            3: '192',
                            4: '192',
                            5: '192',
                            6: '192'}

    listQuartiles = [42.5]
    for quartile in listQuartiles:
        dfThreshTrain = pd.DataFrame(columns=list(df_Posteriors.columns))  # mismas columnas que posteriors
        dfThreshTest = pd.DataFrame(columns=list(df_Posteriors.columns))  # mismas columnas que posteriors

        for idVideo in namesUniques:
            dfVid = df_Posteriors.loc[df_Posteriors.loc[:, 'idVideo'] == idVideo]
            nameVideo = idVideo
            #La emocion del video
            realEmotion=dfVid["emotion"].unique()[0]
            threshold = np.percentile(dfVid.loc[:, "embs_"+str(realEmotion)], quartile)
            #Añado los posteriors de los frames que superan el umbral, mantienen idx originales
            dfThreshTrain=dfThreshTrain.append(dfVid.loc[dfVid.loc[:, "embs_"+str(realEmotion)] >= threshold])


        df_CompFiltradoTrain=df_Comp.loc[dfThreshTrain['idx']]

        print("Quartile: "+str(quartile))
        #De 0 a 7
        for emotion in range(8):
            indicesEmotion=df_CompFiltradoTrain.loc[df_CompFiltradoTrain.loc[:, 'emotion'] == emotion].index
            framesEmotion = len(df_Comp.loc[df_Comp.loc[:, 'emotion'] == emotion].index)
            framesEmotion_filtTrain = len(df_CompFiltradoTrain.loc[df_CompFiltradoTrain.loc[:, 'emotion'] == emotion].index)
            print("Emotion: "+emot_dict[emotion])
            print("Frames totales: "+str(framesEmotion)+"| Frames filtrados:  "+str(framesEmotion_filtTrain))
            print("Porcentaje Train: "+str(np.round(framesEmotion_filtTrain/framesEmotion*100,decimals=2))+" %")

        print("Frames totales :"+str(len(df_CompFiltradoTrain))+"/"+str(len(df_Comp)))
        print("-----------------------------------------------------")

        path="/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdQuartile"
        os.makedirs(path, exist_ok=True)
        df_CompFiltradoTrain.to_csv(os.path.join(path, "labelsEmotion_Quartile_" + str(quartile) + ".csv"), sep=";", header=True, index=False)


