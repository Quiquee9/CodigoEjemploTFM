import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    #https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    #pathCSV = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsTrain8/Posteriorq25Fold0.csv"
    pathCSV ="/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsTrain8/PosteriorTodoFold0.csv"
    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dict = {0:"Neutra",
                 1:"Calma",
                 2:"Feliz",
                 3:"Triste",
                 4:"Ira",
                 5:"Miedo",
                 6:"Asco",
                 7:"Sorpresa"}
    #Lee el CSV con todos los posteriors
    df_Posteriors = pd.read_csv(pathCSV, sep=";", header=0)
    df_Posteriors["actor"]+=1
 #Posteriors se hicieron con creadorPosterior.py
    namesUniques=df_Posteriors["video_name"].unique()
    for idVideo in namesUniques:

        dfVid = df_Posteriors.loc[df_Posteriors.loc[:, 'video_name'] == idVideo]
        nameVideo = idVideo
        emotionVideo = emot_dict[dfVid["emotion"].unique()[0]]
        actorVideo = dfVid["actor"].unique()[0]
        plt.title("ID vídeo: "+idVideo+' | Emoción: '+emotionVideo)
        x = range(len(dfVid))
        plt.plot(x, dfVid["pred0"], color='blue', label='Neutra')
        plt.plot(x, dfVid["pred1"], color='cyan', label='Calma')
        plt.plot(x, dfVid["pred2"], color='green', label='Feliz')
        plt.plot(x, dfVid["pred3"], color='red', label='Triste')
        plt.plot(x, dfVid["pred4"], color='black', label='Ira')
        plt.plot(x, dfVid["pred5"], color='purple', label='Miedo')
        plt.plot(x, dfVid["pred6"], color='pink', label='Asco')
        plt.plot(x, dfVid["pred7"], color='yellow', label='Sorpresa')

        plt.legend()  # Mostrar leyenda

        plt.xlabel('Número de frame')
        plt.ylabel('Probabilidad')

        os.makedirs(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsPlot8Todo/EMOTIONS", emotionVideo), exist_ok=True)
        os.makedirs(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsPlot8Todo/ACTOR", str(actorVideo)), exist_ok=True)

        plt.savefig(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsPlot8Todo/EMOTIONS",emotionVideo, idVideo+".png"))
        plt.savefig(os.path.join("/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioDePosteriors/posteriorsPlot8Todo/ACTOR", str(actorVideo),idVideo+".png"))

        plt.close('all')



