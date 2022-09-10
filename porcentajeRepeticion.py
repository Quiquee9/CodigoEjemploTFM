from __future__ import print_function
import argparse

import os, sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from src.utils.plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report

from src.data_loaders.data_loader_saliencyQuique import Plain_Dataset_saliency

from src.architectures.deep_emotion_saliencyQuique import Deep_Emotion_Saliency as Deep_Emotion_saliency_48x48

from src.utils.args_utils import str2bool
from sklearn.model_selection import KFold
from src.utils.args_utils import seed_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def porcentage(data1, data2,fold_With_CSV):
    print("Dataset pequeño: "+fold_With_CSV+"/"+data1)
    print("Dataset grande: "+fold_With_CSV+"/"+data2)

    porcetajes=[]
    for fold in [0, 1, 2, 3, 4]:
        #Si se compara con un fichero que no esta por fold, poner aqui y  comentar la otra
        todos="/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdQuartile/"
        fichero=todos+"labelsEmotion_Quartile_90.csv"
        df_Comp1 = pd.read_csv(fichero, sep=";", header=0)

        df_Comp1 = pd.read_csv(fold_With_CSV+"/fold"+str(fold)+"/"+data1, sep=";", header=0)
        df_Comp2 = pd.read_csv(fold_With_CSV+"/fold"+str(fold)+"/"+data2, sep=";", header=0)
        if(False):
            fold_With_CSV3="/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdQuartile/"
            data1_1=fold_With_CSV3+"labelsEmotion_Quartile_35.csv"
            fold_With_CSV2="/home/enriquediez/Guided-EMO-SpatialTransformer-main/data/FiltradoReg35"
            data2_2="labelsEmotion_Quartile_T0.4.csv"
            df_Comp1 = pd.read_csv(data1_1, sep=";", header=0)
            df_Comp2 = pd.read_csv(fold_With_CSV2+"/fold"+str(fold)+"/"+data2_2, sep=";", header=0)

        listaPeq = list(df_Comp1["path"])
        listaGran = list(df_Comp2["path"])

        #Devuelve los elementos comunes
        interseccion = list(set(listaPeq) & set(listaGran))
        interseccion.sort()
        print("Fold:",fold)
        #print("Porcentaje: "+str(len(interseccion)*100/len(listaGran)))
        #porcetajes.append(len(interseccion)*100/len(listaGran))
        print("Porcentaje: " + str(len(interseccion) * 100 / len(listaPeq)))
        porcetajes.append(len(interseccion)*100/len(listaPeq))
    print("Media:")
    print(round(np.mean(porcetajes),2))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='File')
    parser.add_argument('-d2', '--data2', type=str, required=True,
                        help='File2')
    parser.add_argument('-f', '--foldPath', type=str, required=True,
                        help='Root path with the folds')
    args = parser.parse_args()
    seed_torch(seed=2020)

    porcentage(args.data, args.data2,args.foldPath)
