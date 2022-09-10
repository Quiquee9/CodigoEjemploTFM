import os

import pandas as pd

def get_GMM(emotions_AffectNet, df_AUs_labels, out_path):
    """
    cols2select = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
                   "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r", "AU01_c", "AU02_c", "AU04_c",
                   "AU05_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c",
                   "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c"]
    """
    cols2select = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
                   "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

    df_means = pd.DataFrame([], columns=cols2select+["emotion"])
    df_std = pd.DataFrame([], columns=cols2select+["emotion"])
    for k,v in emotions_AffectNet.items():
        df_emotion_i = df_AUs_labels.loc[df_AUs_labels["expression"] == k]
        #Get avg and variance per AU
        df_aux = df_emotion_i[cols2select].mean()
        df_aux["emotion"] = v
        df_means = df_means.append(pd.DataFrame([df_aux], columns=cols2select+["emotion"]))
        #STD
        df_aux = df_emotion_i[cols2select].std()
        df_aux["emotion"] = v
        df_std = df_std.append(pd.DataFrame([df_aux], columns=cols2select+["emotion"]))
        print(v, ": ", len(df_emotion_i))
    df_means.to_csv(os.path.join(out_path, "GMM_mean.csv"), index=False, sep=";")
    df_std.to_csv(os.path.join(out_path, "GMM_std.csv"), index=False, sep=";")

def get_GMM_covarianceMatris(emotions_AffectNet, df_AUs_labels, out_path):
    cols2select = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
                   "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

    for k,v in emotions_AffectNet.items():
        df_emotion_i = df_AUs_labels.loc[df_AUs_labels["expression"] == k]
        #Get avg and variance per AU
        df_aux = df_emotion_i[cols2select]
        covMatrix=df_aux.cov()
        covMatrix.to_csv(os.path.join(out_path, "GMM_covM_"+v+".csv"), index=False, sep=";")
if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    path_AffectNet_AUs_labels= "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/AUS_with_labels.csv"
    out_path = "/home/enriquediez/Guided-EMO-SpatialTransformer-main/EstudioAUs/GMM_AUs"
    df_AUs_labels = pd.read_csv(path_AffectNet_AUs_labels, sep=";", header = 0)

    emotions_AffectNet = {
        0:"Neutral", #75374
        1:"Happy", #134914
        2:"Sad", #25958
        3:"Surprise", # 14590
        4:"Fear", #6878
        5:"Disgust", #4303
        6:"Anger", #25382
    }
    get_GMM(emotions_AffectNet, df_AUs_labels, out_path)
    get_GMM_covarianceMatris(emotions_AffectNet, df_AUs_labels, out_path)
"""
    0: "Neutral",  # 75374
    1: "Happy",  # 134915
    2: "Sad",  # 25959
    3: "Surprise",  # 14590
    4: "Fear",  # 6878
    5: "Disgust",  # 4303
    6: "Anger",  # 25382

    Obtenidas:
    Neutral :  74015
Happy :  132879
Sad :  25460
Surprise :  14330
Fear :  6712
Disgust :  4233
Anger :  24854

"""


