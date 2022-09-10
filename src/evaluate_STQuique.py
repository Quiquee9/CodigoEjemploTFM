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

#-d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale
# -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -m saliency -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/logsQuique --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquique/20211030_204825_Modelo3clases/trained_models
def get_weigths(root_path_weights, fold=-1):
    list_of_possible_weights = os.listdir(root_path_weights)
    df_path_weights = pd.DataFrame(list_of_possible_weights, columns=["path"])
    df_path_weights[["tmp", "deepEmo", "totalEpochs", "bs", "lr", "fold", "imgSize", "currentEpoch", "modality"]] = \
        df_path_weights["path"].str.split("-", -2, expand=True)
    #Remove wrong rows:
    df_path_weights = df_path_weights.loc[df_path_weights["modality"].notna()]
    if(fold<0): #Load complete model:
        fold = "COMPLETE"
    pd.options.mode.chained_assignment = None  # default='warn'
    df_complete_models = df_path_weights.loc[df_path_weights["fold"] == str(fold)]
    df_complete_models["currentEpoch"] = pd.to_numeric(df_complete_models["currentEpoch"])
    df_complete_models_new = df_complete_models.sort_values(by="currentEpoch", ascending=False)

    #Get first row (last top model)
    try:
        top_path = df_complete_models_new["path"].iloc[0]
    except:
        print("ERROR!! There is no model for fold: ", str(fold))
        top_path = -1
    return top_path





def eval_5CV(k_folds,test_dataset, batch_size, root_path_weights, modality, logs_path):
    avg_acc = 0
    avg_acc_video_weigthed = 0
    avg_acc_video_maxVot = 0
    avg_acc_video_AUCVot = 0
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Reduce in 1 the actor index
    test_dataset.df_file["actor"] -= 1
    n_actors = np.unique(test_dataset.df_file["actor"])

    dataFrameForCMvideoALL = pd.DataFrame(columns=['label', 'predMaxVot', 'predAuc'])
    dataFrameForCMframeALL= pd.DataFrame(columns=['label', 'pred'])

    os.makedirs(logs_path, exist_ok=True)
    logTXT = os.path.join(logs_path, "log.txt")
    open(logTXT, 'a').close()
    #Select data to test
    for fold, (_, test_ids) in enumerate(kfold.split(n_actors)):
        # Print
        print(f'FOLD {fold}')
        print("ACTORS IN TEST: ", str(test_ids))
        test_idx = np.array(list(test_dataset.df_file.loc[test_dataset.df_file["actor"].isin(list(test_ids))].index))  # test_ids

        #Añado train para sacar posteriors y normalizar
        train_idx = np.array(list(test_dataset.df_file.loc[~test_dataset.df_file["actor"].isin(list(test_ids))].index))

        # GET SAMPLES FROM ACTORS:
        print('--------------------------------')
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        testloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=0,pin_memory=True)
        #Load weights
        weigths_path = get_weigths(root_path_weights, fold)
        print("Loaded weight: ", weigths_path)
        #Eval model:
        fold_logs_path = os.path.join(logs_path, "fold"+str(fold))
        os.makedirs(fold_logs_path, exist_ok=True)
        accuracy, acc_video_weighted, acc_video_maxVoting, acc_AUC_video,dataFrameForCMvideo,dataFrameForCMframe = eval_nw(modality, os.path.join(root_path_weights,weigths_path), testloader, test_dataset, fold_logs_path)
        dataFrameForCMvideoALL=pd.concat([dataFrameForCMvideoALL,dataFrameForCMvideo])
        dataFrameForCMframeALL=pd.concat([dataFrameForCMframeALL,dataFrameForCMframe])

        avg_acc+=accuracy
        avg_acc_video_weigthed += acc_video_weighted
        avg_acc_video_maxVot+=acc_video_maxVoting
        avg_acc_video_AUCVot+=acc_AUC_video

    classes = ('Neutra', 'Calma', 'Feliz', 'Triste', 'Ira', 'Miedo', 'Asco', 'Sorpresa')
    labels = dataFrameForCMframeALL["label"]
    pred = dataFrameForCMframeALL["pred"]
    print("Results for total frame level:")
    cm = confusion_matrix(labels, pred)
    plot_confusion_matrix(cm, classes, os.path.join(logs_path, "FrameTotalcm.png"),title="Matriz de confusión a nivel de fotograma")
    plt.tight_layout()
    plt.show()
    print(classification_report(labels, pred, target_names=classes, digits=4))

    labels = dataFrameForCMvideoALL["label"]
    predsMax = dataFrameForCMvideoALL["predMaxVot"]
    predsAuc = dataFrameForCMvideoALL["predAuc"]
    print("Results for Max voting and AUC:")
    cm = confusion_matrix(labels, predsMax)
    plot_confusion_matrix(cm, classes, os.path.join(logs_path, "VideoMaxVotinTotalcm.png"))
    plt.tight_layout()
    plt.show()
    print(classification_report(labels, predsMax, target_names=classes, digits=4))

    cm = confusion_matrix(labels, predsAuc)
    plot_confusion_matrix(cm, classes, os.path.join(logs_path, "VideoAucVotinTotalcm.png"),title="Matriz de confusión a nivel de vídeo")
    plt.tight_layout()
    plt.show()
    print(classification_report(labels, predsAuc, target_names=classes, digits=4))


    print("FINAL ACCURACY (FRAME LEVEL) :", avg_acc/5)
    print("FINAL ACCURACY WEIGHTED (VIDEO LEVEL) :", avg_acc_video_weigthed / 5*100)
    print("FINAL ACCURACY MAX. VOTING (VIDEO LEVEL) :", avg_acc_video_maxVot / 5*100)
    print("FINAL ACCURACY MAX. AUC (VIDEO LEVEL) :", avg_acc_video_AUCVot / 5*100)





def eval_nw(modality, weights, test_loader, test_dataset, logs_path):
    #Load model:
    if (modality == "saliency" or modality == "landmarks"):
        net = Deep_Emotion_saliency_48x48(n_classes=8, training=False)

    #print("Deep Emotion:-", net)
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    #Model Evaluation on test data
    classes = ('Neutral','Calm','Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise')

    #Extract predictions:
    if modality == "saliency" or modality == "landmarks":
        preds, posteriors, labels, idx = eval_landmSaliency(net, test_loader, device)
    else:  # modality == "original" or modality == "baseline":
        preds, posteriors, labels, idx = eval_original(net, test_loader, device)

    all_labels = labels.cpu().detach().numpy()
    all_preds = preds.cpu().detach().numpy()
    all_idx = idx.cpu().detach().numpy()
    all_posteriors = np.concatenate(posteriors, axis=0)


    #PREDICTION AT VIDEO LEVEL:
    avg_acc_weighted_video, avg_acc_maxVoting_video, avg_acc_AUC_video,dataFrameForCMvideo = get_accuracy_on_video(all_posteriors, all_labels, all_idx, test_dataset)

    # PREDICTION AT FRAME LEVEL:
    #Extract metrics: accuracy, confussion matrix...
    correct = torch.where(preds == labels, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda()).sum()
    accuracy = 100.0 * (correct / len(all_labels)).cpu().detach().numpy()
    print("Accuracy: ", str(100.0 * (correct / len(all_labels))))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes, os.path.join(logs_path, "cm.png"))
    plt.close()

    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    dataFrameForCMframe= pd.DataFrame(columns=['label', 'pred'])
    dataFrameForCMframe["label"]=all_labels
    dataFrameForCMframe["pred"]=all_preds

    df_labels = pd.DataFrame([],columns=[])
    df_labels["preds"] = all_preds
    df_labels["labels"] = all_labels
    df_labels["idx"] = all_idx.astype(int)
    df_labels["path"] = test_dataset.df_file["path"].iloc[all_idx.astype(int)].values

    #save logs:
    os.makedirs(logs_path, exist_ok=True)
    df_labels.to_csv(os.path.join(logs_path, "df_predictions.csv"), sep=";", header=True, index=False)
    return accuracy, avg_acc_weighted_video, avg_acc_maxVoting_video, avg_acc_AUC_video,dataFrameForCMvideo, dataFrameForCMframe




def get_accuracy_on_video(preds_list, labels_list, idx_list,test_dataset):
    # preds_list = np.concatenate(preds_list, axis=0)
    # labels_list = np.concatenate(labels_list, axis=0)
    # idx_list = np.concatenate(idx_list, axis=0)
    df_test = test_dataset.df_file
    df_test_reorder = df_test.loc[idx_list].reset_index(inplace=False, drop=True)
    df_test_reorder[["pred0","pred1","pred2","pred3","pred4","pred5","pred6","pred7"]] = preds_list
    df_test_reorder["top_pred"] = preds_list.argmax(axis=-1)
    df_test_reorder["top_conf"] = preds_list.max(axis=-1)
    df_test_reorder["labels"] = labels_list
    #Order by name of audio
    group_vid = df_test_reorder.groupby(by="video_name")
    acierto = 0
    acierto_maxVot = 0
    acierto_AUCVot = 0
    Ntotal = 0
    dataFrameForCMvideo  = pd.DataFrame(columns=['label', 'predMaxVot', 'predAuc'])
    for video_name, df_video in group_vid:
        # Ponderated voting:
        top_classes = df_video["top_pred"].unique()
        top_scores = []
        for clase in top_classes:
            df_class = df_video.loc[df_video["top_pred"] == clase]
            avg_confidence = np.mean(df_class["top_conf"])
            score_class = 0.5 * (avg_confidence) + 0.5 * (len(df_class))
            top_scores.append(score_class)
        index_class = np.argmax(np.array(top_scores))
        acierto += int(top_classes[index_class] == df_video["labels"].values[0])
        # max voting
        max_vot = df_video["top_pred"].mode()[0]
        acierto_maxVot += int(max_vot == df_video["labels"].values[0])
        Ntotal += 1
        # AUC suma
        AUC_list = []
        AUC_list.append(sum(df_video["pred0"]))
        AUC_list.append(sum(df_video["pred1"]))
        AUC_list.append(sum(df_video["pred2"]))
        AUC_list.append(sum(df_video["pred3"]))
        AUC_list.append(sum(df_video["pred4"]))
        AUC_list.append(sum(df_video["pred5"]))
        AUC_list.append(sum(df_video["pred6"]))
        AUC_list.append(sum(df_video["pred7"]))
        AUC_vot = AUC_list.index(max(AUC_list))
        acierto_AUCVot += int(AUC_vot == df_video["labels"].values[0])
        """
        quartile=35
        AUC_list = []
        AUC_list.append(sum(df_video["pred0"]*(df_video["pred0"]>=np.percentile(df_video["pred0"], quartile))))
        AUC_list.append(sum(df_video["pred1"]*(df_video["pred1"]>=np.percentile(df_video["pred1"], quartile))))
        AUC_list.append(sum(df_video["pred2"]*(df_video["pred2"]>=np.percentile(df_video["pred2"], quartile))))
        AUC_list.append(sum(df_video["pred3"]*(df_video["pred3"]>=np.percentile(df_video["pred3"], quartile))))
        AUC_list.append(sum(df_video["pred4"]*(df_video["pred4"]>=np.percentile(df_video["pred4"], quartile))))
        AUC_list.append(sum(df_video["pred5"]*(df_video["pred5"]>=np.percentile(df_video["pred5"], quartile))))
        AUC_list.append(sum(df_video["pred6"]*(df_video["pred6"]>=np.percentile(df_video["pred6"], quartile))))
        AUC_list.append(sum(df_video["pred7"]*(df_video["pred7"]>=np.percentile(df_video["pred7"], quartile))))
        AUC_vot = AUC_list.index(max(AUC_list))
        acierto_AUCVot += int(AUC_vot == df_video["labels"].values[0])
        """

        df2 = {'label': df_video["labels"].values[0], 'predMaxVot': max_vot, 'predAuc': AUC_vot}
        dataFrameForCMvideo = dataFrameForCMvideo.append(df2, ignore_index=True)
    avg_acc_weighted_video = acierto/Ntotal
    avg_acc_maxVoting_video = acierto_maxVot / Ntotal
    avg_acc_AUC_video = acierto_AUCVot / Ntotal
    print("VAL ACCURACY VIDEO LEVEL (PONDERADO): ", str(avg_acc_weighted_video))
    print("VAL ACCURACY VIDEO LEVEL (MAX. VOTING): ", str(avg_acc_maxVoting_video))
    print("VAL ACCURACY VIDEO LEVEL (MAX. AUC): ", str(avg_acc_AUC_video))

    return avg_acc_weighted_video, avg_acc_maxVoting_video, avg_acc_AUC_video,dataFrameForCMvideo




def eval_landmSaliency(net, test_loader, device):
    all_preds = torch.empty(0)
    all_preds = all_preds.to(device)
    all_labels = torch.empty(0)
    all_labels = all_labels.to(device)
    all_idx = torch.empty(0)
    all_idx = all_idx.to(device)
    posteriors_list = []
    with torch.no_grad():
        for data, labels, land, idx in test_loader:
            data, labels, land, idx = data.to(device), labels.to(device), land.to(device), idx.to(device)
            outputs = net(data, land)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            all_preds = torch.cat((all_preds, classs), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_idx = torch.cat((all_idx, idx), dim=0)
            posteriors_list.append(pred.cpu().numpy())
            # wrong = torch.where(classs != labels, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
            # acc = 1 - (torch.sum(wrong) / 64)
            # total.append(acc.item())
    return all_preds, posteriors_list, all_labels, all_idx


def eval_original(net, test_loader, device):
    all_preds = torch.empty(0)
    all_preds = all_preds.to(device)
    all_labels = torch.empty(0)
    all_labels = all_labels.to(device)
    all_idx = torch.empty(0)
    all_idx = all_idx.to(device)
    posteriors_list = []
    with torch.no_grad():
        for data, labels, idx in test_loader:
            data, labels, idx = data.to(device), labels.to(device), idx.to(device)
            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            all_preds = torch.cat((all_preds, classs), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_idx = torch.cat((all_idx, idx), dim=0)
            posteriors_list.append(pred.cpu().numpy())
    return all_preds, posteriors_list, all_labels, all_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Training file with the dataset files (train.csv and test.csv)')
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the images')
    parser.add_argument('-l', '--landmark_root', type=str,
                        help='Root path with the landmarks or saliencies')

    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size', default=32)
    parser.add_argument('-logs', '--logs_folder', type=str, help='Path to save logs of evaluation', default='./')
    parser.add_argument('-m','--modality', type=str, help='Choose the architecture of the model (baseline, original, landmarks or saliency)', default="original")
    parser.add_argument('-tl', '--tl_preTrained_weights', type=str, required=True,
                        help='Path to the pre-trained weigths folder with.pt files')

    args = parser.parse_args()
    seed_torch(seed=2020)

    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    if args.modality == "saliency":
        test_dataset = Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                              dataroot_land=args.landmark_root, transform=transformation)


    eval_5CV(5, test_dataset, args.batch_size, args.tl_preTrained_weights, args.modality, args.logs_folder)







