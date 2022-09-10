from __future__ import print_function
import argparse

import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

#-kf 2 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -bs 128 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/ -m saliency -tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_STN_7classes/20210913_144129/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-153-original.pt -nClasses 7 -emb2extract posteriors
#para el de 3 clases cambiar numero de clases a 3 y usar modelo affecnet saliency y uno de epocas altas el ultimo complete
#-kf 2 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -bs 128 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/ -m saliency -tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt -nClasses 3 -emb2extract posteriors

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from src.utils.plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_loaders.data_loaders import Plain_Dataset
from src.data_loaders.data_loader_land import Plain_Dataset_land
from src.data_loaders.data_loader_saliency import Plain_Dataset_saliency

from src.architectures.deep_emotion_saliency_FeatExtractor import Deep_Emotion_Saliency as Deep_Emotion_saliency_48x48
from src.architectures.deep_emotion_original import Deep_Emotion_Original as Deep_Emotion_Original_48x48
from src.architectures.deep_emotion_baseline import Deep_Emotion_Baseline as Deep_Emotion_Baseline_48x48

from src.utils.args_utils import str2bool
from sklearn.model_selection import KFold
from src.main_CV_Pytorch_correctDropOut import seed_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def get_embs_5CV(k_folds,test_dataset, batch_size, root_path_weights, modality, logs_path, n_classes_preTrained, type_emb):

    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Reduce in 1 the actor index
    test_dataset.df_file["actor"] -= 1
    n_actors = np.unique(test_dataset.df_file["actor"])
    # Select data to test
    for fold, (_, test_ids) in enumerate(kfold.split(n_actors)):
        # Print
        print(f'FOLD {fold}')
        print("ACTORS IN TEST: ", str(test_ids))
        test_idx = np.array(
            list(test_dataset.df_file.loc[test_dataset.df_file["actor"].isin(list(test_ids))].index))  # test_ids

        # Añado train para sacar posteriors y normalizar
        train_idx = np.array(list(test_dataset.df_file.loc[~test_dataset.df_file["actor"].isin(list(test_ids))].index))

        # GET SAMPLES FROM ACTORS:
        print('--------------------------------')
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)

        testloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=0,
                                pin_memory=True)
        #cambiar el sampler para hacer las de test o train
        testloaderTrain = DataLoader(test_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=0,
                                     pin_memory=True)

        # Load weights
        weigths_path = get_weigths(root_path_weights, fold)
        print("Loaded weight: ", weigths_path)
        # Eval model:
        fold_logs_path = os.path.join(logs_path, "fold" + str(fold))
        os.makedirs(fold_logs_path, exist_ok=True)
        df_embs = extract_embs(modality,  os.path.join(root_path_weights, weigths_path), testloaderTrain, test_dataset, fold_logs_path, type_emb,
                               n_classes=n_classes_preTrained)


def extract_embs(modality, weights, test_loader, test_dataset, logs_path, type_emb, layer=-1, n_classes = 8):
    #Load model:
    if (modality == "saliency" or modality == "landmarks"):
        net = Deep_Emotion_saliency_48x48(n_classes=n_classes, training=False)
    elif (modality == "original"):
        net = Deep_Emotion_Original_48x48(n_classes=n_classes, training=False)
    elif (modality == "baseline"):
        net = Deep_Emotion_Baseline_48x48(n_classes=n_classes, training=False)
    print("Deep Emotion:-", net)
    #Load previous weigths:
    if (weights != None):
        print("Transfer learning of the model...")
        loaded_wegiths_dict = torch.load(weights)
        try:
            net.load_state_dict(loaded_wegiths_dict, strict=False)
        except RuntimeError:
            # remove last layers
            print("ERROR LOADING MODEL, REMOVING LAST LAYER fc2")
            loaded_wegiths_dict.pop("fc2.weight")
            loaded_wegiths_dict.pop("fc2.bias")
            net.load_state_dict(loaded_wegiths_dict, strict=False)

    #net.load_state_dict(torch.load(weights))
    net.training = False
    net.to(device)
    net.eval()

    #Extract predictions:
    if modality == "saliency" or modality == "landmarks":
        posteriors, idx, embs_FC_810, embs_FC_50 = eval_landmSaliency(net, test_loader, device, type_emb)
    else:  # modality == "original" or modality == "baseline":
        posteriors, idx, embs_FC_810, embs_FC_50 = eval_original(net, test_loader, device, type_emb)


    if (type_emb == "fc810"):
        all_embs_FC_810 = embs_FC_810.cpu().detach().numpy()
        df_labels = pd.DataFrame(all_embs_FC_810, columns=["embs_"+str(i) for i in range(0,810)])
    elif (type_emb == "fc50"):
        all_embs_FC_50 = embs_FC_50.cpu().detach().numpy()
        df_labels = pd.DataFrame(all_embs_FC_50, columns=["embs_"+str(i) for i in range(0,50)])
    else:
        all_posteriors = np.concatenate(posteriors, axis=0)
        df_labels = pd.DataFrame(all_posteriors, columns=["embs_"+str(i) for i in range(0,all_posteriors.shape[-1])])
    #Add index
    all_idx = idx.cpu().detach().numpy()
    df_labels["idx"] = all_idx.astype(int)
    #Add path
    df_labels["name"] = test_dataset.df_file["path"].iloc[all_idx.astype(int)].values

    #save logs:
    os.makedirs(logs_path, exist_ok=True)
    df_labels.to_csv(os.path.join(logs_path, "df_embs_"+type_emb+".csv"), sep=";", header=True, index=False)
    return df_labels



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
    Ntotal = 0
    for video_name, df_video in group_vid:
        #Ponderated voting:
        top_classes = df_video["top_pred"].unique()
        top_scores = []
        for clase in top_classes:
            df_class = df_video.loc[df_video["top_pred"]==clase]
            avg_confidence = np.mean(df_class["top_conf"])
            score_class = 0.5*(avg_confidence)+0.5*(len(df_class))
            top_scores.append(score_class)
        index_class = np.argmax(np.array(top_scores))
        acierto += int(top_classes[index_class] == df_video["labels"].values[0])
        #max voting
        max_vot = df_video["top_pred"].mode()[0]
        acierto_maxVot += int(max_vot==df_video["labels"].values[0])
        Ntotal+=1
    avg_acc_weighted_video =  acierto/Ntotal
    avg_acc_maxVoting_video =  acierto_maxVot / Ntotal
    print("VAL ACCURACY VIDEO LEVEL (PONDERADO): ", str(avg_acc_weighted_video))
    print("VAL ACCURACY VIDEO LEVEL (MAX. VOTING): ", str(avg_acc_maxVoting_video))
    return avg_acc_weighted_video, avg_acc_maxVoting_video




def eval_landmSaliency(net, test_loader, device, type_emb):

    all_embs_FC_810 = torch.empty(0)
    all_embs_FC_810 = all_embs_FC_810.cpu()

    all_embs_FC_50 = torch.empty(0)
    all_embs_FC_50 = all_embs_FC_50.cpu()

    all_idx = torch.empty(0)
    all_idx = all_idx.cpu()
    posteriors_list = []
    counter = 1
    with torch.no_grad():
        for data, labels, land, idx in test_loader:
            print("Extracting "+str(counter*test_loader.batch_size))
            data, labels, land, idx = data.to(device), labels.to(device), land.to(device), idx.to(device)
            outFC1, outFC2, out = net(data, land)
            pred = F.softmax(out, dim=1)
            if(type_emb=="fc810"):
                all_embs_FC_810 = torch.cat((all_embs_FC_810, outFC1.cpu()), dim=0)
            elif (type_emb == "fc50"):
                all_embs_FC_50 = torch.cat((all_embs_FC_50, outFC2.cpu()), dim=0)
            else:
                posteriors_list.append(pred.cpu().numpy())
            all_idx = torch.cat((all_idx, idx.cpu()), dim=0)
            counter+=1
    return posteriors_list, all_idx, all_embs_FC_810, all_embs_FC_50


def eval_original(net, test_loader, device, type_emb):
    all_embs_FC_810 = torch.empty(0)
    all_embs_FC_810 = all_embs_FC_810.cpu()

    all_embs_FC_50 = torch.empty(0)
    all_embs_FC_50 = all_embs_FC_50.cpu()

    all_idx = torch.empty(0)
    all_idx = all_idx.cpu()
    posteriors_list = []
    with torch.no_grad():
        for data, labels, idx in test_loader:
            data, labels, idx = data.to(device), labels.to(device), idx.to(device)
            outFC1, outFC2, out = net(data)
            pred = F.softmax(out, dim=1)
            if (type_emb == "fc810"):
                all_embs_FC_810 = torch.cat((all_embs_FC_810, outFC1.cpu()), dim=0)
            elif (type_emb == "fc50"):
                all_embs_FC_50 = torch.cat((all_embs_FC_50, outFC2.cpu()), dim=0)
            else:
                posteriors_list.append(pred.cpu().numpy())
            all_idx = torch.cat((all_idx, idx.cpu()), dim=0)
    return posteriors_list, all_idx, all_embs_FC_810, all_embs_FC_50





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-kf', '--kfolds', type=int, help='Number of folds if CV', default=5)
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Training file with the dataset files (train.csv and test.csv)')
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the images')
    parser.add_argument('-l', '--landmark_root', type=str,
                        help='Root path with the landmarks or saliencies')

    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size', default=32)
    parser.add_argument('-nClasses', '--n_classes_preTrained', type=int, help='Number of classes of pre-trained model', default=3)
    parser.add_argument('-logs', '--logs_folder', type=str, help='Path to save logs of evaluation', default='./')
    parser.add_argument('-m','--modality', type=str, help='Choose the architecture of the model (baseline, original, landmarks or saliency)', default="original")
    parser.add_argument('-tl', '--tl_preTrained_weights', type=str, required=True,
                        help='Path to the pre-trained weigths folder with.pt files')
    parser.add_argument('-emb2extract', '--type_embedding', type=str, required=True,
                        help='Layer of embeddings to extract [fc810, fc50, posteriors]')

    args = parser.parse_args()
    seed_torch(seed=2020)

    transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    if args.modality == "saliency":
        test_dataset = Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                              dataroot_land=args.landmark_root, transform=transformation)
    elif args.modality == "landmarks":
        test_dataset = Plain_Dataset_land(csv_path=args.data, dataroot=args.data_root,
                                          dataroot_land=args.landmark_root, transform=transformation)
    else:  # args.modality == "original" or args.modality == "baseline":
        test_dataset = Plain_Dataset(csv_path=args.data, dataroot=args.data_root, transform=transformation)

    get_embs_5CV(args.kfolds, test_dataset, args.batch_size, args.tl_preTrained_weights, args.modality, args.logs_folder,
                 args.n_classes_preTrained, args.type_embedding)







