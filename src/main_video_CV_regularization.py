from __future__ import print_function
import argparse

import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data_loaders.data_loader_saliencyQuique import Plain_Dataset_saliency

from src.architectures.deep_emotion_saliencyQuiqueRegulariz import Deep_Emotion_Saliency as Deep_Emotion_saliency_48x48

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from src.utils.args_utils import str2bool
from time import time  # importamos la función time para capturar tiempos
import datetime
#REDUCE RANDOMNESS:
import random
import numpy as np
from src.utils.args_utils import seed_torch
#   -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv
#    -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500
#    -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquique -m saliency
#    --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_STN_7classes/20210913_144129/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-153-original.pt

# 2º tl data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def Train(epochs,k_folds,train_dataset,test_dataset,device,img_size=48, class_weigths=None,
          nb_layers=1, nb_lstm_units=100, embedding_dim=3, batch_size=3,classes=8, save_path="./",
          num_epochs_stop=20, l2=0.0, modality="original", preTrainedWeigths=None, lr=0.001, train_complete_model_flag=False,
          num_workers=0):

    '''
    Training Loop
    '''

    #Para fijar actores comento esto y hago diccionario
    #kfold = KFold(n_splits=k_folds, shuffle=True)
    #n_actors = np.unique(train_dataset.df_file["actor"])
    results = {}
    epochs2converge = {}
    print("SAVING DATA IN: ", (save_path))

    '''
    Training Loop
    '''
    #Reduce in 1 the actor index
    train_dataset.df_file["actor"] -= 1
    test_dataset.df_file["actor"] -= 1
    dict_test_actors_folds = {
        0: [1, 4, 13, 14, 15],
        1: [2, 5, 6, 12, 17],
        2: [9, 10, 11, 18, 19],
        3: [7, 16, 20, 22, 23],
        4: [0, 3, 8, 21]
    }
    tiempo_ejecucion = []
    print("===================================Start Training===================================")
    for fold in [0, 1, 2, 3, 4]:
        tiempo_inicial = time()
        # Print
        print(f'FOLD {fold}')
        print("ACTORS IN TEST: ", str(dict_test_actors_folds[fold]))
        train_idx = np.array(
            list(train_dataset.df_file.loc[~train_dataset.df_file["actor"].isin(list(dict_test_actors_folds[fold]
                                                                                     ))].index))  # train_ids
        #train_idx=train_idx[range(0, len(train_idx), 2)]

        test_idx = np.array(
            list(test_dataset.df_file.loc[test_dataset.df_file["actor"].isin(list(dict_test_actors_folds[fold]
                                                                                  ))].index))  # test_ids

        # CREATE WRITER PER FOLD:
        writer = SummaryWriter(log_dir=os.path.join(save_path, "logs", "fold_" + str(fold)))

        #GET SAMPLES FROM ACTORS:
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        #print("TEST INDX: ", test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=0)
        testloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=0)

        net = Deep_Emotion_saliency_48x48()


        # START LEARNING WITH PRE-TRAINED WEIGHTS FROM OTHER DS (OR NOT) - TRANSFER LEARNING
        if (preTrainedWeigths != None):
            print("Transfer learning of the model...")
            loaded_wegiths_dict = torch.load(preTrainedWeigths)
            try:
                net.load_state_dict(loaded_wegiths_dict, strict=False)
            except RuntimeError:
                #remove last layers
                loaded_wegiths_dict.pop("fc2.weight")
                loaded_wegiths_dict.pop("fc2.bias")
                net.load_state_dict(loaded_wegiths_dict, strict=False)



        net.to(device)
        criterion = nn.CrossEntropyLoss()
        #añado L2 penalty
        optmizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
        # Create output directory of nws:
        os.makedirs(os.path.join(save_path, "trained_models"), exist_ok=True)
        # Early Stopping parameters:
        epochs_no_improve = 0
        min_val_acc = 0
        last_top_acc = 0
        last_top_acc_epoch = 0

        print("------------------ START TRAINING of fold ---------------------")
        for e in range(epochs):
            print('\nEpoch {} / {} \nFold number {} / {}'.format(e + 1, epochs, fold + 1, k_folds))
            # Train the model  #
            net.train()
            net.trainingState = True

            if modality == "saliency" or modality == "landmarks":
                train_loss, train_correct = train_model_saliency(net, train_loader, optmizer, criterion, device)
            else:# modality == "original" or modality == "baseline":
                train_loss, train_correct = train_model_original(net, train_loader, optmizer, criterion, device)

            print(">>>>>>>>>>>> ON EPOCH: ", str(e))
            print(">> training : ")
            train_loss = (train_loss / len(train_idx))
            writer.add_scalar("Loss/train", train_loss, e)
            train_acc = train_correct.double() / len(train_idx) * 100
            writer.add_scalar("Accuracy/train", train_acc, e)

            print('TRAINING: Fold: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(fold, e, train_loss,train_acc))

            #evaluate the fold#
            print(">> eval : ")
            net.eval()
            net.trainingState=False
            if modality == "saliency" or modality == "landmarks":
                validation_loss, val_correct = eval_model_saliencyORlandmarks(net, testloader, criterion, device)
            else: #modality == "original" or modality == "baseline":
                validation_loss, val_correct = eval_model_original(net, testloader, criterion, device)

            validation_loss = validation_loss / len(test_idx)
            results[fold] = 100.0 * (val_correct / len(test_idx))
            epochs2converge[fold] = e
            # Print accuracy
            print('VALIDATION: Fold: {}. Iteration: {}. Loss: {}. Accuracy: {}'.format(fold, e, validation_loss, results[fold]))
            print('--------------------------------')

            writer.add_scalar("Loss/val", validation_loss, e)
            writer.add_scalar("Accuracy/val", results[fold], e)
            # Send data at the end of the epoch
            writer.flush()

            # EARLY STOPPING:
            if results[fold] > min_val_acc:
                # Save the model
                # Save BEST weigths to recover them posteriorly
                torch.save(net.state_dict(),
                           os.path.join(save_path, "trained_models",
                                        'TMP-deep_emotion-{}-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr,
                                                                                                fold,
                                                                                                img_size, e, modality)))
                epochs_no_improve = 0
                min_val_acc = results[fold]
                last_top_acc = min_val_acc#100.0 * (val_correct / len(test_ids))
                last_top_acc_epoch = e
                epochs2converge[fold] = e

            else:
                epochs_no_improve += 1

            if e > (num_epochs_stop - 1) and epochs_no_improve == num_epochs_stop:
                print('Early stopping IN EPOCH: !', str(e), " - Best weigths saved in TMP-deep_emotion....")
                # EARLY STOPPING
                results[fold] = last_top_acc
                epochs2converge[fold] = last_top_acc_epoch
                break

        tiempo_final = time()
        tiempo_ejecucion.append(tiempo_final - tiempo_inicial)
        # save LAST MODEL
        torch.save(net.state_dict(),
                   os.path.join(save_path, "trained_models",
                                'deep_emotion-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr,
                                                                                         fold, img_size,modality)))
        print("===================================Training Finished===================================")
        # Close tensorboard writer
        writer.flush()
        writer.close()

        # RECOVER BEST MODEL BASED ON VALIDATION:
        top_model_weights = os.path.join(save_path, "trained_models",
                                         'TMP-deep_emotion-{}-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size,
                                                                                                 lr, fold,
                                                                                                 img_size,
                                                                                                 last_top_acc_epoch,
                                                                                           modality))
        print(">>> Loading TOP nw for test eval: ", top_model_weights)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    epochs2stop = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
        epochs2stop+=epochs2converge[key]
    print(f'Average: {sum / len(results.items())} %')
    print(f'Average convergence epochs: {epochs2stop / len(results.items())} ')
    print('Mean time training per fold was: {}'.format(datetime.timedelta(seconds=np.mean(tiempo_ejecucion))))
    print(tiempo_ejecucion)
    # RUN & SAVE LAST MODEL TRAINING
    if (train_complete_model_flag):
        print("TRAINING LAST MODEL ...")
        if (class_weigths):
            # Obtain classes for training with unbalanced DS
            class_weigths_values = check_balance_in_data(train_dataset.df_file)
        else:
            class_weigths_values = None
        writer = SummaryWriter(log_dir=os.path.join(save_path, "logs", "COMPLETE"))
        train_complete_model(epochs, train_dataset, device, writer, img_size=img_size, class_weigths=class_weigths_values,
                             batch_size=batch_size, lr=lr, save_path=save_path, modality=modality, num_workers=num_workers,
                             num_epochs_stop=num_epochs_stop)
        writer.flush()
        writer.close()



def train_model_original(net, train_loader, optmizer, criterion, device):
    train_loss = 0
    train_correct = 0
    for data, labels, _ in train_loader:
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optmizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

        loss.backward()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        #Update model's weigths
        optmizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    return train_loss, train_correct

def train_model_saliency(net, train_loader, optmizer, criterion, device):
    train_loss = 0
    train_correct = 0
    for data, labels, saliency, _ in train_loader:
        data, labels, saliency = data.to(device, non_blocking=True), labels.to(device, non_blocking=True), saliency.to(device, non_blocking=True)
        optmizer.zero_grad()
        outputs = net(data,saliency)
        loss = criterion(outputs, labels)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

        loss.backward()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        #Update model's weigths
        optmizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    return train_loss, train_correct

def eval_model_saliencyORlandmarks(net, testloader, criterion, device):
    val_correct, validation_loss = 0, 0
    preds_list = []
    labels_list = []
    idx_list = []
    with torch.no_grad():
        for data, labels, land, idx in testloader:
            data, labels, land = data.to(device, non_blocking=True), labels.to(device, non_blocking=True), land.to(device, non_blocking=True)
            outputs = net(data, land)
            validation_loss += criterion(outputs, labels).item()
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            val_correct += (classs == labels).sum().item()
            #Append data:
            preds_list.append(pred.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            idx_list.append(idx.cpu().numpy())
    #get_accuracy_on_video(preds_list, labels_list, idx_list, testloader)
    return validation_loss, val_correct

def eval_model_original(net, testloader, criterion, device):
    val_correct, validation_loss = 0, 0
    with torch.no_grad():
        for data, labels, _ in testloader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = net(data)
            validation_loss += criterion(outputs, labels).item()
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            val_correct += (classs == labels).sum().item()
    return validation_loss, val_correct


def get_accuracy_on_video(preds_list, labels_list, idx_list,testloader):
    preds_list = np.concatenate(preds_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    idx_list = np.concatenate(idx_list, axis=0)
    df_test = testloader.dataset.df_file
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
    print("VAL ACCURACY VIDEO LEVEL (PONDERADO): ", str(acierto/Ntotal))
    print("VAL ACCURACY VIDEO LEVEL (MAX. VOTING): ", str(acierto_maxVot / Ntotal))






def train_complete_model(epochs,train_dataset,device, writer,img_size, class_weigths, batch_size, lr,
          save_path, modality, num_workers, num_epochs_stop):

    train_loader= DataLoader(train_dataset,batch_size=batch_size,shuffle = True,num_workers=num_workers, pin_memory=True)

    if img_size == 48 and (modality == "saliency" or modality == "landmarks"):
        net = Deep_Emotion_saliency_48x48()


    net.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weigths)
    optmizer = optim.Adam(net.parameters(), lr=lr)

    epochs_no_improve = 0
    min_val_acc = 0
    last_top_acc = 0
    last_top_acc_epoch = 0
    epochs2converge = 0
    results = 0
    print("------------------ START TRAINING of fold ---------------------")
    for e in range(epochs):
        print('\nEpoch {} / {} - FINAL MODEL'.format(e + 1, epochs))
        # Train the model  #
        net.train()
        net.trainingState = True
        # Train 1 epoch
        if modality == "saliency" or modality == "landmarks":
            train_loss, train_correct = train_model_saliency(net, train_loader, optmizer,
                                                             criterion, device)
        else: # modality == "original" or modality == "baseline":
            train_loss, train_correct = train_model_original(net, train_loader, optmizer,
                                                             criterion, device)

        # Print validation accuracy & save
        print(">>>>>>>>>>>> ON EPOCH: ", str(e))
        print(">> training : ")
        train_loss = (train_loss / len(train_dataset))
        print("TRAIN LOSS: ", str(train_loss))
        writer.add_scalar("Loss/train", train_loss, e)
        train_acc = 100*(train_correct.double() / len(train_dataset))
        print("Train Accuracy: ", str(train_acc))
        writer.add_scalar("Accuracy/train", train_acc, e)
        # EARLY STOPPING:
        if train_acc > min_val_acc:
            # Save the model
            # Save BEST weigths to recover them posteriorly
            torch.save(net.state_dict(),
                       os.path.join(save_path, "trained_models",
                                    'TMP-deep_emotion-{}-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr,
                                                                                      "COMPLETE",
                                                                                      img_size, e, modality)))
            epochs_no_improve = 0
            min_val_acc = train_acc
            last_top_acc = 100.0 *(train_correct.double() / len(train_dataset))
            last_top_acc_epoch = e
            epochs2converge = e

        else:
            epochs_no_improve += 1

        if e > (num_epochs_stop - 1) and epochs_no_improve == num_epochs_stop:
            print('Early stopping IN EPOCH: !', str(e), " - Best weigths saved in TMP-deep_emotion....")
            # EARLY STOPPING
            results = last_top_acc
            epochs2converge = last_top_acc_epoch
            break

    print("EPOCHS TO CONVERGE: ", str(epochs2converge), ", ACC: ", str(results))
    # save final model
    torch.save(net.state_dict(),
               os.path.join(save_path, "trained_models",
                            'deep_emotion-{}-{}-{}-{}-{}-{}.pt'.format(epochs, batch_size, lr, "COMPLETE", img_size,
                                                                       modality)))
    print("===================================Training Finished===================================")


def check_balance_in_data(df_data):
    #df_data = pd.read_csv(traincsv_file, header=0)
    #PLOT LABELS TO SEE DISTRIBUTION OF DATA
    labels_complete = df_data["emotion"]
    classes = labels_complete.unique()
    import matplotlib.pyplot as plt
    plt.hist(labels_complete)
    plt.xlabel("Classes")
    plt.ylabel("Number of samples per class")
    plt.xticks(classes)
    plt.show()
    #EXTRACT WEIGHTS TO COMPENSATE UNBALANDED DATASETS
    dict_classes_weigths = {}
    for cls in classes:
        n_clss = len(df_data.loc[df_data["emotion"] == cls])
        dict_classes_weigths[cls] = n_clss
    min_v = min(list(dict_classes_weigths.values()))
    for cls in classes:
        dict_classes_weigths[cls] = min_v/dict_classes_weigths[cls]
    #order dict by key:
    sorted_dict = dict(sorted(dict_classes_weigths.items()))
    return torch.FloatTensor(list(sorted_dict.values())).cuda()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-kf', '--kfolds', type=int, help='Number of folds if CV', default=5)
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Training file with the dataset files (train.csv and test.csv)')
    parser.add_argument('-t', '--test', type=str,
                        help='Test file', default='ninguno')
    parser.add_argument('-r', '--data_root', type=str, required=True,
                        help='Root path with the images')
    parser.add_argument('-l', '--landmark_root', type=str,
                        help='Root path with the landmarks or saliencies')
    parser.add_argument('-imgSize', '--img_size', type=int,
                        help='Type of model to use, with input images of 48x48. Options:[48]',
                        default=48)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, help='value of learning rate', default=0.001)
    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size', default=32)
    parser.add_argument('-s', '--seed', type=int, help='Seed to feed random generators', default=2020)
    parser.add_argument('-logs', '--logs_folder', type=str, help='Path to save logs of training', default='./')
    parser.add_argument('-m','--modality', type=str, help='Choose the architecture of the model (baseline, original, landmarks or saliency)', default="original")
    parser.add_argument("--train",
                        type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Train the complete model after cross-validation. ")
    parser.add_argument('-tl', '--tl_preTrained_weights', type=str, required=False, default=None,
                        help='Path to the pre-trained weigths (.pt file)')

    args = parser.parse_args()

    print("PROCESSING MODALITY: ", args.modality)
    print("WEIGHTS: ", args.tl_preTrained_weights)

    #Prepare environment:
    os.environ["PYTHONWARNINGS"] = "ignore"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    seed_torch(seed=args.seed)

    # Create out folder for logs and models:
    now = datetime.datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")


    #Convert input images to expected nw size (48x48) or (100x100) and normalize values
    transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    # Create DS generator
    train_dataset = None
    if args.modality == "saliency":
        train_dataset= Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                              dataroot_land=args.landmark_root, transform = transformation)

        #Añado esto si no se mete nada de -t se usa lo de siempre, sino se usa el otro csv
        if args.test == "ninguno":
            test_dataset = Plain_Dataset_saliency(csv_path=args.data, dataroot=args.data_root,
                                                  dataroot_land=args.landmark_root, transform = transformation)
        else:
            test_dataset = Plain_Dataset_saliency(csv_path=args.test, dataroot=args.data_root,
                                                  dataroot_land=args.landmark_root, transform=transformation)

    #Train nw
    Train(args.epochs, args.kfolds,train_dataset, test_dataset, device, class_weigths=False,
          batch_size=args.batch_size, img_size=args.img_size, lr=args.learning_rate, num_workers=6,
          modality=args.modality, num_epochs_stop=30 , train_complete_model_flag=args.train, preTrainedWeigths=args.tl_preTrained_weights,
          save_path=os.path.join(args.logs_folder, current_time))
    #num_epochs_stop antes 30


