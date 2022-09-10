# TFM Enrique Díez Benito

Repositorio con el código de Enrique Díez Benito del TFM "Diseño e implementación de modelos computacionales basados en redes neuronales profundas para el reconocimiento de emociones en contenidos multimedia".



## 1. Introducción
Todo el código está basado a partir del github de 
[Guided-EMO-SpatialTransformer](https://github.com/cristinalunaj/Guided-EMO-SpatialTransformer)

Se incluye todo el código y modelos que se han usado para hacer el TFM, sin modificación alguna. El directorio raíz que se uso es "/home/enriquediez/Guided-EMO-SpatialTransformer-main", por lo que para usar en otros entornos hay que cambiar esa url de todos los ficheros del código.

Los paquetes que he usado son los del /requirements.txt

### 1.1 Entrenamiento
Para el entrenamiento se ha usado la versión modificada src/main_video_CV.py que incluye entre otras cosas el tiempo de ejecución de cada fold
así como el tiempo miedo. Un ejemplo de su entrenamiento:

    python main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv 
    -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 
    -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquique -m saliency 
    --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_STN_7classes/20210913_144129/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-153-original.pt
Para guardar los logs uso este comando modificado que crea un txt:

    python -u main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdQuartile/labelsEmotion_Quartile_35.csv 
    -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale 
    -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquiqueQuartile -m saliency 
    --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt| tee logq35.txt

Los txt con los logs los guardo dentro de la carpeta de los logs del modelo en un fichero llamado log.txt, por ejemplo data/RAVDESSquique/20211109_103717_SinTL/log.txt
Además, he modificado el codigo para que con -d se indique el csv de entrenamiento y -t el de validacion, si no se usa el -t los datos de entrenamiento y validacion son los mismos (manteniendo los actores de fold, obviamente). Esto es importante porque para los modelos con filtrado a -d entran el csv con frames filtrados y a -t el csv con todos los fotogramas.

El modelo con mayor regularización se entrena en src/main_video_CV_regularization.py con la modificacion del dropout en src/architectures/deep_emotion_saliencyQuiqueRegulariz.py.
### 1.2 Evaluación

Para la evaluación con AUC uso src/evaluate_STQuique.py y la modificación con Z-Score src/evaluate_STQuiqueConZScore.py.
Están modificadas para que el log de más información y creen las matrices de confusión etc.
Un ejemplo:

    -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv
    -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale
    -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale
    -m saliency
    -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/evaluateQuique/modelosThresholdReg35lr0.001/th07Final
    --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modeloThresholdReg35/20220216_221409th07/trained_models
    
Todos los resultados se guardan dentro de una carpeta en data/evaluateQuique

## 2. Entrenamiento de modelos
Explicaré un poco por encima como he entrenado todos los modelos de TFM y cuáles son los .py para crear el filtrado o lo que sea. Además indicaré dónde se han guardado los modelos (sus pesos y logs) y sus evaluaciones (matrices de confusión, resultados, ...)

### 2.1 Modelos entrenados con todos los fotogramas
Estos modelos son los que ya teniamos de antes. Para entrenarlos se puede usar:
    
    python main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquique -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_STN_7classes/20210913_144129/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-153-original.pt

En este caso es el de TL con el modelo de 7 clases.
Para obtener el porcentaje de acierto del modelo solo entrenado con AffectNet:




## Download datasets
For reproducing the experiments, firt you need to download the dataset used for the experiments: 

- [x] [FER-2013](https://www.kaggle.com/msambare/fer2013)
- [x] [AffectNet](http://mohammadmahoor.com/affectnet/)


Once downloaded, put them in your working directory, in what follows, we will refer to these directories as: 

* \<FER2013-dir> : Directory with the images of FER-2013 
* \<AffectNet-dir>: Directory with the images of AffectNet

## Prepare dataset (5 CV)
For replicating our results, we have uploaded the used dataset distribution to: data/datasets_distribution
In case you want to extract your own distributions, you can run the following code: 

    python3 src/DSpreparation/AffectNet_ds_processing.py -rt <AffectNet-dir>/Manually_Annotated_file_lists/training.csv
 
## Documentation

[Documentation](https://linktodocumentation)


## Installation

Install my-project with npm


    npm install my-project
AffectNet:

    python3 src/ImagePreprocessing/LandmarkExtractor.py -o <AffectNet-dir>/LANDMARKS_dlib_MTCNN -olandm <AffectNet-dir>/AFFECTNET/LANDMARKS_dlib_MTCNN_npy
    -d <AffectNet-dir>/Manually_Annotated_compressed/Manually_Annotated_Images -trainCSV ~/Guided-EMO-SpatialTransformer/data/datasets_distribution/AffectNet/polarity_complete_5folds.csv
    -trainCSVSep ; -ds AffectNet -logs <AffectNet-dir>/AFFECTNET

