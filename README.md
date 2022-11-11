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
Explicaré un poco por encima cómo he entrenado todos los modelos del TFM y cuáles son los .py para crear el filtrado y la metodología seguida. Además, indicaré dónde se han guardado los modelos (sus pesos y logs) y sus evaluaciones (matrices de confusión, resultados, ...)

### 2.1 Modelos entrenados con todos los fotogramas
Estos modelos son los que ya teniamos de antes. Para entrenarlos se puede usar:
    
    python main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquique -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_STN_7classes/20210913_144129/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-153-original.pt

En este caso es el de TL con el modelo de 7 clases.

Para obtener el porcentaje de acierto del modelo solo entrenado con AffectNet de 7 clases uso el `src/embeddingsExtractor/featureExtractor_ST.py` con los siguientes parámetros:
		
		-kf 2 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -bs 128 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/ -m saliency -tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_STN_7classes/20210913_144129/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-153-original.pt -nClasses 7 -emb2extract posteriors
Con los posteriors creados en `EstudioDePosteriors/posteriorsRADVESS7/df_embs_posteriors.csv` se usa `EstudioDePosteriors/eval7classes.py` para evaluar (`EstudioDePosteriors/eval7classes.py` es una modificacion con una normalización, que me comento Cristina). En esa carpeta se encuentran los .py para evaluar los de 3, hacer plot de los posteriors de 7 clases y posteriors de 8. En `EstudioDePosteriors/posteriorsTrain8` se encuentran los posteriors de entrenamiento del modelo con todos los frames y del de q25 de 8 clases. En esta carpeta están todos los plots también de los posterior.

Los pesos de los modelos se guardan en `data/RAVDESSquique` y los resultados en `data/evaluateQuique/modelosCompletos`.
### 2.2 Modelos filtrados por percentiles y umbrales fijos
Para crear el .csv con los frames filtrados por umbrales fijos uso `newDatasetThreshold.py` que crea los csv para una serie de umbrales fijos para entrenamiento y test. `newDatasetThresholdBalance.py` hace lo mismo pero balanceando las clases, los csv se guardan en `data/datasets_distribution/RADVESSthresholdTRAIN` y `data/datasets_distribution/RADVESSthresholdTEST` para cada umbral. Como se parte de los posterior de 7 de AffectNet no hace falta dividir estos datos por fold. Para el caso de percentiles se crean los csv con `newDatasetQuartiles.py` y se guardan los csv por percentil en `data/datasets_distribution/RADVESSthresholdQuartile`.
Para entrenar los modelos se usan esos csv, dependiendo del modelo, por ejemplo con train y test de threshold de 0.3:

	python main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdTRAIN/labelsEmotion_Threshold_0_3.csv -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdTEST/labelsEmotion_Threshold_0_3.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquiqueTh -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt > log03.txt
	
O con umbrales, usando todos los del test (hay que definir -d y -t también):
	
	python -u main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdTRAIN/labelsEmotion_Threshold_0_1.csv -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquiqueTh -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt| tee log01.txt

En el caso de percentiles:

	python -u main_video_CV.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/RADVESSthresholdQuartile/labelsEmotion_Quartile_25.csv -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/RAVDESSquiqueQuartile -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt| tee logq25b.txt
	
Los modelos se guardan en las carpetas en `data/RAVDESSquiqueQuartile` y `data/RAVDESSquiqueTh` y los resultados en `data/evaluateQuique/modelosQuartile` y `data/evaluateQuique/modelosTh`.

### 2.3 Modelos iterativos y modelo iterativo de dos etapas mixto
Para los modelos iterativos se entrenan en varios pasos, para los modelos de TL3 (transfer learning de 3 clases para todos):

Primero, se necesitan obtener el filtrado con modelo de la iteracion previa. Hay dos .py que hacen más o menos lo mismo. Con `modeloRecursivo/creadorPosteriors.py` se crea el nuevo filtrado con varios parámetros ya sea cuartiles o threshold a partir de un modelo (uno por fold). Por ejemplo, `modeloRecursivo/ModeloQ25` o `Q35` tienen por fold los csv de los frames filtrados por percentiles, threshold o lo que se indique. Con este .py también se pueden obtener los posteriors de 8 clases si se descomenta la línea 100. En el caso de usar etapas en los modelos iterativos uso `modeloRecursivo/creadorPosteriorsEtapas.py` con los párametros de ejemplo:
		
		-d /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -m saliency -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/csvModelosEtapasQuartileTL3 --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modelosEtapasTL3/20220306_124934_q20/trained_models

En --tl se pone el modelo de la iteración previa para obtener los csv filtrados para la nueva iteración. Para indicar con qué percentil se filtra hay que cambiar la línea 92 de `modeloRecursivo/creadorPosteriorsEtapas.py`  para indicar el nuevo percentil, en el ejemplo anterior se pondría 10, ya que la anterior etapa de era de q20. Esto crea los csv en la carpeta indicada en logs que es `modeloRecursivo/csvModelosEtapasQuartileTL3`. Así se hace para todas las iteraciones. El primer modelo iterativo se entrena como los modelos filtrados de una única iteración, comentados anteriormente, después se usa `creadorPosteriorsEtapas.py`  según lo indicado y, ahora, se entrena la nueva iteración con `modeloRecursivo/entrenamientoRecursivo.py`. En este .py se indica la carpeta con los csv en -d y además hay que poner el percentil a mano modificando el nombre del fichero de la línea 73, en este caso sería poner percentil 10.
	
	python -u entrenamientoRecursivo.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/csvModelosEtapasQuartileTL3 -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modelosEtapasTL3 -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/models_logs/AFFECTNET_LOGS_cd/saliency_20210611_131052/trained_models/TMP-deep_emotion-500-128-0.001-COMPLETE-48-252-saliency.pt | tee logq10_TL3etapa8.txt

En este comando hay que indicar también la carpeta donde se guarda los modelos recursivos en este caso `modeloRecursivo/modelosEtapasTL3`.

Para el caso de usar TLF, es decir modelos de transferencia de pesos y filtrado iterativos hay que usar de igual forma `creadorPosteriorsEtapas.py`, pero ahora el entrenamiento se hace con `modeloRecursivo/entrenamientoRecursivoModeloPorFold.py`. Es muy parecido al otro, también hay que cambiar la línea 74 indicando el percentil de la nueva etapa, se indica con -d la carpeta con los csv y ahora en --tl hay que indicar los pesos del modelo previo. Por ejemplo:

	python -u entrenamientoRecursivoModeloPorFold.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/csvModelosEtapasQuartile -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modelosEtapasQuartile -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modelosEtapasQuartile/20211211_145801_q70/trained_models | tee logq70-q60-TLF.txt

En la siguiente etapa (tras filtrar con q50 con `creadorPosteriorsEtapas.py` y modificar el `entrenamientoRecursivoModeloPorFold.py` para que use el q50):

	python -u entrenamientoRecursivoModeloPorFold.py -kf 5 -d /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/csvModelosEtapasQuartile -t /home/enriquediez/Guided-EMO-SpatialTransformer-main/data/datasets_distribution/AV_Speech_labels_emotionCOMPLETE_numeric_PNG.csv -r /mnt/RESOURCES/enriquediez/RAVDESS/AV_VIDEOS_FRAMES_48x48_grayscale -l /mnt/RESOURCES/enriquediez/RAVDESS/SALIENCY_48x48_grayscale -imgSize 48 -e 500 -lr 0.001 -bs 128 -s 2020 -logs /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modelosEtapasQuartile -m saliency --tl /home/enriquediez/Guided-EMO-SpatialTransformer-main/modeloRecursivo/modelosEtapasQuartile/20211211_175950_q60/trained_models | tee logq60-q50-TLF.txt

En esta carpeta de `modeloRecursivo` se encuentran los csv con los fotogramas filtrados por etapas y los pesos de modelos por etapas de percentiles usando TL3 (parten todos del mismo peso de 3 clases de AffectNet) o TLF (se van transferiendo los pesos). Están todos los modelos con modificaciones indicados en el word con los resultados. Además, cuando se incluye REG en los ficheros es que son los mismos ficheros pero hechos con la mejor regularización. Los resultados estan en `data/evaluateQuique` en `modeloPorEtapasQuartile`, ...

En el caso de los modelos de fltrado de dos etapas mixto se hace una carpeta con los modelos filtrados usando `creadorPosteriors.py` usando el modelo de q35 con mejor regularización, luego se usa el entrenamiento  de `modeloRecursivo/entrenamientoRecursivoModeloPorFoldREG.py` modificando el .py para que use el umbral que toque e indicando en -d la carpeta de los .csv y en --tl el modelo de q35. Los modelos estan en `modeloRecursivo/modeloThresholdReg35`, los importantes son con los de th en minuscula que son los que se hicieron con lr bien. Y los resultados están en `data/evaluateQuique/modelosThresholdReg35lr0.001`.
### 2.4 Regularización y menos fps
Para los de regularización ya los he comentado son modificaciones en los py de entrenamiento donde se añade al nombre de fichero `REG`. Para reducir los FPS en la línea 88 de `src/main_video_CV_regularization.py` hay que descomentar la línea y cambiar el 2 (15 fps)  de `#train_idx=train_idx[range(0, len(train_idx), 2)]` por 3 (10 fps), 4, ...

Los modelos están en `data/pruebaRegulariz`, `data/regFPSTodosFrame` y `data/regq35bajandoFPS` (submuestrea a partir de los datos filtrados de q35). Los resultados están en `data/evaluateQuique/modelosRegularizacionBaseline`, `data/evaluateQuique/pruebasBajandoFPSTodosFrame` y `data/evaluateQuique/pruebasBajandoFPSq35`.
### 2.5 Modelos con AU
Para el de AU45, el filtrado se hace con `newDatasetThreholdAUs.py` y los MVN con `newDatasetGMM_AUs.py` o `newDatasetGMM_AUsPerVideo.py`, si el percentil se calcula con todos los fotogramas o solo por video. En la carpeta `EstudioAUs` se encuentra el código que crea los modelos MVN (que usa el filtrado) y genera gráficas.  Los csv que se generan tras los filtrados se guardan en `data/datasets_distribution` en `AUth` (AU45), `GMM` (usando percentiles de distancia con todos los fotogramas), `GMM_perVideo` (usando percentiles de distancia por video, en este caso datos para train) y `GMM_perVideoTest` (datos para usar en test).

Los datos del modelo de AU45 está en `data/filtAU` por umbral y evaluado en `data\evaluateQuique\filtAU` (las que tienen una b antes son las correctas). Los otros tres modelos se encuentran con los nombres `GMM_models` , `GMM_models_perVideo` y `GMM_models_perVideoTest`.
