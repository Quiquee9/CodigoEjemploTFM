import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


plt.rcParams['figure.figsize']=(9,7)

#Threshold Test todos
num_frames_Training = [2, 3, 80, 412, 1458, 5048] #num frames por emocion
num_frames_Training = np.multiply(num_frames_Training,8) #8 emociones
num_frames_Training=np.append(num_frames_Training,154108)
accuracy_video = [23.70, 26.33, 37.82, 43.55, 47.70, 53.12, 54.27]
accuracy_frame = [21.70, 22.33, 32.45, 35.48, 41.28, 44.66, 46.78]
thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(thresholds, num_frames_Training, color="black",ax=ax1, label='Número de frames')
sns.lineplot(thresholds, accuracy_video, color="red", ax=ax2, label='Tasa de acierto (%) vídeo')
sns.lineplot(thresholds, accuracy_frame, color="orange", ax=ax2, label='Tasa de acierto (%) frame')

ax1.set_xlabel('Umbral fijo')
ax1.set_ylabel('Número de frames entrenamiento')
ax2.set_ylabel('Tasa de acierto (%)')
plt.title("Tasa de acierto (%) por umbral (Test todos)")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_ylim(ax1.get_xlim()[0],180000)
ax2.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.tight_layout()
ax2.grid(True)
plt.grid(linewidth=0.5,axis='both')
plt.show()

#Threshold Test threshold
accuracy_video = [21.56, 24.35, 34.67, 42.27, 47.76, 53.12, 54.27]
accuracy_frame = [42.71, 36.17, 37.72, 36.40, 41.29, 44.66, 46.78]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(thresholds, num_frames_Training, color="black",ax=ax1, label='Número de frames')
sns.lineplot(thresholds, accuracy_video, color="red", ax=ax2, label='Tasa de acierto (%) vídeo')
sns.lineplot(thresholds, accuracy_frame, color="orange", ax=ax2, label='Tasa de acierto (%) frame')

ax1.set_xlabel('Umbral fijo')
ax1.set_ylabel('Número de frames entrenamiento')
ax2.set_ylabel('Tasa de acierto (%)')
plt.title("Tasa de acierto (%) por umbral (Test filtrado)")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_ylim(ax1.get_xlim()[0],180000)
ax2.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.tight_layout()
ax2.grid(True)
plt.grid(linewidth=0.5,axis='both')
plt.show()

#Quartiles
num_frames_Training = [154108,149677,145753,138187,130541,123019,115418,99982,88478,77426,61921,39081,31377,16075] #num frames por emocion
accuracy_video = [54.27, 54.60, 53.75, 53.73, 54.40, 53.90, 55.05, 56.62,56.57, 55.03, 53.47, 54.25, 52.05, 53.01]
accuracy_frame = [46.78, 46.88, 46.21, 46.06, 46.91, 46.35, 47.00, 47.47,46.91, 46.50, 45.79, 44.82, 43.88, 43.96]
quartiles = [0, 2.5, 5, 10, 15, 20, 25, 35,42.5, 50, 60, 75, 80, 90]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(quartiles, num_frames_Training,marker="o",color="black",ax=ax1, label='Número de frames')
sns.lineplot(quartiles, accuracy_video, marker="o",color="red", ax=ax2, label='Tasa de acierto (%) vídeo')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="orange", ax=ax2, label='Tasa de acierto (%) frame')

sns.lineplot(quartiles, np.tile(54.27,len(accuracy_frame)), color="gray", ax=ax2, label='Tasa de acierto (%) vídeo Baseline')

ax1.set_xlabel('Percentil umbral')
ax1.set_ylabel('Número de frames de entrenamiento')
ax2.set_ylabel('Tasa de acierto (%)')
plt.title("Tasa de acierto (%) por umbrales con percentiles")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_ylim(ax1.get_xlim()[0],180000)
ax2.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.xticks(np.arange(min(quartiles), max(quartiles)+5, 5))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.show()

#Modelo iterativo percentiles filtrado con TL3 iterativos
accuracy_video = [53.01, 54.73, 55.87, 53.35, 54.80, 55.73, 53.97, 56.00, 55.75, 54.27 ]
accuracy_frame = [43.96,  44.69,  46.19,  45.60,  46.12,  46.84,  46.61,  48.19,  47.70,  46.78  ]
quartiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quartiles = list(reversed(quartiles))

fig, ax1 = plt.subplots()
sns.lineplot(quartiles, accuracy_video, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame')

sns.lineplot(quartiles, np.tile(54.27,len(accuracy_frame)), color="gray", ax=ax1, label='Tasa de acierto (%) vídeo Baseline')

ax1.set_xlabel('Percentil umbral')
ax1.set_ylabel('Tasa de acierto (%)')
plt.title("Tasa de acierto (%) modelo de filtrado iterativo")
ax1.legend(loc="upper left")
ax1.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.xticks(np.arange(min(quartiles), max(quartiles)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

#Modelo iterativo percentiles filtrado y pesos iterativos
accuracy_video = [53.01, 54.27, 54.05, 56.87, 55.78, 56.25, 56.97, 56.81, 55.68, 53.02]
accuracy_frame = [43.96,44.66,  45.48, 46.50,  46.95,  47.31,  47.60,  47.47,  47.18,  46]
quartiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quartiles = list(reversed(quartiles))

fig, ax1 = plt.subplots()
sns.lineplot(quartiles, accuracy_video, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame')

sns.lineplot(quartiles, np.tile(54.27,len(accuracy_frame)), color="gray", ax=ax1, label='Tasa de acierto (%) vídeo Baseline')

ax1.set_xlabel('Percentil umbral')
ax1.set_ylabel('Tasa de acierto (%)')
plt.title("Tasa de acierto (%) modelo de filtrado y transferencia de pesos iterativos")
ax1.legend(loc="upper left")
ax1.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.xticks(np.arange(min(quartiles), max(quartiles)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

#Modelo iterativo etapas quartiles comparacion con el otro
accuracy_video = [53.01, 54.27, 54.05, 56.87, 55.78, 56.25, 56.97, 56.81, 55.68, 53.02]
accuracy_frame = [43.96,44.66,  45.48, 46.50,  46.95,  47.31,  47.60,  47.47,  47.18,  46]
quartiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quartiles = list(reversed(quartiles))

fig, ax1 = plt.subplots()
sns.lineplot(quartiles, accuracy_video, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo - Modelo de filtrado y transferencia de pesos iterativos')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="yellow", ax=ax1, label='Tasa de acierto (%) frame - Modelo de filtrado y transferencia de pesos iterativos')

accuracy_video = [54.27, 54.60, 53.95, 53.73, 54.40, 53.90, 55.05, 56.62,56.57, 55.03, 53.47, 54.25, 52.05, 53.01]
accuracy_frame = [46.78, 46.88, 46.21, 46.06, 46.91, 46.35, 47.00, 47.47,46.91, 46.50, 45.79, 44.82, 43.88, 43.96]
quartiles = [0, 2.5, 5, 10, 15, 20, 25, 35,42.5, 50, 60, 75, 80, 90]

sns.lineplot(quartiles, accuracy_video, marker="o",color="darkred", ax=ax1, label='Tasa de acierto (%) vídeo - Modelo con frames filtrados por percentiles')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame - Modelo con frames filtrados por percentiles')
sns.lineplot(quartiles, np.tile(54.27,len(accuracy_frame)), color="gray", ax=ax1, label='Tasa de acierto (%) vídeo Baseline')

ax1.set_xlabel('Percentil umbral')
ax1.set_ylabel('Tasa de acierto (%)')
plt.title("Comparación modelos iterativos con modelos con frames filtrados por percentiles")
ax1.legend(loc="upper left")
ax1.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.xticks(np.arange(min(quartiles), max(quartiles)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.show()

#Comparacion percentiles TLF y TL3 v1
accuracy_videoTLF = [53.01, 54.27, 54.05, 56.87, 55.78, 56.25, 56.97, 56.81, 55.68, 53.02]
accuracy_frameTLF = [43.96,44.66,  45.48, 46.50,  46.95,  47.31,  47.60,  47.47,  47.18,  46]
quartiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quartiles = list(reversed(quartiles))

percentRepTLF = [26.72, 60.71, 71.02, 74.74, 81.18, 83.89, 85.68, 88.11, 89.67]
quartilesPercen= list(reversed([0, 10, 20, 30, 40, 50, 60, 70, 80]))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(quartiles, accuracy_videoTLF, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo TLF')
sns.lineplot(quartiles, accuracy_frameTLF, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame TLF')
sns.lineplot(quartilesPercen, percentRepTLF, marker="o",color="green", ax=ax2, label='Porcentaje de datos de entrenamiento TLF')

accuracy_videoTL3 = [53.01, 54.73, 55.87, 53.35, 54.80, 55.73, 53.97, 56.00, 55.75, 54.27]
accuracy_frameTL3 = [43.96,44.69, 46.19,  45.60,  46.12,  46.84,  46.61,  48.19,  47.70,  46.78]
percentRepTLF = [26.72, 51.44, 64.75, 71.39, 76.16, 79.91, 82.78, 86.22, 89.67]

sns.lineplot(quartiles, accuracy_videoTL3, marker="o",color="darksalmon", ax=ax1, label='Tasa de acierto (%) vídeo TL3')
sns.lineplot(quartiles, accuracy_frameTL3, marker="o",color="yellow", ax=ax1, label='Tasa de acierto (%) frame TL3')
sns.lineplot(quartilesPercen, percentRepTLF, marker="o",color="yellowgreen", ax=ax2, label='Porcentaje de datos de entrenamiento TL3')

ax1.set_xlabel('Quartile')
ax1.set_ylabel('Tasa de acierto (%)')
ax2.set_ylabel('Datos repetidos con etapa anterior (%)')
plt.title("Modelo Iterativo TL3 vs TLF")
ax1.legend(loc="upper left")
ax2.legend(loc="lower right")

plt.xticks(np.arange(min(quartiles), max(quartiles)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

#Comparacion percentiles TLF y TL3 v2
accuracy_videoTLF = [53.01, 54.27, 54.05, 56.87, 55.78, 56.25, 56.97, 56.81, 55.68, 53.02]
accuracy_frameTLF = [43.96,44.66,  45.48, 46.50,  46.95,  47.31,  47.60,  47.47,  47.18,  46]
quartiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quartiles = list(reversed(quartiles))

percentRepTLF = [42.27, 90.44, 93.71, 93.03, 96.65, 97.46, 97.88, 98.96, 100]
quartilesPercen= list(reversed([0, 10, 20, 30, 40, 50, 60, 70, 80]))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(quartiles, accuracy_videoTLF, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo TLF')
sns.lineplot(quartiles, accuracy_frameTLF, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame TLF')
sns.lineplot(quartilesPercen, percentRepTLF, marker="o",color="green", ax=ax2, label='Porcentaje de datos de entrenamiento TLF')

accuracy_videoTL3 = [53.01, 54.73, 55.87, 53.35, 54.80, 55.73, 53.97, 56.00, 55.75, 54.27]
accuracy_frameTL3 = [43.96,44.69, 46.19,  45.60,  46.12,  46.84,  46.61,  48.19,  47.70,  46.78]
percentRepTLF = [42.27, 75.75, 86.24, 88.79, 90.83, 92.99, 94.58, 96.84, 100]

sns.lineplot(quartiles, accuracy_videoTL3, marker="o",color="darksalmon", ax=ax1, label='Tasa de acierto (%) vídeo TL3')
sns.lineplot(quartiles, accuracy_frameTL3, marker="o",color="yellow", ax=ax1, label='Tasa de acierto (%) frame TL3')
sns.lineplot(quartilesPercen, percentRepTLF, marker="o",color="yellowgreen", ax=ax2, label='Porcentaje de datos de entrenamiento TL3')

ax1.set_xlabel('Quartile')
ax1.set_ylabel('Tasa de acierto (%)')
ax2.set_ylabel('Datos repetidos con etapa anterior (%)')
plt.title("Modelo Iterativo TL3 vs TLF")
ax1.legend(loc="upper left")
ax2.legend(loc="lower right")

plt.xticks(np.arange(min(quartiles), max(quartiles)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()

#Results regularization
plt.rcParams['figure.figsize']=(9,9)
# assign data of lists.
dict = {'Tasa de acierto (%) a nivel de frame': [47.47,50.56,48.31,48.46,48.43,49.38,48.57],
        'Tasa de acierto (%) a nivel de vídeo': [56.62,60.08, 55.35, 56.87, 56.25, 58.08,57.35],
        'Regularización': ['Sin regularización extra','Dropout=0.5 |λ=0.001','Dropout=0.5 |λ=0.01','Dropout=0.5 |λ=0.0001',
                           'Dropout=0.75|λ=0.001','Dropout=0.3 |λ=0.001','Dropout=0.4 |λ=0.001']}
# Create DataFrame
results = pd.DataFrame(dict)
ax = sns.scatterplot(x="Tasa de acierto (%) a nivel de frame", y="Tasa de acierto (%) a nivel de vídeo", data=results,hue="Regularización")
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.03, point['y'], str(point['val']),fontsize=10)
label_point(results['Tasa de acierto (%) a nivel de frame'], results['Tasa de acierto (%) a nivel de vídeo'], results['Regularización'], plt.gca())
plt.grid(linewidth=0.5)
plt.tight_layout()
plt.show()

plt.rcParams['figure.figsize']=(9,7)

#Modelo iterativo etapas quartiles con regularizacion
"""
accuracy_video = [57.88, 57.40, 59.48, 58.53, 57.60, 57.15]
accuracy_frame = [47.90,  48.10,  49.35,  49.89,  49.48,  49.28]
quartiles = [0, 20, 30, 40, 60, 80]
quartiles = list(reversed(quartiles))

fig, ax1 = plt.subplots()
sns.lineplot(quartiles, accuracy_video, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo Etapas con reg')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame Etapas con reg')

sns.lineplot(quartiles, np.tile(54.27,len(accuracy_frame)), color="gray", ax=ax1, label='Tasa de acierto (%) vídeo Baseline')

ax1.set_xlabel('Percentil umbral')
ax1.set_ylabel('Tasa de acierto (%)')
plt.title("Accuracy modelo iterativo por percentiles con y sin regularizacion")
accuracy_video = [53.01, 54.27, 54.05, 56.87, 55.78, 56.25, 56.97, 56.81, 55.68, 53.02]
accuracy_frame = [43.96,44.66,  45.48, 46.50,  46.95,  47.31,  47.60,  47.47,  47.18,  46]
quartiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quartiles = list(reversed(quartiles))
sns.lineplot(quartiles, accuracy_video, marker="o",color="green", ax=ax1, label='Tasa de acierto (%) vídeo Etapas sin reg')
sns.lineplot(quartiles, accuracy_frame, marker="o",color="yellow", ax=ax1, label='Tasa de acierto (%) frame Etapas sin reg')

ax1.legend(loc="upper left")
ax1.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1]+2)
plt.xticks(np.arange(min(quartiles), max(quartiles)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.gca().invert_xaxis()
plt.show()
"""
#Modelo q35 por reshold 0.001
accuracy_video = [59.4,60.88,61.67, 61.60,60.63,60.78,60.85]
accuracy_frame = [51.05,51.45,51.27, 51.34,50.97,51.14,51.30]
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2]
percent = [58.09,70.68, 77.82, 82.82, 86.69, 89.82, 94.97]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
sns.lineplot(thresholds, accuracy_video, marker="o",color="red", ax=ax1, label='Tasa de acierto (%) vídeo')
sns.lineplot(thresholds, accuracy_frame, marker="o",color="orange", ax=ax1, label='Tasa de acierto (%) frame')
sns.lineplot(thresholds, percent, marker="o",color="green", ax=ax2, label='Porcentaje de datos de entrenamiento')

sns.lineplot(thresholds, np.tile(60.08,len(accuracy_frame)), color="black", ax=ax1, label='Tasa de acierto (%) vídeo iteración 1 (percentil 35)')
sns.lineplot(thresholds, np.tile(54.27,len(accuracy_frame)), color="gray", ax=ax1, label='Tasa de acierto (%) vídeo Baseline')

ax1.set_xlabel('Umbral fijo')
ax1.set_ylabel('Tasa de acierto (%)')
ax2.set_ylabel('Datos (%)')
plt.title("Tasa de acierto de modelos de filtrado iterativo de dos etapas mixto partiendo percentil 35")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

#plt.xticks(np.arange(min(thresholds), max(thresholds)+10, 10))
plt.tight_layout()
#ax1.grid(True)
plt.grid(linewidth=0.5)
plt.gca().invert_xaxis()
ax1.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1]+2)
plt.show()
