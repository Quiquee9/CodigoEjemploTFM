# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import numpy as np
from scipy import stats


results = pd.read_csv("./data/intervalosDeConfianzaModelos.csv",sep=";",header=None)
results = results.iloc[::-1]
results.columns = ['Modelo', "Tasa de acierto (%)", "ROC-AUC", "Lower", "Upper", "label"]
plt.rcParams['figure.figsize']=(9,5)

#plt.xlim(62,76)
#plt.ylim(0.66,0.81)
flag=0
flag1=0
flag2=0

fig, ax = plt.subplots()
#For confidence intervals
for lower, upper,meanMAE, y, label in zip(results['Lower'], results['Upper'], results['Tasa de acierto (%)'], range(len(results)),results['label']):
    y=y/2
    if label=="Mio":
        flag=flag+1
        if flag==1:
            plt.plot((lower, upper), (y, y), 'o-',color='purple',label="Sin regularización añadida")
        else:
            plt.plot((lower, upper), (y, y), 'o-',color='purple')
    if label=="Reg":
        flag1=flag1+1
        if flag1==1:
            plt.plot((lower, upper), (y, y), 'o-',color='magenta',label="Con mejor regularización")
        else:
            plt.plot((lower, upper), (y, y), 'o-',color='magenta')
    if label=="otros":
        flag2=flag2+1
        if flag2==1:
            plt.plot((lower, upper), (y, y), 'o-',color='deeppink',label="Otros trabajos")
        else:
            plt.plot((lower, upper), (y, y), 'o-',color='deeppink')
    plt.plot(meanMAE,y,'|',color='black')
plt.yticks([x/2 for x in range(len(results))], list([x.replace('\r','') for x in results["Modelo"]]))
plt.title("Intervalos de confianza al 95% de la tasa de acierto (%) a nivel de vídeo")
plt.xlabel("Tasa de acierto (%)")
plt.ylabel("Modelos")
plt.subplots_adjust(left=0.33)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), title='Tipos de modelos: ')
plt.show()