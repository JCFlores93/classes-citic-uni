# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:02:22 2018

@author: CTIC - UNI
"""

#Importando librerias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

#Importando data
data = pd.read_csv('countryGDP.csv')
data

#Eliminando missings 
data1 = data['GDP per Capita'].dropna()

#Histograma
plt.hist(data1)

# 10 000 = la cantidad de datos 
b = np.arange(data1.min(), data1.max(),10000)

#Gráfico personalizado, separaciones
plt.hist(data1, bins=b)
#Transparencia
plt.hist(data1, bins=b, alpha=0.5)
#Color
plt.hist(data1, bins=b, alpha=0.5, color='green')
plt.title('PBI Nacional')
plt.xlabel('PBI en dólares')
plt.ylabel('Frecuencia')
plt.show

plt.hist(data1, bins=b, alpha=0.5, color='green',
         orientation='horizontal')

plt.hist(data1, bins=b, alpha=0.5,color='green',
         histtype='step')

#Importando libreria
import seaborn as sns
#Cargando base de datos
tips = sns.load_dataset('tips')

#Analisis bivariado
g=sns.FacetGrid(tips, row='smoker', col='time')
g.map(plt.hist, "total_bill")

g=sns.FacetGrid(tips, row='day', col='time')
g.map(plt.hist, "total_bill")

g=sns.FacetGrid(tips, row='day', col='sex')
g.map(plt.hist, "total_bill")

g=sns.FacetGrid(tips, row='time', col='sex')
g.map(plt.hist, "total_bill")

g=sns.FacetGrid(tips, row='time', col='sex')
g.map(plt.hist, "tip")


