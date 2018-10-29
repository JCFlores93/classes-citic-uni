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


#Boxplots

#Descargando base 
iris = sns.load_dataset('iris')

#Análisis univariado
iris.boxplot()
iris.hist()

#Analisis bivariado
iris.boxplot(column='sepal_length', by='species')
sns.boxplot(x="sex", y="total_bill",hue="time", data=tips)


#Scatter 
#####################
plt.scatter(data['Population'], 
            data['GDP per Capita'])

plt.scatter(tips['total_bill'], 
            tips['tip'])

plt.scatter(tips['total_bill'], 
            tips['tip'], marker= 'x')

plt.scatter(tips['total_bill'], 
            tips['tip'], marker= 'x', alpha=0.5,
            s=100, color='green')


#cargando data 
feliz = pd.read_csv('happy2015.csv')
feliz.columns

#importando librerias 
from pandas.tools.plotting import scatter_matrix
scatter_matrix(feliz)

#subsetting
feliz.columns
scatter_matrix(feliz[['Happiness Score', 'Economy (GDP per Capita)']])

sub_feliz = feliz[['Happiness Score', 
                   'Economy (GDP per Capita)', 
                   'Trust (Government Corruption)',
                   'Generosity']]
scatter_matrix(sub_feliz)

iris.plot(kind="scatter", x="sepal_length", y="sepal_width")

g = sns.FacetGrid(iris, hue='species', size=5)
g.map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()


#Barplot
###############

feliz.plot.bar()

# Subsetting
feliz2 = feliz[:10][['Country', 'Happiness Rank', 
              'Trust (Government Corruption)']]

#Descargando base de datos 
titanic = sns.load_dataset('titanic')

sns.barplot(x="sex", y="survived", hue="class", data=titanic)
sns.barplot(x="sex", y="survived", hue="deck", data=titanic)

#Pie chart
feliz['Region'].value_counts()
plt.pie(feliz['Region'].value_counts(), 
        labels=feliz['Region'].value_counts().index)

#Line Chart
##############################################################
plt.plot(feliz['Happiness Score'])
feliz['Happiness Score'].plot(kind='line', marker='o')

# Funcion de densidad
##############################################################
x1 = np.random.normal(0,1,10000)
plt.hist(x1)

sns.distplot(x1, hist=True, kde=True, color='blue')
sns.distplot(x1, hist=False, kde=True, color='blue')
sns.distplot(x1, hist=True, kde=False, color='blue')

sns.distplot(x1, 
             hist=True, 
             kde=True, 
             color='blue', 
             hist_kws={'edgecolor':'black'})


