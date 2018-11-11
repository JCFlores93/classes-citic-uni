# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:54:26 2018

@author: CTIC - UNI
"""

#Importando librerías 
import numpy as np 
import pandas as pd

#Importando dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#Generando variables x e y
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#Separando base en train y test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Escalamiento
from sklearn.preprocessing import StandardScaler
#Generando objeto
sc=StandardScaler()
#Aprendiendo y transformando
x_train_sc = sc.fit_transform(x_train)
#Aplico lo aprendido: Transformando
x_test_sc = sc.transform(x_test)

#Viendo objetos aprendidos
sc.mean_
sc.var_

#Discriminante lineal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Contar categorias de variables objetivo
dataset.columns
dataset['Purchased'].value_counts()

#Aplicando LDA
lda = LDA(n_components=1)
x_train_lda = lda.fit_transform(x_train_sc, y_train)
x_test_lda = lda.transform(x_test)
coeficientes=lda.coef_