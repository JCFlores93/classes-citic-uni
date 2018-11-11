# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:12:47 2018

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

#Kernel PCA
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf')
x_train_kpca = kpca.fit_transform(x_train_sc)
x_test_kpca = kpca.transform(x_test_sc)

#Regresion logistica
#############################################################
from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(random_state=0)
logistic.fit(x_train, y_train)
y_est_train=logistic.predict(x_train)

#Matriz de confusion
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train, y_est_train)
(cm[0,0]+cm[1,1])/np.sum(cm)


#Regresion logistica
#############################################################
from sklearn.linear_model import LogisticRegression
logistic2=LogisticRegression(random_state=0)
logistic2.fit(x_train_kpca, y_train)
y_est_train_kpca=logistic2.predict(x_train_kpca)

#Matriz de confusion
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train, y_est_train_kpca)
(cm[0,0]+cm[1,1])/np.sum(cm)


