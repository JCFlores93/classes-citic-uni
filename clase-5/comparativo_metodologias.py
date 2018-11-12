# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:59:32 2018

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

#PCA - Componentes principales
from sklearn.decomposition import PCA
#Generando objetos
pca=PCA(n_components=1)
#Aprendiendo 
x_train_pca=pca.fit_transform(
        x_train_sc)
#Aplicando lo aprendido
x_test_pca=pca.transform(
        x_test_sc)

#LDA - Componentes principales
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#Aplicando LDA
lda = LDA(n_components=1)
x_train_lda = lda.fit_transform(x_train_sc, y_train)
x_test_lda = lda.transform(x_test)

#Kernel PCA - Componentes principales
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=1, kernel='rbf')
x_train_kpca = kpca.fit_transform(x_train_sc)
x_test_kpca = kpca.transform(x_test_sc)


#Regresion logistica
#############################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
logistic=LogisticRegression(random_state=0)

# 1) Variables originales
logistic.fit(x_train, y_train) #modificar aqui x
y_est_train=logistic.predict(x_train) #modificar aqui x,y
#Matriz de confusion
cm=confusion_matrix(y_train, y_est_train) #modificar y
prec_train=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

y_est_test = logistic.predict(x_test) #modificar x
cm=confusion_matrix(y_test, y_est_test) #modificar y
prec_test=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

#####################################################
# 2) Escalamiento
logistic.fit(x_train_sc, y_train) #modificar aqui x
y_est_train_sc=logistic.predict(x_train_sc) #modificar aqui x,y
#Matriz de confusion
cm=confusion_matrix(y_train, y_est_train_sc) #modificar y
prec_train_sc=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

y_est_test_sc = logistic.predict(x_test_sc) #modificar x
cm=confusion_matrix(y_test, y_est_test_sc) #modificar y
prec_test_sc=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

#####################################################
# 3) PCA
logistic.fit(x_train_pca, y_train) #modificar aqui x
y_est_train_pca=logistic.predict(x_train_pca) #modificar aqui x,y
#Matriz de confusion
cm=confusion_matrix(y_train, y_est_train_pca) #modificar y
prec_train_pca=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

y_est_test_pca = logistic.predict(x_test_pca) #modificar x
cm=confusion_matrix(y_test, y_est_test_pca) #modificar y
prec_test_pca=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

#####################################################
# 4) LDA
logistic.fit(x_train_lda, y_train) #modificar aqui x
y_est_train_lda=logistic.predict(x_train_lda) #modificar aqui x,y
#Matriz de confusion
cm=confusion_matrix(y_train, y_est_train_lda) #modificar y
prec_train_lda=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

y_est_test_lda = logistic.predict(x_test_lda) #modificar x
cm=confusion_matrix(y_test, y_est_test_lda) #modificar y
prec_test_lda=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

#####################################################
# 5) Kernel PCA
logistic.fit(x_train_kpca, y_train) #modificar aqui x
y_est_train_kpca=logistic.predict(x_train_kpca) #modificar aqui x,y
#Matriz de confusion
cm=confusion_matrix(y_train, y_est_train_kpca) #modificar y
prec_train_kpca=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre

y_est_test_kpca = logistic.predict(x_test_kpca) #modificar x
cm=confusion_matrix(y_test, y_est_test_kpca) #modificar y
prec_test_kpca=(cm[0,0]+cm[1,1])/np.sum(cm) #modificar nombre






