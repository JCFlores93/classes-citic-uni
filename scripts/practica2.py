# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:49:54 2018

@author: jean
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

data = pd.read_csv('caso.csv')
data = data.iloc[:,1:]
data.describe()
data.shape

columnsName = list(data.columns)
for columnName in columnsName:
    print('*========' + columnName +'========*')
    print(data[columnName].describe())
    
def num_missing(x):
    return sum(x.isnull())

#Aplicamos por columna:
print ("Valores perdidos por columna")
print (data.apply(num_missing, axis=0))

tipos = data.columns.to_series().groupby(data.dtypes).groups

from scipy.stats import mode
for column in ['antig_tc', 'linea_sf','deuda_sf']:
    data[column] = data[column].fillna(data[column].median())
    
#Create one-hot encoder
one_hot = LabelBinarizer()

#One-hot encode feature
data_casa = one_hot.fit_transform(data['casa'])

#View feature classes
columns_data_casa = one_hot.classes_.tolist()
   
data_casa = pd.DataFrame(
        data=data_casa,
        columns=columns_data_casa
        )

train = pd.concat([data, data_casa], axis=1)

#Eliminamos casa y casa_f
train = train.drop(['casa', 'casa_f'], axis=1)

#Tratamos la columna zona
#One-hot encode feature
data_zona = one_hot.fit_transform(data['zona'])

#View feature classes
columns_data_zona = one_hot.classes_.tolist()
   
data_zona = pd.DataFrame(
        data=data_zona,
        columns=columns_data_zona
        )

train = pd.concat([train, data_zona], axis=1)
train = train.drop('zona', axis=1)

#Tratando la columna nivel_educ

train['nivel_educ'].unique()
#Crear un mapper
scale_mapper = {
        "PROFESIONAL": 5,
        "TECNICO": 4,
        "SUPERIOR": 3,
        "EDUCACION BASICA": 2,
        "SIN EDUCACION": 1,
        }
train['nivel_educ'] = train['nivel_educ'].replace(scale_mapper)
len(train.columns)
train = train.drop('dias_lab', axis=1)
train = train.drop('empleo', axis=1)
#Generando variables x e y
x=train.drop('mora', axis=1)
y=train[['mora']]

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


