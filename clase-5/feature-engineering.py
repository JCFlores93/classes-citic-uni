# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:30:29 2018

@author: CTIC - UNI
"""

#Importando librerías
import numpy as np 
import pandas as pd

#Importando data
dataset = pd.read_csv('Wine.csv')

#Vista rápida
dataset['Alcohol'].describe()
type(dataset.columns)

for x in dataset.columns:
    type(x)
    dataset[x].describe()
    print(x)
data_columns = dataset.columns

#Generando variables
x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

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

#Aplicando componentes principales
from sklearn.decomposition import PCA
#Generando objetos
pca=PCA(n_components=13)
#Aprendiendo 
x_train_pca=pca.fit_transform(
        x_train_sc)
#Aplicando lo aprendido
x_test_pca=pca.transform(
        x_test_sc)

#Calculando coeficientes de los componentes
componentes = pca.components_
#Verificacion
verificacion = np.sum(componentes**2,axis=0)
verificacion

#Varianza de los componentes 
var_explicada = pca.explained_variance_
ratio_var_exp= pca.explained_variance_ratio_

#Calculo de correlaciones
correl = np.corrcoef(x_train_sc,rowvar=False)

correl_pca = np.corrcoef(x_train_pca,rowvar=False)

#################################################
##### Primer componente
#################################################
#Aplicando componentes principales
from sklearn.decomposition import PCA
#Generando objetos
pca_4=PCA(n_components=3)
#Aprendiendo 
x_train_pca_3=pca_3.fit_transform(
        x_train_sc)
#Aplicando lo aprendido
x_test_pca_3=pca_3.transform(
        x_test_sc)

#Calculando coeficientes de los componentes
componentes_3 = pca_3.components_
#Verificacion
verificacion_3 = np.sum(componentes_3**2,axis=0)
verificacion_3

#Varianza de los componentes 
var_explicada_3 = pca_3.explained_variance_
ratio_var_exp_3 = pca_3.explained_variance_ratio_

#Calculo de correlaciones
correl_3 = np.corrcoef(x_train_sc,rowvar=False)

correl_pca_3 = np.corrcoef(x_train_pca,rowvar=False)
