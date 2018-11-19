# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:49:54 2018

@author: jean
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.covariance import EllipticEnvelope

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

print(data['edad'][334])

print(data[data['edad'] < 30]['edad'])

from scipy.stats import mode
for column in ['antig_tc', 'linea_sf','deuda_sf']:
    data[column] = data[column].fillna(data[column].median())

columna = 'atraso'
sns.set(style="whitegrid")
ax = sns.boxplot(x=data[columna])
norm = data[columna]
np.percentile(norm,[0.1,50,75,90])

feature_edad = data[columna]
np.percentile(feature_edad,[25,50,95])

#################################
#Create a feature
feature_edad = data['clasif_sbs']

def indicies_of_outliers(x):
    q1, q3= np.percentile(x, [25,75])
    iqr = q3 -q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    print('====>',upper_bound)
    print('====>',lower_bound)
    return np.where((x > upper_bound) | (x < lower_bound))

outliers_edad = list(indicies_of_outliers(feature_edad))
    
#data = data.drop(data.index[outliers_edad])
########################################

data['edad']=np.where(data.edad<=9.5,9.5,
   np.where(data.edad>=61.5,61.5,data.edad))

data['atraso']=np.where(data.atraso>=5,5,data.atraso)

data['ingreso']=np.where(data.ingreso>=10150,10150,data.ingreso)
data['linea_sf']=np.where(data.linea_sf>=20524.52625,20524.52625,data.linea_sf)
data['deuda_sf']=np.where(data.deuda_sf>=12240.91875,12240.91875,data.deuda_sf)

data['score']=np.where(data.score<=137,137,
   np.where(data.score>=257,257,data.score))

data['clasif_sbs']=np.where(data.clasif_sbs>=2.5,2.5,data.clasif_sbs)
    
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
data_casa = data_casa.drop('OTRAS', axis=1)
train = pd.concat([data, data_casa], axis=1)

#Eliminamos casa y casa_f
train = train.drop(['casa', 'casa_f'], axis=1)

#Tratamos la columna zona
#One-hot encode feature
data_zona = one_hot.fit_transform(train['zona'])

#View feature classes
columns_data_zona = one_hot.classes_.tolist()
   
data_zona = pd.DataFrame(
        data=data_zona,
        columns=columns_data_zona
        )
data_zona = data_zona.drop('Ucayali', axis=1)
train = pd.concat([train, data_zona], axis=1)
train = train.drop('zona', axis=1)

#Tratando la columna nivel_educ

train['nivel_educ'].unique()
#Crear un mapper
scale_mapper = {
        "PROFESIONAL": 1,
        "TECNICO": 2,
        "SUPERIOR": 3,
        "EDUCACION BASICA": 4,
        "SIN EDUCACION": 5,
        }
train['nivel_educ'] = train['nivel_educ'].replace(scale_mapper)

#Tratamos la columna zona
#One-hot encode feature
data_nivel_educ = one_hot.fit_transform(train['nivel_educ'])

#View feature classes
#columns_data_nivel_educ = one_hot.classes_.tolist()
columns_data_nivel_educ = ["PROFESIONAL","TECNICO","SUPERIOR","EDUCACION BASICA","SIN EDUCACION"]
data_nivel_educ = pd.DataFrame(
        data=data_nivel_educ,
        columns=columns_data_nivel_educ
        )
data_nivel_educ = data_nivel_educ.drop('SIN EDUCACION', axis=1)
train = pd.concat([train, data_nivel_educ], axis=1)
train = train.drop('nivel_educ', axis=1)

len(train.columns)
#train = train.drop('dias_lab', axis=1)
#train = train.drop('empleo', axis=1)

train['edad'].unique()

train_edad = []
for row in train['edad']:
    if(row <= 35):
        train_edad.append("JOVEN")
    elif (row > 35 and row <=50):
         train_edad.append("ADULTO-JOVEN")
    else:
        train_edad.append("ADULTO-MAYOR")
train['edad'] = train_edad
#Crear un mapper
scale_mapper_edad = {
        "JOVEN": 1,
        "ADULTO-JOVEN": 2,
        "ADULTO-MAYOR": 3
        }
train['edad'] = train['edad'].replace(scale_mapper_edad)
#Tratamos la columna zona
#One-hot encode feature
data_edad = one_hot.fit_transform(train['edad'])

#View feature classes
columns_data_edad = one_hot.classes_.tolist()
columns_data_edad = ['JOVEN', 'ADULTO-JOVEN', 'ADULTO-MAYOR']
   
data_edad = pd.DataFrame(
        data=data_edad,
        columns=columns_data_edad
        )
data_edad = data_edad.drop('ADULTO-MAYOR', axis=1)
train = pd.concat([train, data_edad], axis=1)
train = train.drop('edad', axis=1)

train2 = train.copy()


def converting_cast(s):
    return s.replace(',', '')

def converting_to_float(x):
    return int(x)

train['dias_lab'] = train['dias_lab'].apply(converting_cast)
train['dias_lab'] = train['dias_lab'].apply(converting_to_float)


def converting_to_date(x):
    return x.timestamp()

def converting_string_to_date(x):
    return x * 1000

#COnvertido a string -> fecha
train2['empleo'] = pd.to_datetime(train2['empleo'])

train['empleo'] = pd.to_datetime(train['empleo'])

#a milisegundos
def converting_date_to_miliseconds(x):
    return int(round(x.timestamp() * 1000))


train['empleo'] = train['empleo'].apply(converting_date_to_miliseconds)

train['empleo'] = train['empleo'].apply(converting_string_to_date)
train['empleo'] = train['empleo'].apply(converting_cast)





#Generando variables x e y
x=train.drop('mora', axis=1)
y=train[['mora']]

#Separando base en train y test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

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


