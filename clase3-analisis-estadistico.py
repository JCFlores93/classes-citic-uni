# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 19:35:08 2018

@author: CTIC - UNI
"""

#Importando librerias
import pandas as pd
import numpy as np

#Generando data frame
df = pd.DataFrame([[1, np.nan,2, np.nan],
                   [2,3,5,np.nan],
                   [np.nan, 4, 6, np.nan]])

#Asignando nombres a las columnas 
df.columns = ['x1','x2','x3','x4']

#Asignando indices( nombres a las filas)
df.index = ['a','b','c']

df['x2']['b']

#contando valores en missing
df.isnull()
df[['x4']].isnull()
#df[['x4'][2:]].isnull()
type(df[['x4',1:]])
df.isnull()['x4'] #Columna x4
df.isnull()[df.index == 'b'] #Fila b
type(df[df.index == 'b'].isnull())

sum(df['x4'].isnull())
len(df['x4']sum(df['x4'].isnull())/len(df['x4']))

#Tratamiento de missings 

#1) Eliminando observaciones
df.dropna() #Elimina filas o columnas
df.dropna(axis=1) # Columnas
df.dropna(axis = 0) #Filas

# Elimina filas o columnas donde todo es missing
df.dropna(how="all")
df.dropna(how="all", axis=1)

#establecer umbrales
df.dropna(thresh=1) #Por lo menos un poblado
df.dropna(axis=1, thresh=3) #Por lo menos 3 poblaciones

# Guardar los cambios
df
df.dropna(axis=1, thresh=3, inplace=True)
df
#De manera analoga
df=df.dropna(axis=1, thresh 3)

#2) imputacion de missings
#Generando data frame
df = pd.DataFrame([[1, np.nan,2, np.nan],
                   [2,3,5,np.nan],
                   [np.nan, 4, 6, np.nan]])
#Asignando nombres a las columnas 
df.columns = ['x1','x2','x3','x4']

#Asignando indices( nombres a las filas)
df.index = ['a','b','c']

# Asignando 0 la missing
df.fillna(0)
df.x1= df.x1.fillna(9999)
df

# Asignando promedio al missing de x2
df 
df.x2= df.x2.fillna(np.mean(df['x2']))
df


df = pd.DataFrame([[1, np.nan,2, np.nan],
                   [2,3,5,np.nan],
                   [np.nan, 4, 6, np.nan]])
#Asignando nombres a las columnas 
df.columns = ['x1','x2','x3','x4']

#Asignando indices( nombres a las filas)
df.index = ['a','b','c']

#Completar data con el valor previo
df
df.fillna(method='ffill', axis=0)
df.fillna(method='ffill', axis=1)

#Completar con el valor siguiente
df
df.fillna(method='bfill', axis=0)
df

norm = np.random.normal(loc=5000, scale=125, size=(1000,1))
np.percentile(norm, [0,1,5,95,99,100])

ej1 = np.where(5>3,1,0)

norm2 = np.random.normal(loc=5000, scale=125, size=(1000,1))
df2 = pd.DataFrame(norm2)
df2.columns = ['norm']
df2['norm_acot']=np.where(df2.norm <= 4701,4701,np.where(df2.norm >= 5312,5312,df2.norm))
df2.describe()

#Eliminando
df3=df2[(df2.norm<5280)&(df2.norm>4720)]


