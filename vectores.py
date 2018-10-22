# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 19:48:16 2018

@author: USUARIO
"""
import numpy as np
#Generando vectores
v1 = [2, 5, 7]
v2 = np.array([1, 3, 5])
type(v2)

v3 = np.ones(3)
v3

v4 = np.arange(1, 8)
v4

#Operaciones con vectores
v1 + v2 #Suma
v1 - v2 #Resta
v1 @ v2 #Producto Escalar
-3 * v2 #Operaciones combinadas

#Trabajando con matrices 
m1 = np.array([[2,6], [3, 8]])
m1

m2 = np.array([[1, 7], [3, 2]])
m2

#Operaciones con matrices 
#Suma, resta y operaciones combinadas
m1 + m2 
m1 - m2
2 * m1 - 3 * m2

#Dimensiones de una matriz
m1.shape

#Cantidad de elementos
m1.size

#Ejemplos
#reshape ==> 4 filas y 2 columnas
m3 = np. arange(8).reshape(4,2)
m3

#reshape ==> 2 filas y 4 columnas
m4 = np. arange(8).reshape(2,4)
m4
