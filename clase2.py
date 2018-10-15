# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:19:13 2018

@author: CTIC - UNI
"""
#Generacion de numeros aleatorios
#################################################

#libreria random 
import random 
 
#Distribucion uniforme
random.uniform(0,1)

#Distribucion entero
random.randint(1,6)

#Distribucion normal
print(random.normalvariate(0,1))


#libreria numpy
import numpy as np
import matplotlib.pyplot as plt
#Distribucion uniforme
unif = np.random.uniform(0,10, size=(10000,1))
plt.hist(unif)

#Generando semilla aleatoria
#params => limite inferior, limite superior
print(random.uniform(0,1))
random.seed(123)
print(random.uniform(0,1))

#Simulando distribucion normal
#params => promedio, desviacion estandar
norm = np.random.normal(15,3, size=(100,1))
plt.hist(norm)

#Simulando distribucion entero
entero = np.random.randint(1,6, size=(100000, 1))
plt.hist(entero)

#Calculo estadistico
np.mean(norm)

#Calculo del máximo
np.max(norm)

#Calculo del mínimo
np.min(norm)

#Calculo de la desviacion estandar
np.max(norm)

#Calculo de la varianza
np.var(norm)

#Cálculo de percentiles 
np.percentile(norm, 50)
np.percentile(norm, 10)
np.percentile(norm, 90)
np.percentile(norm, 99.9)
np.percentile(norm, 0.01)
np.percentile(norm, [1,5,25,50,75,95,99])

#libreria scipy.stats
from scipy.stats import kurtosis, skew
kurtosis(norm)
skew(norm)
