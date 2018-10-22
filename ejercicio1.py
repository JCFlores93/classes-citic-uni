# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:35:30 2018

@author: USUARIO
"""

import pandas as pd 

ventas = pd.read_csv('./ventas.csv')
print(ventas)


#libreria random 
import random 
import numpy as np
import matplotlib.pyplot as plt

np.mean(ventas)
np.max(ventas)
np.min(ventas)
np.var(ventas)
np.std(ventas)
ventas.describe()

#Analizando ventas
np.percentile(ventas[[ 'ventas ' ]])