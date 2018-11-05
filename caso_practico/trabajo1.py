# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:52:26 2018

@author: jean
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns


data = pd.read_csv('caso.csv')
data1 = data.drop(['id','empleo','dias_lab'], axis=1)

plt.hist(data1.mora)

data1.mora.describe()

#Análisis univariado
#Categorica
zona = data1[['zona','mora']]
zona.zona.plot.bar()
zona.hist()
type(zona)
iris.hist()
