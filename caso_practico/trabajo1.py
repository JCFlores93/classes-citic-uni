# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:52:26 2018

@author: jean
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns; sns.set()

data = pd.read_csv('caso.csv')

data_zona = data[['zona']]
data_zona.describe()
data_zona.apply(pd.value_counts).plot(kind='bar', 
       subplots=False, 
       legend=True, 
       title='Departamento')

sns.distplot(data['edad'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})
plt.show()
plt.clf()

data_edad = data[['edad']]
data_edad.mean()
sns.set(style="whitegrid")
ax = sns.boxplot(x=data_edad)

g=sns.FacetGrid(data,row='clasif_sbs',col='nivel_educ')
g.map(plt.hist,"linea_sf")

sns.boxplot(x="linea_sf",y="nivel_educ",data=data)

plt.scatter(x=data['edad'],y=data['linea_sf'])
plt.show()

ax = sns.scatterplot(x="linea_sf", y="clasif_sbs", data=data)
