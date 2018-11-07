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


sns.distplot(data['casa_f'],
             hist=False,
             rug=True,
             kde_kws={'shade':True})
plt.show()
plt.clf()


#Análisis bivaraido - scatter
sns.regplot(x="linea_sf",
            y="edad",
            data=data)
plt.show()
plt.clf()

data1 = data.drop(['id','empleo','dias_lab','casa_f'], axis=1)


data1.describe()

#Variables categoricas
# mora
# casa
# zona
# nivel educacional



plt.hist(data1.mora)

data1.mora.describe()

#Análisis univariado
#Categorica
zona = data1[['zona','mora']]
zona.zona.plot.bar()
zona.hist()
type(zona)
iris.hist()
