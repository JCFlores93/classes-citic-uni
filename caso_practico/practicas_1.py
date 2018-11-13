# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 08:51:53 2018

@author: jean
"""

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
sns.set(style="whitegrid")
ax = sns.boxplot(x=data_edad)

#f =data[['antig_tc']]
#f.boxplot()
#plt.show()

#tips = sns.load_dataset("tips")
#g = sns.FacetGrid(tips, col="time")
#g.map(plt.hist, "tip");

#timer_dinner =  tips[(tips.tip >= 7) & (tips.time == 'Lunch')]

#g=sns.FacetGrid(tips,row='smoker',col='time')
#g.map(plt.hist,"total_bill")

#g=sns.FacetGrid(data,row='clasif_sbs',col='nivel_educ')
#g.map(plt.hist,"antig_tc")

g=sns.FacetGrid(data,row='clasif_sbs',col='nivel_educ')
g.map(plt.hist,"linea_sf")

sns.boxplot(x="linea_sf",y="nivel_educ",data=data)

plt.scatter(x=data['edad'],y=data['linea_sf'])
plt.show()

ax = sns.scatterplot(x="linea_sf", y="clasif_sbs", data=data)

#linea_sf = data['linea_sf']
#casa_f = data['casa_f']
#clasif_sbs = data['clasif_sbs']

#sns.boxplot(x="casa_f",y="linea_sf",
#           hue="clasif_sbs",data=data)
#https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f

#plt.plot(linea_sf, casa_f, color='g')
#plt.plot(linea_sf, clasif_sbs, color='orange')
#plt.xlabel('Countries')
#plt.ylabel('Population in million')
#plt.title('Pakistan India Population till 2010')
#plt.show()

# Set style of scatterplot
#sns.set_context("notebook", font_scale=1.1)
#sns.set_style("ticks")

# Create scatterplot of dataframe
#sns.lmplot('casa_f', # Horizontal axis
#           'linea_sf', # Vertical axis
#           data=data, # Data source
#           fit_reg=False, # Don't fix a regression line
#           hue="clasif_sbs", # Set color
#           scatter_kws={"marker": "D", # Set marker style
#                        "s": 100}) # S marker size
# Set title
#plt.title('Histogram of IQ')

# Set x-axis label
#plt.xlabel('Time')

# Set y-axis label
#plt.ylabel('Deaths')

#g=sns.FacetGrid(data,row='mora',col='nivel_educ')
#g.map(plt.hist,"linea_sf")

#g = sns.FacetGrid(data, col="nivel_educ")
#g.map(sns.barplot, "mora", "linea_sf");

#timer_dinner =  tips[(tips.tip >= 7) & (tips.time == 'Lunch')]

#f = data['antig_tc']
#plt.hist(f)
#plt.title('Histogram of score')
#plt.show()

#sns.distplot(data_zona, kde=False, rug=True)
#data_zona.boxplot(column=data_zona['nivel_educ'])

#data.hist(column='zona')
#plt.hist(data_zona)
#plt.show()
#plt.clf()

#data1 = data.drop(['id','empleo','dias_lab','casa_f'], axis=1)

#data1.describe()

#Variables categoricas
# mora
# casa
# zona
# nivel educacional

#plt.hist(data1.mora)

#data1.mora.describe()

#Análisis univariado
#Categorica
#zona = data1[['zona','mora']]
#zona.zona.plot.bar()
#zona.hist()
#type(zona)
#iris.hist()
