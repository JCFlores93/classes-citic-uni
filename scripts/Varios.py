# Rank and Sort
################

# Importando librerías
import numpy as np
import pandas as pd

# Importando data
data=pd.read_csv('endangeredLang.csv')

disc1=data.groupby(['Countries','Degree of endangerment'])['ID'].count()
disc1=data.groupby(['Countries','Degree of endangerment'],as_index=False)['ID'].agg({'n':'count'})
disc1=data.groupby(['Countries','Degree of endangerment'],as_index=False)['Number of speakers'].agg({'minimo':'min',
                  'maximo':'max'})

disc1=data.groupby(['Countries','Degree of endangerment'],as_index=False)['Number of speakers'].agg({'ejem1':[min,max,sum]})
data.columns

#Function	Description
#count	Number of non-null observations
#sum	Sum of values
#mean	Mean of values
#mad	Mean absolute deviation
#median	Arithmetic median of values
#min	Minimum
#max	Maximum
#mode	Mode
#abs	Absolute Value
#prod	Product of values
#std	Unbiased standard deviation
#var	Unbiased variance
#sem	Unbiased standard error of the mean
#skew	Unbiased skewness (3rd moment)
#kurt	Unbiased kurtosis (4th moment)
#quantile	Sample quantile (value at %)
#cumsum	Cumulative sum
#cumprod	Cumulative product
#cummax	Cumulative maximum
#cummin	Cumulative minimum


# Guardando data
dataf=data[0:5]

writer = pd.ExcelWriter('output.xlsx')
dataf.to_excel(writer,'Sheet1')
writer.save()


dataf.to_csv('aaa.csv')

# Matriz de varianzas y covarianzas
dataf2=data.iloc[:,3:6]
a=np.cov(dataf2,rowvar=False)
# Matriz de correlaciones
b=np.corrcoef(dataf2,rowvar=False)












