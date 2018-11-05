# Rank and Sort
################

# Importando librerías
import numpy as np
import pandas as pd

# Importando data
data=pd.read_csv('GlobalFirePower.csv')

# Ordenando por indices
df1=data.sort_index(axis=0)
df2=data.sort_index(axis=1)

# Ordenando por información de la base
df3=data.sort_values(by='Combat Tanks')
df4=data.sort_values(by='Combat Tanks',
                     ascending=False)

df_peru=data[data.Country=='Peru']

df5=data.sort_values(by='Combat Tanks',
                     ascending=False)[['Country',
                     'Combat Tanks']][0:5]

data.columns
df6=data.sort_values(by=['Combat Tanks',
                         'Submarines'])

# Discretizando variables
data['Estado']=np.where(data.Number<100,'Extinto',
    np.where(data.Number<50000,'Peligro','Normal'))

# Resumendo información
data['Estado'].value_counts()
disc1=data.groupby(['Estado']).mean()
disc2=data.groupby(['Estado']).max()
disc3=data.groupby(['Estado']).min()
disc4=data.groupby(['Estado']).std()
disc5=data.groupby(['Estado']).median()

describe=pd.DataFrame(data.describe())



# Guardando data
dataf=data[0:5]

writer = pd.ExcelWriter('output.xlsx')
dataf.to_excel(writer,'Sheet1')
writer.save()


dataf.to_csv('aaa.csv')













