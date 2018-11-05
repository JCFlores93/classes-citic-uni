# Importando librerías
import numpy as np
import pandas as pd

# Cargando base de datos
data=pd.read_csv('endangeredLang.csv')

# Manipulación de DataFrames
#############################

# Ver los n primeros registros
data.head(n=5)
data_5reg=data.head(n=5)

# Selección de columnas
data.columns

df1=data['Countries']

df2=data[['Countries','Name in English']]

# Selección de filas
df3=data[3:10]

data['Number of speakers'].describe()
df4=data[data['Number of speakers']<=5000]

df5=data[data['Number of speakers']<=
         np.mean(data['Number of speakers'])]

df6=data[data['Number of speakers']==0]
df6.columns
df6['Degree of endangerment'].describe()
df5['Number of speakers'].describe()

# Generando un nuevo data frame
data2=pd.DataFrame(data)

data.index
data.columns

# Eliminando registros: primera fila
data2.drop(data2.index[0],inplace=True)

# Eliminando registros: múltiples filas
data2.drop(data2.index[:5],inplace=True)

# Eliminando registros:
data2.drop([12,14],inplace=True)

# Eliminando columnas
data2.drop(['Latitude'],axis=1,inplace=True)
data2.drop(['Longitude','Countries'],axis=1,
           inplace=True)

# Subsetting
s1=data[6:] #Registros desde el 6
s2=data[:6] #Registros hasta el 6
s3=data[::6] #Registros de 6 en 6

#Registros(0-7), Columnas(0-6)
s4=data.iloc[0:8,0:7] 
s5=data[['Countries','Latitude']][4:8]

s6=data.iloc[[5,9],0:7]
s7=data[['Countries','Latitude']].iloc[[5,9,59],:]

s8=data[data['Countries']=='Peru']
s9=data[data.Countries=='Peru'] #mas eficiente

s10=data[data['Countries'].isin(['Peru','Italy'])]
s11=data[(data['Countries']=='Peru') | 
        (data['Countries']=='Italy')]
s12=data[(data.Countries=='Peru') |
        (data.Countries=='Italy')]

# Renombrar
data.columns
data.rename(columns={'Number of speakers':'Number'},
            inplace=True)
data.columns

data['Degree of endangerment'].value_counts()






























































