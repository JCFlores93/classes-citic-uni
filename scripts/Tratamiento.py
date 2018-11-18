# Trabajando con valores missings
#################################

# Importando librerías
import pandas as pd
import numpy as np

# Generando data frame
df=pd.DataFrame([[1,np.nan,2,np.nan],
                 [2,3,5,np.nan],
                 [np.nan,4,6,np.nan]])

# Asignando nombres a las columnas
df.columns=['x1','x2','x3','x4']
# Asignando indice (nombres a las filas)
df.index=['a','b','c']
# Subsetting
df['x2']['b']

# Contando valores en missing
df.isnull() # Todo el dataframe
df.isnull()['x4'] # Columna x4
df.isnull()[df.index=='b'] # Fila b
df[df.index=='b'].isnull()

sum(df['x4'].isnull())
len(df['x4'])
sum(df['x4'].isnull())/len(df['x4'])

# Tratamiento de missings
#########################

# 1) Eliminando observaciones

# Elimina filas o columnas con al menos 1 miss
df.dropna() 
df.dropna(axis=1) # columnas
df.dropna(axis=0) # filas

# Elimina filas o columnas donde todo es miss
df.dropna(how="all") # axis=0 por default
df.dropna(axis=1,how="all")

# Establecer umbrales
df.dropna(thresh=2) # Por lo menos 2 poblado
df.dropna(axis=1,thresh=2) # Por lo menos 2 poblado
df.dropna(axis=1,thresh=3) # Por lo menos 3 poblado

# Guardar cambios
df
df.dropna(axis=1,thresh=3,inplace=True) # Por lo menos 3 poblado
df
# De manera análoga
df=df.dropna(axis=1,thresh=3)

# 2) Imputación de missings
df=pd.DataFrame([[1,np.nan,2,np.nan],
                 [2,3,5,np.nan],
                 [np.nan,4,6,np.nan]])
df.columns=['x1','x2','x3','x4']
df.index=['a','b','c']

# Asignando 0 al missing
df.fillna(0)
# Asignando 999 al missing de x1
df
df.x1=df.x1.fillna(9999)
df
# Asignando promedio al missing de x2
df
df.x2=df.x2.fillna(np.mean(df['x2']))
df

# Generando dataframe inicial
df=pd.DataFrame([[1,np.nan,2,np.nan],
                 [2,3,5,np.nan],
                 [np.nan,4,6,np.nan]])
df.columns=['x1','x2','x3','x4']
df.index=['a','b','c']

# Completar data con el valor previo
df
df.fillna(method='ffill',axis=0)
df.fillna(method='ffill',axis=1)

# Completar data con el valor siguiente
df
df.fillna(method='bfill',axis=0)
df.fillna(method='bfill',axis=1)

# Tratamiento de outliers
##########################
norm=np.random.normal(5000,125,size=(1000,1))
np.percentile(norm,[0,1,5,95,99,100])


np.percentile(norm,[25,50,75])

ej1=np.where(5>3,1,0)

# Acotando
df2=pd.DataFrame(norm)
df2.columns=['norm']
df2['norm_acot']=np.where(df2.norm<=4720,4720,
   np.where(df2.norm>=5280,5280,df2.norm))

df2.describe()

# Eliminando valores extremos
df3=df2[(df2.norm<5280)&(df2.norm>4720)]

















































 














