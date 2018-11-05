# Gestión de bases de datos
###########################

# Importando librerías
import numpy as np
import pandas as pd

# Importando data
notas1=pd.read_csv('Notas1.csv',sep=";")
notas2=pd.read_csv('Notas2.csv',sep=";")
notas3=pd.read_csv('Notas3.csv',sep=";")
notas4=pd.read_csv('Notas4.csv',sep=";")

# Append: Concatenar
notas12=pd.concat([notas1,notas2])
notas34=pd.concat([notas3,notas4]) # Cruzo mal
# Renombrar
notas3.rename(columns={'ECONOMIA':'Economia',
                       'ECONOMETRIA':'Econometria'},
inplace=True)
# Cruce correcto
notas34=pd.concat([notas3,notas4])

# Merge
notas1234=pd.merge(notas12,notas34,on='Estudiante')

# Eliminación de duplicados
notas=pd.read_csv('Notas.csv',sep=";")

notas_c1=pd.merge(notas3,notas,on='Estudiante')
notas_c2=pd.merge(notas3,notas,on='Estudiante',
                  how='left')
notas_c3=pd.merge(notas3,notas,on='Estudiante',
                  how='right')

notas_nodup=notas.drop_duplicates(subset=None,
                                  keep='first')

notas_nodup2=notas.drop_duplicates(subset='Estudiante',
                                  keep='first')






















