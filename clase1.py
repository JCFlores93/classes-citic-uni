# -*- coding: utf-8 -*-
"""
Editor de Spyder
Este es un archivo temporal
"""

add = lambda x, y: x + y
print(add(3, 4))
str1 = "Hola"


#si un valor es menor que 10, 20 , >= 20

k = 8 
def getValue(k):
    if k < 10:
        return 'Menor que 10'
    if k < 20:
        return 'Menor que 20'
    if k >= 20:
        return 'Mayor o igual que 20'

k = 8 
def function1(k):
    if k < 10:
        print('Menor que 10')
    elif k <= 20:
        print( 'Menor o igual que 20')
    elif k <= 30:
        print( 'Menor o igual que 30')
    else:
        print('Mayor que 30')
        
def function2(k):
    print('Es multiplo de 3' if(k % 3 == 0) else 'no es multiplo de 3')
    
function1(11)
function2(9)


contador = 0 
while contador <= 5: 
    print('Hola', contador)
    contador += 1
    
print(list(range(5)))
lista_nombres = ('jean', 'silvana','rocio')
print(type(lista_nombres))

for x in lista_nombres: 
    print('Hola ', x)
    
import math 
math.sin(0)
math.cos(0)

from random import uniform
print(uniform(0,1))

from random import randint as rnd 
print(rnd(1,6))

#Subsetting 
##################################################################
numbers = (1, 2, 3, 70, 60, 55, 21, 15, 4)
print(numbers)
print(numbers[4])
print(numbers[2:6])

cadena = "La casa verde"
len(cadena) 
cadena.upper()
cadena.lower()



























