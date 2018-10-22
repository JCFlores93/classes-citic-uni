# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:52:58 2018

@author: USUARIO
"""

#Ejercicio portafolio de dos activos 
def portaf2(w1, w2, r1, r2, std1, std2, rho):
    rendimiento = w1 * r1 + w2 * r2
    volatilidad = ((w1 * std1) ** 2 + (w2 * std2)** 2 + 2 *w1*w2*rho*std1*std2)**0.5
    print('El rendimiento esperado del portafolio es: ', round(rendimiento*100,2), '%')
    print('La volatilidad esperado del portafolio es: ', round(volatilidad*100,2), '%')

portaf2(0.7,0.3,0.05,0.08,0.02,0.03,0.5)
portaf2(0.7,0.3,0.05,0.08,0.02,0.03,0)
portaf2(0.7,0.3,0.05,0.08,0.02,0.03,-0.5)