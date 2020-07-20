##LABORATORIO DE COMPORTAMIENTO SOCIAL E INTELIGENCIA ARTIFICIAL
#Laurent Avila Chauvet / Julio 2020
#Reto Kellogg @Talent-Home (DIL)

import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def Find(Data1,Data2,Element):
    Index = [i for i, x in enumerate(Data1) if x == Element]
    R = [Data2[i] for i in Index]
    return R 

def Prop(x,Data1,Data2):
    R = [0 for i in range(x)]
    for i in range(x):
        R[i] = sum(Find(Data1,Data2,i+1))/sum(Data2)
    return R

def SK(x, y): 
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    SSxy = np.sum(y*x) - n*my*mx 
    SSxx = np.sum(x*x) - n*mx*mx 
    S = SSxy / SSxx 
    K = my - S*mx 
    return(S,K) 
    
with open('Data_1A.csv', 'r') as File:
    Reader = csv.reader(File)
    Data = [i for i in Reader]

Region = [int(i[10]) for i in Data]
Venta = [int(i[9]) for i in Data]
Pobla = [int(i[11]) for i in Data]

MVenta = Prop(4,Region,Venta)
MPobla = Prop(4,Region,Pobla)

[Sensitivity, Bias] = SK(np.array(MPobla),np.array(MVenta))

print('Sensitivity: ', Sensitivity)
print('Bias:        ',Bias)

X = np.array(MPobla)
Y = np.array(MVenta)
R = sm.OLS(Y,sm.add_constant(X)).fit()
X2 = np.linspace(0,1,100)

plt.figure(1)   
plt.plot(np.array(MVenta), np.array(MPobla), 'ro')
plt.plot(X2, X2*R.params[1] + R.params[0],'k-')
plt.title('PAKETITO FROOT LOOPS 25GR')
plt.xlabel('Poblacion')
plt.ylabel('Ventas')
plt.axis([0, 1, 0, 1 ])
plt.show()




