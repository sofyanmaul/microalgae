# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:11:31 2022

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import r2_score

x_data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y_data=np.array([2.6, 2.8, 3.65, 8.65, 10.9, 13.15, 18, 19, 20.95, 21.00, 22.50, 23.15, 21.50])
Xo=y_data[0]
Xmaks=max(y_data)

def func(x,miumax):
    return Xo*np.exp((miumax)*x_data)/(1-((Xo/Xmaks)*(1-np.exp((miumax)*x_data))))

popt, pcov = optimize.curve_fit(func,x_data,y_data, 1)
print('miumax = {:6.4f} \n'.format(*popt))
y_calc = func(x_data,popt[0])
r2=r2_score(y_data,y_calc)
print('r2 score for the model is', r2)

plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data*10**(-0), label='Data')
plt.xlabel('Days')
plt.ylabel('$\it{Chlorella}$ $\it{sp.}$ cell density (10^6)')
plt.plot(x_data, y_calc*10**(-0),'-b', label='Fitted function')
plt.legend(loc='best')
plt.show()

#%%
tabel = np.array([y_calc])

header = ['Data Fitting']
garis = '-'*40
print(garis)
print ('{:^1s}'.format(*header))
print (garis)
for baris in np.transpose(tabel):
    print ('{:^10.4f}'.format(*baris))
print(garis)
