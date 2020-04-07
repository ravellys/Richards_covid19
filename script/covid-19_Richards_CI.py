# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:50:20 2020

@author: ravel
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import hydroeval as hy
from scipy import stats

def nstf(ci):
    # Convert to percentile point of the normal distribution.
    pp = (1. + ci) / 2.
    # Convert to number of standard deviations.
    return stats.norm.ppf(pp)

def date_to_str(X):
    str_X=np.datetime_as_string(X, unit='D')
    x=[]
    
    for i in str_X:
        x.append(i[5:7]+'/'+ i[-2:])
    return np.array(X)    

def desacum(X):
    a = [X[0]]
    for i in range(len(X)-1):
        a.append(X[i+1]-X[i])
    return a
     
def f(C, t, r, p, K, alfa):
    
    return r*(C**p)*(1-(C/K)**alfa)

def C(t, r, p, K, alfa,Co):
 
    C = odeint(f, Co,t, args=(r, p, K, alfa))
    return C.ravel()

#import mensured data
FILE = "Covid_Brasil-PE.csv"
t0=0
extrapolação = 5

data_covid = pd.read_csv(FILE, header = 0, sep = ";")
data_covid=data_covid[['DateRep','Deaths','Cases']]
day_0_str=data_covid['DateRep'][0][-4:]+'-'+data_covid['DateRep'][0][3:5]+'-'+data_covid['DateRep'][0][:2]
date = np.array(day_0_str, dtype=np.datetime64)+ np.arange(len(data_covid))
date= date[t0:]

cumdata_covid = data_covid[['Deaths','Cases']].cumsum()

cumdata_cases = cumdata_covid['Cases'].values[t0:]
days_mens = np.linspace(1,len(cumdata_cases),len(cumdata_cases))

# Chamada da rotina curve_fit
from scipy.optimize import curve_fit

binf = [0.01,0,0,0,cumdata_cases[0]-10**-6]
bsup = [40,1,100000,100,cumdata_cases[0]+10**-6]

popt, pcov = curve_fit(C, days_mens, cumdata_cases, 
                       p0 = [2,.5,50068.679 ,1,cumdata_cases[0]],
                       bounds = (binf,bsup),
                       absolute_sigma = True)

r, p, K, alfa,Co = popt 

perr = np.sqrt(np.diag(pcov))
Nstd = nstf(.95)
popt_up = popt + Nstd * perr
popt_dw = popt - Nstd * perr

print("r = %f " % (r))
print("p = %f " % (p))
print("K = %f " % (K))
print("alfa = %f " % (alfa))
print("Co = %f " % (Co))

date_future = np.array(date[0], dtype=np.datetime64)+ np.arange(len(date)+extrapolação)
days_future = np.linspace(1,len(cumdata_cases)+extrapolação,len(cumdata_cases)+extrapolação)
Cum_cases_estimated = C(days_future, *popt)
Cum_cases_estimated_up=C(days_future, *popt_up)
Cum_cases_estimated_dw=C(days_future, *popt_dw)

NSE = hy.nse(cumdata_cases,C(np.linspace(1,len(cumdata_cases),len(cumdata_cases)), r, p, K, alfa,Co) )

fig = plt.figure(figsize = (6,4))
plt.plot( pd.to_datetime(date), cumdata_cases,'o',label='Medidos')
plt.plot( date_future, Cum_cases_estimated_up, '--',color = 'r', label='IC-95%')
plt.plot( date_future, Cum_cases_estimated_dw, '--',color = 'r')
plt.plot( date_future, Cum_cases_estimated, '-',label='Estimados')

plt.text(date_future[0],.65*max(Cum_cases_estimated),
         'Parametros:'\
         +'\n'+'r = %.2f' % (r) \
         +'\n'+ "p = %.2f " % (p)\
         +'\n'+ "K = %.2f " % (K)\
         +'\n'+ "alfa = %.2f " % (alfa)\
         +'\n'+ "Co = %.2f " % (Co) ,
         family = 'serif',
         fontsize=12,
         bbox=dict(facecolor='blue', alpha=0.3))

plt.text(date_future[0],0.3*max(Cum_cases_estimated),
         'NSE: %.2f'%(NSE),
         family = 'serif',
         fontsize=12,
         bbox=dict(facecolor='blue', alpha=0.3))

plt.ylabel("Casos Totais",fontsize=15)
plt.xticks( family = 'serif',fontsize=11,rotation=90)

plt.grid()
plt.legend(loc=9, ncol=1, shadow=True, title="Legenda")
plt.title(FILE[:-4], family = 'serif',fontsize=15)
plt.show()


fig = plt.figure(figsize = (6,4))

plt.plot( pd.to_datetime(date), desacum(cumdata_cases),'o',label='Medidos')
plt.plot( date_future, desacum(Cum_cases_estimated), '-',label='Estimados')
plt.plot( date_future, desacum(Cum_cases_estimated_up), '--',color = 'r', label='IC-95%')
plt.plot( date_future, desacum(Cum_cases_estimated_dw), '--',color = 'r')

plt.text(date_future[0],.5*max(desacum(Cum_cases_estimated)),
         'Parametros:'\
         +'\n'+'r = %.2f' % (r) \
         +'\n'+ "p = %.2f " % (p)\
         +'\n'+ "K = %.2f " % (K)\
         +'\n'+ "alfa = %.2f " % (alfa)\
         +'\n'+ "Co = %.2f " % (Co) ,
         family = 'serif',
         fontsize=12,
         bbox=dict(facecolor='blue', alpha=0.3))

plt.locator_params(axis='x', nbins=1000)
plt.xticks(family = 'serif',fontsize=11,rotation=90)
plt.ylabel("Casos diários",fontsize=15)

plt.grid()
plt.legend(loc=9, ncol=1, shadow=True, title="Legenda")
plt.title(FILE[:-4], family = 'serif',fontsize=15)
plt.show()


###################
# importar bibliotecas
import plotly.graph_objects as go
import numpy as np
from plotly.offline import plot

x = pd.to_datetime(date)
y = cumdata_cases

fig = go.Figure()
fig.update_layout(title_text="Casos de COVID-19 no Brasil",
                  title_font_size=30,
                  yaxis_title="Nº total de casos \n (escala logarítmica)",
                  font=dict(
                          family="Serif",
                          size=16,
                          color="black"
                          ))
fig.update_layout(yaxis_type="log")


fig.add_trace(go.Scatter(
    x=date_future, y=Cum_cases_estimated,
    name='Estimado'
))
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    name='medidos',
    marker=dict(color='purple', size=8)
))

fig.add_trace(go.Scatter(
    x=date_future[-1*extrapolação:], y=Cum_cases_estimated[-1*extrapolação:],
    mode='markers',
    name='estimados futuros',
    marker=dict(color='red', size=8)
))

fig.add_trace(go.Scatter(
    x=date_future[-1*extrapolação:], y=Cum_cases_estimated_up[-1*extrapolação:],
    name='Intervalo de confiança (95%)',
    marker=dict(color='red', size=8)
))
fig.add_trace(go.Scatter(
    x=date_future[-1*extrapolação:], y=Cum_cases_estimated_dw[-1*extrapolação:],
    name='Intervalo de confiança (95%)',
    marker=dict(color='red', size=8)
))

plot(fig,filename="Previsão Futura dos Casos de COVID-19 (Brasil).html")



