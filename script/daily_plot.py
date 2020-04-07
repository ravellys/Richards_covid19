import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.ticker import FuncFormatter
from math import log10, floor

def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor (log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 0)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def desacum(X):
    a = [X[0]]
    for i in range(len(X)-1):
        a.append(X[i+1]-X[i])
    return a

def size_pop(FILE, população):
    for i in população:
        if i[0] == FILE[9:-4]:
            return float(i[1])

população = [["Espanha",46.72],["Itália",60.43],["SP",45.92],["MG",21.17],["RJ",17.26],["BA",14.87],["PR",11.43],["RS",11.37],["PE",9.6],["CE",9.13],["Pará",8.6],["SC",7.16],["MA",7.08],["GO",7.02],["AM", 4.14],["ES",4.02],["PB",4.02],["RN",3.51],["MT",3.49],["AL", 3.4],["PI",3.3],["DF",3.1],["MS",2.8],["SE",2.3],["RO",1.78],["TO",1.6],["AC",0.9],["AM",0.85],["RR",0.61],["Brasil",210.2]]
população = np.array(população)

mypath = 'C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/data_simulated/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
path_out = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/imagens/daily_cases/"

estados = ["COVID-19 Brasil.CSV", "COVID-19 SP.CSV", "COVID-19 RJ.CSV","COVID-19 AM.CSV","COVID-19 DF.CSV","COVID-19 PE.CSV", "COVID-19 CE.CSV", "COVID-19 PR.CSV"]

inf = []
for i in onlyfiles:

    estado = i
    
    df_simulated = pd.read_csv(mypath+estado,header = 0 , sep =";")

    df_plot = df_simulated[["Cases"]]
    df_plot["datetime"] = pd.to_datetime(df_simulated["date"])
    df_plot[estado[9:-4]] = desacum(df_simulated["Cases"].values)
    
    max_cases = max(df_plot[estado[9:-4]])
    for i in range(len(df_plot)):
        if df_plot[estado[9:-4]][i] == max_cases:
            pos_max = i
    max_day = df_plot["datetime"][pos_max]
    print(max_day)
    
    inf.append([estado[9:-4], max_cases, pos_max, max_day])
    
    fig,ax = plt.subplots(1, 1)
    figure = df_plot.plot(ax =ax,kind = "line", x = "datetime", y = estado[9:-4],
                             grid = True,rot = 90,figsize= (5,5), legend = None)
    figure.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax.set_ylim(0,max_cases*1.1)
    val = format_func(max_cases)        
    ax.annotate(val, (max_day, max_cases), fontsize = 14,ha='center', va='bottom')

    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.set_title(estado[9:-4], family = "Serif", fontsize = 18)
    figure.set_ylabel("Daily Cases", family = "Serif", fontsize = 16)
    figure.set_xlabel(" ")
    
    fig.savefig(path_out + estado[:-4]+'.png', dpi = 300,bbox_inches='tight',transparent = True)

inf = np.array(inf)

df_inf = pd.DataFrame(inf, columns = ["Estado","max_cases","diamax","datamax"])
path_out ="C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/inf/"
df_inf.to_csv(path_out+"inf_max_day.csv",sep=";")

def bar_plt(atributo, title_name,df,logscale):
    fig, ax = plt.subplots(1, 1)
    df = df.sort_values(by=[atributo])

    figure = df.plot.bar(ax =ax, x = "Estado", y =atributo,figsize = (15,8), legend = None,width=.75, logy = logscale)
    figure.set_xlabel(" ")
    figure.set_title(title_name, family = "Serif", fontsize = 22)
    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.yaxis.set_major_formatter(plt.FuncFormatter(format_func)) 

    for p in ax.patches:
        b = p.get_bbox()
        val = format_func(b.y1 + b.y0,1)        
        ax.annotate(val, ((b.x0 + b.x1)/2, b.y1 +0.25/100), fontsize = 14,ha='center', va='bottom',rotation = 90)

    plt.show()
    path_out ="C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/imagens/"
    fig.savefig(path_out+atributo+'_barplot.png', dpi = 300,bbox_inches='tight',transparent = True)

bar_plt(atributo = "diamax", title_name = "Number of days between the start \nand the peak of the epidemic", df = df_inf, logscale = True)
bar_plt(atributo = "max_cases", title_name = "Maximum number of daily cases", df = df_inf, logscale = True)
