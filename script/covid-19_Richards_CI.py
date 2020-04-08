import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import hydroeval as hy
from scipy import stats
from os import listdir
from os.path import isfile, join
from SALib.sample import saltelli
from SALib.analyze import sobol
from math import log10, floor

def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor (log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 1)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def bounds(FILE):
    pasta = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/range/"
    df_range = pd.read_csv(pasta+"/"+FILE, header = 0, sep = ";")
    p0 = df_range["p0"].values
    bsup = df_range["bsup"].values
    binf = df_range["binf"].values
    return [p0.tolist(),bsup.tolist(),binf.tolist()]


def Sensitivity(mypath,FILE,pop):
    print(FILE)
    t0 = 0 
    data_mensured = pd.read_csv(mypath+"/"+FILE, header = 0, sep = ";")
    data_mensured = data_mensured[['DateRep','Cases']]
    cumdata_covid = data_mensured[['Cases']].cumsum()
    cumdata_cases = cumdata_covid['Cases'].values[t0:]
    t = np.linspace(1,len(cumdata_cases),len(cumdata_cases))

    for i in população:
        if i[0] == FILE[9:-4]:
            pop = float(i[1])

    problem = {
            'num_vars': 4,
            'names': ['r', 'p', 'K','alfa'],
            'bounds': [[0.001, 50],
                       [0., 1.0],
                       [np.log10(0.01*pop*10**6), np.log10(.5*pop*10**6)],
                       [-6,2]]
            }

    nsamples = 5000
    param_values = saltelli.sample(problem,nsamples)

    Y = []
    W = []

    for i in param_values:
        r, p, K, alfa = i
        alfa = 10**alfa
        K = 10**K
        Co = cumdata_cases[0]
    
        cases_simulated = C(t, r, p, K, 10**alfa,Co)

        NSE = hy.nse(cases_simulated,cumdata_cases)
        if NSE <= 1:
            NSE = NSE
        else:
                NSE = -1000000000
        if NSE >= 0.75:
            W.append([r, p, K, alfa,NSE])
        
        Y.append([r, p, K, alfa,NSE])

    Y = np.array(Y)
    W = np.array(W)
    print(W)
    
    df_Y = pd.DataFrame(Y,columns = ['r', 'p', 'K','alfa','NSE'])
    df_W = pd.DataFrame(W,columns = ['r', 'p', 'K','alfa','NSE'])

    Si = sobol.analyze(problem,df_Y["NSE"].values)
    df_S = pd.DataFrame(Si["S1"], columns = ["S1"])
    df_S["ST"] = Si["ST"]
    df_S.to_csv("C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/sensibilidade/"+FILE,sep=";")

    binf = np.array([min(df_W.r),min(df_W.p), min(df_W.K), min(df_W.alfa)])
    bsup = np.array([max(df_W.r),max(df_W.p), max(df_W.K), max(df_W.alfa)])

    NSE_max = max(df_W.NSE)
    for i in range(len(df_W)):
        if df_W.NSE[i] == NSE_max:
            pos_max = i

    p0 = np.array([df_W.r[pos_max],df_W.p[pos_max],df_W.K[pos_max],df_W.alfa[pos_max]])

    par = np.array([binf,bsup,p0])

    df_par = pd.DataFrame(par, ["binf","bsup", "p0"]).transpose()
    df_par.to_csv("C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/range/"+FILE,sep=";")

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
 
    C = odeint(f, Co, t, args=(r, p, K, alfa))
    return C.ravel()

def ajust_curvefit(days_mens,cumdata_cases,p0,bsup,binf):
    popt, pcov = curve_fit(C, days_mens, cumdata_cases,
                           bounds = (binf,bsup),
                           p0 = p0,
                           absolute_sigma = True)
    return popt

from scipy.optimize import minimize

def object_minimize(x,t,cumdata_cases):
 
    cum_cases = odeint(f,x[4],t, args =(x[0],x[1],x[2],x[3]))
    
    
    #return sum(abs(cumdata_cases-cum_cases.ravel()))
    #return sum((np.log10(cumdata_cases)-np.log10(cum_cases.ravel()))**2)

    return sum((cumdata_cases-cum_cases.ravel())**2)

def min_minimize(cumdata_cases,C,p0,t,bsup,binf):
    bnds = ((binf[0],bsup[0]),(binf[1],bsup[1]),(binf[2],bsup[2]),(binf[3],bsup[3]),(binf[4],bsup[4]))
    res = minimize(object_minimize, p0, args = (t,cumdata_cases), bounds = bnds, method='TNC')
    return res.x

def Ajust_SUCQ(FILE,pop,extrapolação,day_0,variavel,pasta):    
        
    data_covid = pd.read_csv(pasta+"/"+FILE, header = 0, sep = ";")
    data_covid = data_covid[['DateRep',variavel]]
    day_0_str = data_covid['DateRep'][0][-4:]+'-'+data_covid['DateRep'][0][3:5]+'-'+data_covid['DateRep'][0][:2]
    date = np.array(day_0_str, dtype=np.datetime64)+ np.arange(len(data_covid))
    
    if date[0]>=np.array(day_0, dtype=np.datetime64):
        t0 = 0
    else:
        dif_dias =np.array('2020-03-18', dtype=np.datetime64)-date[0]
        t0 = dif_dias.astype(int)
    
    date= date[t0:] 
    
    cumdata_covid = data_covid[['Cases']].cumsum()

    cumdata_cases = cumdata_covid['Cases'].values[t0:]
    days_mens = np.linspace(1,len(cumdata_cases),len(cumdata_cases))

    p0,bsup,binf = bounds(FILE)
    p0.append(cumdata_cases[0])
    bsup.append(cumdata_cases[0]+10**-9)
    binf.append(cumdata_cases[0]-10**-9)

    popt = min_minimize(cumdata_cases,C,p0,days_mens,bsup,binf)
    r,p,K,alfa,Co = popt 
       
    solution = C(days_mens, r, p, K, alfa,Co)
    
    NSE = hy.nse(cumdata_cases,solution)
    print(FILE[9:-4])
    print("r = %f " % (r))
    print("p = %f " % (p))
    print("K = %f " % (K))
    print("alfa = %f" %(alfa))
    print("NSE = %.5f " % (NSE))
    print("#######################")
      
    days_mens = np.linspace(1,len(cumdata_cases)+365,len(cumdata_cases)+365)
    
    solution = C(days_mens, r, p, K, alfa,Co)
    date_future = np.array(date[0], dtype=np.datetime64)+ np.arange(len(date)+extrapolação)
    saída = pd.DataFrame(solution, columns=["Cases"])
    saída["date"] = date_future
    path_out = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/data_simulated"      
    saída.to_csv(path_out+"/"+FILE,sep=";")

    return [r,p,K,alfa,NSE]

#import mensured data
população = [["Espanha",46.72],["Itália",60.43],["SP",45.92],["MG",21.17],["RJ",17.26],["BA",14.87],["PR",11.43],["RS",11.37],["PE",9.6],["CE",9.13],["Pará",8.6],["SC",7.16],["MA",7.08],["GO",7.02],["AM", 4.14],["ES",4.02],["PB",4.02],["RN",3.51],["MT",3.49],["AL", 3.4],["PI",3.3],["DF",3.1],["MS",2.8],["SE",2.3],["RO",1.78],["TO",1.6],["AC",0.9],["AM",0.85],["RR",0.61],["Brazil",210.2]]
população = np.array(população)

mypath = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/data_mensured"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#for i in onlyfiles:
#    FILE = i
#    for i in população:
#        if i[0] == FILE[9:-4]:
#            pop = float(i[1])
#            
#    Sensitivity(mypath,FILE,pop)

extrapolação = 365
day_0 = '2020-02-26'
variavel = 'Cases' 

R = []
estados = [ ]
for i in onlyfiles:
    FILE = i
    for i in população:
        if i[0] == FILE[9:-4]:
            pop = float(i[1])
    estados.append(FILE[9:-4])
    R.append(Ajust_SUCQ(FILE,pop,extrapolação,day_0,variavel,pasta = mypath))       
R = np.array(R)
estados = np.array(estados)
df_R = pd.DataFrame(R, columns = ['r','p','K','alfa','NSE'])
df_R["Estado"] = estados

def bar_plt(atributo, title_name,df_R,logscale):
    fig, ax = plt.subplots(1, 1)
    df_R = df_R.sort_values(by=[atributo])

    figure = df_R.plot.bar(ax =ax, x = "Estado", y =atributo,figsize = (15,8), legend = None,width=.75, logy = logscale)
    figure.set_xlabel(" ")
    figure.set_title(title_name, family = "Serif", fontsize = 22)
    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.yaxis.set_major_formatter(plt.FuncFormatter(format_func)) 

    for p in ax.patches:
        b = p.get_bbox()
        val = format_func(b.y1 + b.y0,1)        
        ax.annotate(val, ((b.x0 + b.x1)/2, b.y1 +0.25/100), fontsize = 14,ha='center', va='top',rotation = 90)

    plt.show()
    path_out ="C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/imagens/"
    fig.savefig(path_out+atributo+'_barplot.png', dpi = 300,bbox_inches='tight',transparent = True)

bar_plt(atributo = "r", title_name = "Growth rate at the early stage", df_R = df_R, logscale = False)

bar_plt(atributo = "p", title_name = "Parameter p", df_R = df_R, logscale = False)

bar_plt(atributo = "K", title_name = "Final epidemic size", df_R = df_R, logscale = True)

bar_plt(atributo = "alfa", title_name = "Parameter alfa", df_R = df_R, logscale = True)
