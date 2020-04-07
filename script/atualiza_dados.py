import pandas as pd 
import numpy as np
import requests 

def baixar_arquivo(url, endereco):
    resposta = requests.get(url)
    if resposta.status_code == requests.codes.OK:
        with open(endereco, 'wb') as novo_arquivo:
                novo_arquivo.write(resposta.content)
        print("Download finalizado. Arquivo salvo em: {}".format(endereco))
    else:
        resposta.raise_for_status()

pasta = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/att.csv"
link = 'https://mobileapps.saude.gov.br/esus-vepi/files/unAFkcaNDeXajurGB7LChj8SgQYS2ptm/1f2e9efc2bdd487d4f3b693467aeb925_Download_COVID19_20200406.csv'
baixar_arquivo(link,pasta)
        
df = pd.read_csv(pasta,header = 0, sep = ";")
data = df.data
path_out = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/Richards_covid19/data/data_mensured/"

nome_estado = 'estado'
A=pd.pivot_table(df, values = 'casosNovos', 
                 columns = [nome_estado],
                 aggfunc=np.sum)
estados = A.columns

length = len(df)-1

nome_data = 'data'
day_0_str=df[nome_data][0][-4:]+'-'+df[nome_data][0][3:5]+'-'+df[nome_data][0][:2]
day_last_str=df[nome_data][length][-4:]+'-'+df[nome_data][length][3:5]+'-'+df[nome_data][length][:2]

day_0 = np.array(day_0_str, dtype=np.datetime64)
day_last = np.array(day_last_str, dtype=np.datetime64)

n=0
delta_day = day_last - day_0
delta_day = delta_day.astype(int)+1

for i in range(27):
    df_estado = df[n:n+delta_day]
    n = delta_day*(i+1)
    
    f_case = 0
    while(df_estado["casosAcumulados"].values[f_case]==0):
        f_case = f_case+1
    
    df_estado =df_estado[f_case:]
    
    s_e = df_estado.estado.values[0]
    
    df_estado.columns = ['regiao','estado',	'DateRep', 'Cases','cum-Cases','Deaths','cum-Deaths']
    df_estado.to_csv(path_out+ "COVID-19 "+ s_e + ".csv", sep = ";",index = False)

Serie = []
for j in range(27):
    for i in range (delta_day):
     Serie.append(i)  

df["ordem"] = Serie

df_brasil = pd.pivot_table(df, values = ["casosNovos","casosAcumulados","obitosNovos","obitosAcumulados"], index = "ordem", aggfunc = np.sum)
df_brasil['DateRep'] = df[nome_data][:delta_day] 

f_case = 0
while(df_brasil["casosAcumulados"].values[f_case]==0):
    f_case = f_case+1

df_brasil = df_brasil[f_case:] 
df_brasil.columns = ['cum-Cases','Cases','cum-Deaths','Deaths','DateRep']
df_brasil.to_csv(path_out+ "COVID-19 "+ "Brasil" + ".csv", sep = ";",index = False)
   
