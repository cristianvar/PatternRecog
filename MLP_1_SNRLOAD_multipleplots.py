# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:48:17 2022

@author: Titan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:09:45 2022

@author: crist
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


sns.set_theme(style="darkgrid")

#Model 1 completed

#files to convert symbols into bit sequences. The files contain a matrix showing the number of bits that
#differ from one symbol over the others 64, 50, 46,41, 33, and 26 constellation symbol.
#change it everytime the constellation size change. 
#use the file called BER_dicctionary.py to produce different size matrix depending on the symbol constellation

dic36= r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\36BER_DIC.csv'
dicc = pd.read_csv(dic36,header=None)
#Machine Learning model file path
filename = 'MLP_1_CV_10s.pkl'



data1 = r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\Validation_data_sim.csv'
data2 = r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\Validation_data_exp.csv'
data3 = r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\simulation_10_50.csv'

training = pd.read_csv(data1)
#print(training.describe().transpose())
##############################
#Data filtering for negative values and per signal power

Xt_n = training.iloc[:,[0,1,2,3,4,5]]
Xt_n.columns = ['f1','f2','f3','symbol','power','gain']
#Xt = Xt_n[(Xt_n.f1>0)&(Xt_n.f2>0)&(Xt_n.f3>0)]
Xt = Xt_n[(Xt_n.power>=20)&(Xt_n.power<=50)]
print(Xt.describe().transpose())
xtt=Xt



bertotal=[]
acctotal=[]
snr =np.array(np.arange(20,51))
start_time = time.time()
for a in snr:
    Xt=xtt
    Xt =Xt[(Xt.power==a)]
    
    #print(Xt.head())
    y_train = Xt.iloc[:,3]
    Xt = Xt.iloc[:,[0,1,2]]
    #print(Xt.head())

    Yt = y_train.astype({'symbol':int})


    sc=StandardScaler()
    scaler = sc.fit(Xt)
    data_scaled = scaler.transform(Xt)
    loaded_model = joblib.load(filename)



    loaded_model
    result = loaded_model.score(data_scaled, Yt)
    prediction = loaded_model.predict(data_scaled)
    

    t= confusion_matrix(Yt, prediction)
    nbits= Xt.shape[0]*6
    t= confusion_matrix(Yt, prediction)
    tber=np.multiply(t, dicc)
    suma = tber.sum()
    suma = suma.sum()
    ber=suma/nbits
    
    print('SNR= ',a)
    print('accuracy= ',result)
    print('ber= ',ber)
    
    acctotal.append(result)
    bertotal.append(ber)
print("--- %s seconds ---" % (time.time() - start_time))    
   
#plot of accuracy
plt.figure()
Ann = sns.scatterplot( x=snr,y=bertotal,color='black')
Ann.set_xlabel("SNR dB", fontsize = 15)
Ann.set_ylabel("BER", fontsize = 15)
Ann.set_title('Model 11 ANN BER simulated \n Data range 10 to 50 SNR. Symbols constellation 36')
ax = plt.gca()
#Ann.set_ylim([0.0000001,0.003])
#plt.savefig('Model #11 ANNBERs.jpg', format='jpeg', dpi=500)


#plot of BER
plt.figure()
Annb = sns.scatterplot( x=snr,y=acctotal,color='black')
Annb.set_xlabel("SNR dB", fontsize = 15)
Annb.set_ylabel("Accuracy", fontsize = 15)
Annb.set_title('Model 11 ANN accuracy simulated \n Data range 10 to 50 SNR. Symbols constellation 36')


#plt.savefig('Model #11 ANNACCs.jpg', format='jpeg', dpi=500)



