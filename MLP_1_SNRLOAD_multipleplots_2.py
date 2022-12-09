# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:30:53 2022

@author: Titan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:19:38 2022

@author: Titan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:35:17 2022

@author: Titan
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")


BER=r'C:\Users\crist\OneDrive\Escritorio\PatternRecog-main\PatternRecog-main\BER_simulation_MODELS_LOG10.csv'




ber = pd.read_csv(BER,header=None)


snr = np.array(np.arange(25,31))

n=5
m=-20

ber= ber.to_numpy()




plt.figure()
Annb = sns.lineplot( x=snr,y=ber[n:m,1],color='black',label='Trained range: 10-50')
Annb = sns.lineplot( x=snr,y=ber[n:m,2],color='yellow',label='Trained range: 15-50')
Annb = sns.lineplot( x=snr,y=ber[n:m,3],color='red',label='Trained range: 20-50',marker='H')
Annb = sns.lineplot( x=snr,y=ber[n:m,4],color='cyan',label='Trained range: 25-30')
Annb = sns.lineplot( x=snr,y=ber[n:m,4],color='purple',label='Trained range: 30-35')
Annb = sns.lineplot( x=snr,y=-3,color='blue',label='FEC Limit',marker="D")
Annb.set_xlabel("SNR dB", fontsize = 15)
Annb.set_ylabel("BER (Log 10)", fontsize = 15)
Annb.set_title('BER vs SNR in ANN ',fontsize = 16)
#plt.ylim(0.0000001, 0.03)
# plt.xlim(27, 34)

plt.savefig('Model best.png', format='jpeg', dpi=500,bbox_inches='tight')