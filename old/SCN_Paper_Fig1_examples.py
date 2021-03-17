# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os
sys.path.append('/home/kuenzel/Documents/Python/tools')
sys.path.append('/home/kuenzel/Documents/Python/models')
sys.path.append('/home/kuenzel/Documents/Python/pxko')
import matplotlib.pyplot as plt
import numpy as np
import tk_tools as tkt
from datetime import datetime
from scipy import stats

R = np.load('/home/kuenzel/Dokumente/Python/SCN/Example_Traces.npy')
dt=0.01
taxis=np.linspace(0,1000.0-dt,int(round(1000.0/dt)))
#R[0]=is 10/10

btv_0=[]
atv_0=[]
for iii in range(10):
    nsp = len(R[0]['atv'][iii])
    for kkk in range(nsp):
        atv_0.append(R[0]['atv'][iii][kkk])
    nsp = len(R[0]['btv'][iii])
    for kkk in range(nsp):
        btv_0.append(R[0]['btv'][iii][kkk])
ev=tkt.SimpleDetectAP(R[0]['AVm'],dt=dt,thr=-50,LM=-30)
ha,edges=np.histogram(atv_0,100)
hb,edges=np.histogram(btv_0,100)
plt.subplot(311)
plt.plot(ev['PeakT'],np.ones(len(ev['PeakT']))*40,'g|')
plt.plot(taxis,R[0]['AVm'],'k-')
plt.plot(np.concatenate((edges,[1000])),np.concatenate(([-100],(ha+hb)-100,[-100])),'k',linewidth=2)
plt.title('NA=10,NB=10, ' + str(len(ev['PeakT'])) + 'sp/s')

atv_1=[]
for iii in range(20):
    nsp = len(R[1]['atv'][iii])
    for kkk in range(nsp):
        atv_1.append(R[1]['atv'][iii][kkk])
ev=tkt.SimpleDetectAP(R[1]['AVm'],dt=dt,thr=-50,LM=-30)
ha1,edges=np.histogram(atv_1,100)
plt.subplot(312)
plt.plot(ev['PeakT'],np.ones(len(ev['PeakT']))*40,'g|')
plt.plot(taxis,R[1]['AVm'],'b-')
plt.plot(np.concatenate((edges,[1000])),np.concatenate(([-100],(ha+hb)-100,[-100])),'k')
plt.plot(np.concatenate((edges,[1000])),np.concatenate(([-100],ha1-100,[-100])),'b',linewidth=2)
plt.title('NA=20,NB=0, ' + str(len(ev['PeakT'])) + 'sp/s')

btv_2=[]
for iii in range(20):
    nsp = len(R[2]['btv'][iii])
    for kkk in range(nsp):
        btv_2.append(R[2]['btv'][iii][kkk])
ev=tkt.SimpleDetectAP(R[2]['AVm'],dt=dt,thr=-50,LM=-30)
hb2,edges=np.histogram(btv_2,100)
plt.subplot(313)
plt.plot(ev['PeakT'],np.ones(len(ev['PeakT']))*40,'g|')
plt.plot(taxis,R[2]['AVm'],'r-')
plt.plot(np.concatenate((edges,[1000])),np.concatenate(([-100],ha1-100,[-100])),'b')
plt.plot(np.concatenate((edges,[1000])),np.concatenate(([-100],(ha+hb)-100,[-100])),'k')
plt.plot(np.concatenate((edges,[1000])),np.concatenate(([-100],hb2-100,[-100])),'r',linewidth=2)
plt.title('NA=0,NB=20, ' + str(len(ev['PeakT'])) + 'sp/s')
plt.tight_layout()
plt.show()