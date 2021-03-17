#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:32:24 2019

This file contains the 8th experiment set of the SCN model project.
Input-Output relation of apical, basal and apical+basal synapses.

@author: kuenzel
"""
#
path_on_machine = '/home/kuenzel/Documents/'
cores_on_machine = 4
machine_has_display = True
#
#
from multiprocessing import Pool
import numpy as np
import time
import sys
import os
sys.path.append(path_on_machine + 'Python/tools')
sys.path.append(path_on_machine + 'Python/models')
sys.path.append(path_on_machine + 'Python/pxko')
import matplotlib
if not machine_has_display:
    matplotlib.use('Agg')
import tk_tools as tkt
import smallexc as se
import pandas as pd
import matplotlib.pyplot as plt
#se.load_mechanisms()
import itertools
import scipy.ndimage as ndimage
from progress.bar import ChargingBar

def load_mechanisms():
    import Neuron_Models as nm
    nm.h('nrn_load_dll("/home/kuenzel/Documents/Python/SCN/mech/x86_64/.libs/libnrnmech.so")')

def runmodel(tstop=1000,dt=0.025,a_on=True, b_on=True,seed=32768,
             inpn=(10,10),starta=0.0,startb=0.0,syntau=2,synw_a=0.002,synw_b=0.002,
             itv_a=20,itv_b=20):
    '''
    Creates and runs the model
    ''' 
    
    #-----IMPORTS--------
    from neuron import h
    #load_mechanisms()

    pretime = 0

    T = 21
    
    #CREATE CONDUCTANCE TRACES
    #E_temp = tkt.g_template('e',2,dt,0.15,0.2,geval)
    #gvE=[]
    #for gE in range(Nsyn):
    #    thisg = tkt.make_g_trace()
    #    gvE.append(h.Vector(thisg))
    
    #--------MODEL DEFINITIONS---------
    #CREATE SECTIONS
    Soma = h.Section(name='SO')
    AxonInit = h.Section(name='AI')
    Internode1 = h.Section(name='I1')
    Internode2 = h.Section(name='I2')
    Node1 = h.Section(name='N1')
    Node2 = h.Section(name='N2')
    bDend = h.Section(name='BD')
    bproxDend = h.Section(name='BP')
    pNeurit = h.Section(name='NR')
    aDend = h.Section(name='AD')#
    
    #Geometry & Biophysics
    bDend.L = 175#from tableq
    bDend.diam = 2#assumed, averaged over length
    bDend.nseg = 175
    #
    bproxDend.L = 25#inferred
    bproxDend.diam = 2#assumed
    bproxDend.nseg = 25
    #
    Soma.L = 20#from table
    Soma.diam = 20#assumed'5d
    Soma.nseg = 7
    #
    pNeurit.L = 45#from table
    pNeurit.diam = 3#assumed
    pNeurit.nseg = 7
    #
    AxonInit.L = 50#from table
    AxonInit.diam = 3#assumed
    AxonInit.nseg = 9
    #
    Internode1.L = 1000
    Internode1.diam = 2
    Internode1.nseg = 51
    Internode2.L = 1000
    Internode2.diam = 2
    Internode2.nseg = 51
    Node1.L = 3
    Node1.diam = 2
    Node1.nseg = 3
    Node2.L = 3
    Node2.diam = 2
    Node2.nseg = 3
    #
    aDend.L = 230#from table
    aDend.diam = 2
    aDend.nseg = 21
    #
    #axial Resistance and cm
    for sec in h.allsec():
        sec.Ra = 150
        sec.insert('extracellular')
    Internode1.cm = 0.01
    Internode2.cm = 0.01
    #
    #hh
    Soma.insert('hh')
    Soma.ena = 50
    Soma.gnabar_hh = 0.12
    Soma.gkbar_hh = 0.023
    AxonInit.insert('hh')
    AxonInit.ena = 50
    AxonInit.gnabar_hh = 0.12
    AxonInit.gkbar_hh = 0.023
    pNeurit.insert('hh')
    pNeurit.ena = 50
    pNeurit.gnabar_hh = 0.12
    pNeurit.gkbar_hh = 0.023
    #
    Node1.insert('hh')
    Node1.gnabar_hh = 0.36
    Node2.insert('hh')
    Node2.gnabar_hh = 0.36
    Internode1.insert('leak')
    Internode1.g_leak = 1e-6    
    Internode2.insert('leak')
    Internode2.g_leak = 1e-6 
    #
    #kht
    Soma.insert('kht')
    Soma.gkhtbar_kht = 0.013
    Soma.ek = -80
    pNeurit.insert('kht')
    pNeurit.gkhtbar_kht = 0.013
    pNeurit.ek = -80
    AxonInit.insert('kht')
    AxonInit.gkhtbar_kht = 0.013
    AxonInit.ek = -80
    bproxDend.insert('kht')
    bproxDend.gkhtbar_kht = 0.013
    bproxDend.ek = -80
    #
    #leak
    aDend.insert('leak')
    aDend.g_leak = 0.0001    
    bDend.insert('leak')
    bDend.g_leak = 0.0001    
    bproxDend.insert('leak')
    bproxDend.g_leak = 0.0001
    #
    #Topology
    Node2.connect(Internode2(1))
    Internode2.connect(Node1(1))
    Node1.connect(Internode1(1))
    Internode1.connect(AxonInit(1))
    AxonInit.connect(pNeurit(1))
    aDend.connect(pNeurit(1))
    pNeurit.connect(Soma(1))
    bproxDend.connect(Soma(0))
    bDend.connect(bproxDend(1))

    # GENERAL SETTINGS
    h.dt = dt  # simulation (or "sampling") rate
    h.celsius = T  # simulation global temperature

    # MODEL INSTRUMENTATION & INPUT
    #Electrode = h.IClamp(Soma(0.5))
    #Electrode.delay = 10
    #Electrode.amp =0.2
    #Electrode.dur = 50
    #meaninterval=20
    anc = []
    asyn = []
    astim = []
    atv = []
    #if a_on:
    #    if b_on:
    #        Na = 10
    #    else:
    #        Na = 20
    #else:
    #    Na = 1
    for iasyn in range(inpn[0]):
        astim.append(h.NetStim())
        asyn.append( h.ExpSyn(aDend(0.75+(iasyn/100.0))) )
        atv.append(h.Vector())
        asyn[iasyn].tau = syntau
        asyn[iasyn].e = 0
        astim[iasyn].number = (tstop/itv_a)+20
        astim[iasyn].start = starta
        astim[iasyn].noise = 1
        astim[iasyn].interval = itv_a
        astim[iasyn].seed(seed)
        anc.append(h.NetCon(astim[iasyn],asyn[iasyn]))
        anc[iasyn].record(atv[iasyn])
        anc[iasyn].delay=0.5
        if a_on:
            #anc[iasyn].weight[0]=np.abs(np.random.normal(1))*0.002
            anc[iasyn].weight[0]=synw_a
        else:
            anc[iasyn].weight[0]=0

    bstim=[]
    bsyn=[]
    bnc=[]
    btv=[]
    #if b_on:
    #    if a_on:
    #        Nb = 10
    #    else:
    #        Nb = 20
    #else:
    #    Nb = 1
    for ibsyn in range(inpn[1]):
        bstim.append(h.NetStim())
        bsyn.append( h.ExpSyn(bDend(0.75+(ibsyn/100.0))) )
        btv.append(h.Vector())
        bsyn[ibsyn].tau = syntau
        bsyn[ibsyn].e = 0
        bstim[ibsyn].number = (tstop/itv_b)+20
        bstim[ibsyn].start = startb
        bstim[ibsyn].noise = 1
        bstim[ibsyn].interval = itv_b
        bstim[ibsyn].seed(seed)
        bnc.append(h.NetCon(bstim[ibsyn],bsyn[ibsyn]))
        bnc[ibsyn].record(btv[ibsyn])
        bnc[ibsyn].delay=0.5
        if b_on:
            #bnc[ibsyn].weight[0]=np.abs(np.random.normal(1))*0.002
            bnc[ibsyn].weight[0]=synw_b
        else:
            bnc[ibsyn].weight[0]=0
    #print 'Na=' + str(inpn[0]) + ' + Nb=' + str(inpn[1])#debug
        
    # INFRASTRUCTURE
    SO = h.Vector()
    SO.record(Soma(0.01)._ref_v)
    PA = h.Vector()
    PA.record(pNeurit(0.5)._ref_v)
    PB = h.Vector()
    PB.record(bproxDend(0.5)._ref_v)
    AI = h.Vector()
    AI.record(AxonInit(0.5)._ref_v)  
    DA = h.Vector()
    DA.record(aDend(0.5)._ref_v)
    DB = h.Vector()
    DB.record(bDend(0.5)._ref_v)
    N1 = h.Vector()
    N1.record(Node1(0.5)._ref_v)
    N2= h.Vector()
    N2.record(Node2(0.5)._ref_v)
    
    #----------SIMULATE--------------
    tkt.nrngo(tstop+pretime,-66.5)
    
    # PACK AND EXPORT DATA
    Result = {}
    tSO = np.array(SO)
    tSO = tSO[int(pretime/dt):int((tstop+pretime)/dt)]
    tPA = np.array(PA)
    tPA = tPA[int(pretime/dt):int((tstop+pretime)/dt)]
    tPB = np.array(PB)
    tPB = tPB[int(pretime/dt):int((tstop+pretime)/dt)]
    tDA = np.array(DA)
    tDA = tDA[int(pretime/dt):int((tstop+pretime)/dt)]
    tDB = np.array(DB)
    tDB = tDB[int(pretime/dt):int((tstop+pretime)/dt)]
    tAI = np.array(AI)
    tAI = tAI[int(pretime/dt):int((tstop+pretime)/dt)]
    tN1 = np.array(N1)
    tN1 = tN1[int(pretime/dt):int((tstop+pretime)/dt)]
    tN2 = np.array(N2)
    tN2 = tN2[int(pretime/dt):int((tstop+pretime)/dt)]
    #
    Result['SO']=tSO
    Result['PA']=tPA
    Result['PB']=tPB
    Result['DA']=tDA
    Result['DB']=tDB
    Result['AI']=tAI
    Result['N1']=tN1
    Result['N2']=tN2
    # 
    return(Result)
 
if __name__ == '__main__':
    tstop = 250
    dt = 0.025
    rep = 10
    allweights = np.linspace(0.0009,0.003,36)
    bar = ChargingBar('Simulation running', max=rep*len(allweights)*3 )
    if int(sys.argv[1])==1:#do experiment
        Aspikerate_b=np.zeros((len(allweights),rep))
        Sspikerate_b=np.zeros((len(allweights),rep))
        Aspikerate_a=np.zeros((len(allweights),rep))
        Sspikerate_a=np.zeros((len(allweights),rep))
        Aspikerate_ab=np.zeros((len(allweights),rep))
        Sspikerate_ab=np.zeros((len(allweights),rep))
        res={}
        for icond,thissynw in enumerate(allweights):
            rAspikerate_b=[]
            rSspikerate_b=[]
            rAspikerate_a=[]
            rSspikerate_a=[]
            rAspikerate_ab=[]
            rSspikerate_ab=[]
            for thisrep in range(rep):
                bar.next()
                Rb=runmodel(tstop=tstop,dt=dt,a_on=False, b_on=True,seed=3443268+(thisrep*100)+icond,
                           inpn=(1,20),starta=0.0,startb=0.0,syntau=2,synw_a=0.002,synw_b=thissynw,
                           itv_a=20,itv_b=20)
                Aspt_b = tkt.SimpleDetectAP(Rb['N2'],dt=dt,thr=-50,LM=-30)
                rAspikerate_b.append(len(Aspt_b['PeakT'])*(1000.0/tstop))
                Sspt_b = tkt.SimpleDetectAP(Rb['SO'],dt=dt,thr=-30,LM=-30)
                rSspikerate_b.append(len(Sspt_b['PeakT'])*(1000.0/tstop))
                #
                bar.next()
                Ra=runmodel(tstop=tstop,dt=dt,a_on=True, b_on=False,seed=3443268+(thisrep*100)+icond,
                           inpn=(20,1),starta=0.0,startb=0.0,syntau=2,synw_a=thissynw,synw_b=0.002,
                           itv_a=20,itv_b=20)
                Aspt_a = tkt.SimpleDetectAP(Ra['N2'],dt=dt,thr=-50,LM=-30)
                rAspikerate_a.append(len(Aspt_a['PeakT'])*(1000.0/tstop))
                Sspt_a = tkt.SimpleDetectAP(Ra['SO'],dt=dt,thr=-30,LM=-30)
                rSspikerate_a.append(len(Sspt_a['PeakT'])*(1000.0/tstop))
                #
                bar.next()
                Rab=runmodel(tstop=tstop,dt=dt,a_on=True, b_on=True,seed=3443268+(thisrep*100)+icond,
                           inpn=(10,10),starta=0.0,startb=0.0,syntau=2,synw_a=thissynw,synw_b=thissynw,
                           itv_a=20,itv_b=20)
                Aspt_ab = tkt.SimpleDetectAP(Rab['N2'],dt=dt,thr=-50,LM=-30)
                rAspikerate_ab.append(len(Aspt_ab['PeakT'])*(1000.0/tstop))
                Sspt_ab = tkt.SimpleDetectAP(Rab['SO'],dt=dt,thr=-30,LM=-30)
                rSspikerate_ab.append(len(Sspt_ab['PeakT'])*(1000.0/tstop))
            Aspikerate_b[icond,:]=rAspikerate_b
            Sspikerate_b[icond,:]=rSspikerate_b
            Aspikerate_a[icond,:]=rAspikerate_a
            Sspikerate_a[icond,:]=rSspikerate_a
            Aspikerate_ab[icond,:]=rAspikerate_ab
            Sspikerate_ab[icond,:]=rSspikerate_ab
        res['weights']=allweights
        res['Aspikerate_b']=Aspikerate_b
        res['Sspikerate_b']=Sspikerate_b
        res['Aspikerate_a']=Aspikerate_a
        res['Sspikerate_a']=Sspikerate_a
        res['Aspikerate_ab']=Aspikerate_ab
        res['Sspikerate_ab']=Sspikerate_ab
        np.save(path_on_machine + 'Python/SCN/scnres8.npy',res)
        bar.finish()
    else:
        try:
            res = np.load(path_on_machine + 'Python/SCN/scnres8.npy')
            res = res.tolist()
        except:
            print sys.exit('Data not found yet, use 1 as second argument!')
    
    #------plot part for exp8 begins here-----------
    plt.figure(figsize=(6,3.5))
    plt.suptitle('Axon/Soma Spikerate vs. Synaptic Conductance')
    ax1=plt.subplot(121)
    plt.vlines(0.002*1000.0*20.0,0,140,color='k',linestyle=':')
    plt.title('Axon Spikes')
    plt.ylabel('Spikerate (AP/s)')
    plt.xlabel('Total synaptic input g (nS)')
    plt.errorbar(res['weights']*1000.0*20.0,np.mean(res['Aspikerate_a'],axis=1),yerr=np.std(res['Aspikerate_a'],axis=1),color='b',marker='o')
    plt.errorbar(res['weights']*1000.0*20.0,np.mean(res['Aspikerate_ab'],axis=1),yerr=np.std(res['Aspikerate_ab'],axis=1),color='k',marker='o')
    plt.errorbar(res['weights']*1000.0*20.0,np.mean(res['Aspikerate_b'],axis=1),yerr=np.std(res['Aspikerate_b'],axis=1),color='r',marker='o')
    plt.legend(('20 apical only','10 apical + 10 basal','20 basal only'),loc=2)
    ax2=plt.subplot(122)
    plt.vlines(0.002*1000.0*20.0,0,140,color='k',linestyle=':')
    plt.title('Soma Spikes')
    plt.xlabel('Total synaptic input g (nS)')
    plt.errorbar(res['weights']*1000.0*20.0,np.mean(res['Sspikerate_a'],axis=1),yerr=np.std(res['Sspikerate_a'],axis=1),color='b',marker='o')
    plt.errorbar(res['weights']*1000.0*20.0,np.mean(res['Sspikerate_ab'],axis=1),yerr=np.std(res['Sspikerate_ab'],axis=1),color='k',marker='o')
    plt.errorbar(res['weights']*1000.0*20.0,np.mean(res['Sspikerate_b'],axis=1),yerr=np.std(res['Sspikerate_b'],axis=1),color='r',marker='o')
    plt.tight_layout()    
    plt.show()
