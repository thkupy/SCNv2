#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:01:38 2019

This file contains the 7th experiment set of the SCN model project. It is supposed to deal
with aspects of what determines the appearance of somatic spikes

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
    tstop = 10000
    dt = 0.01
    if int(sys.argv[1])==1:#do experiment
        R=runmodel(tstop=tstop,dt=dt,a_on=True, b_on=True,seed=3443268,
             inpn=(10,10),starta=0.0,startb=0.0,syntau=2,synw_a=0.002,synw_b=0.002,
             itv_a=20,itv_b=20)
        np.save(path_on_machine + 'Python/SCN/scnres7.npy',R)
    else:
        try:
            R = np.load(path_on_machine + 'Python/SCN/scnres7.npy')
            R = R.tolist()
        except:
            print sys.exit('Data not found yet, use 1 as second argument!')
    
    #------plot part for exp7 begins here-----------
    Aspt = tkt.SimpleDetectAP(R['N2'],dt=dt,thr=-50,LM=-30)
    Sspt = tkt.SimpleDetectAP(R['SO'],dt=dt,thr=-30,LM=-30)
    tax = np.linspace(0,tstop-dt,tstop/dt)
    #
    soax=plt.subplot(5,4,9)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['SO'],'m')
    plt.ylabel('Vm (mV)')
    plt.title('Soma')
    plt.legend(('Vm','Soma spt.','Axon spt.'))

    plt.subplot(5,4,5,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['PA'],'g--')
    plt.ylabel('Vm (mV)')
    plt.title('Neurite')
    
    plt.subplot(5,4,13,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['PB'],'r--')
    plt.ylabel('Vm (mV)')
    plt.title('Basal proximal dendrite')
        
    plt.subplot(5,4,1,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['DA'],'g')
    plt.ylabel('Vm (mV)')
    plt.title('Apical dendrite')
           
    plt.subplot(5,4,17,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['DB'],'r')
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (ms)')
    plt.title('Basal distal dendrite')
        
    plt.subplot(5,4,6,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['AI'],'b')
    plt.title('Axon initial segment')
    plt.subplot(5,4,7,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['N1'],'k--')
    plt.xlabel('Time (ms)')
    plt.title('1st node')
    plt.subplot(5,4,8,sharex=soax,sharey=soax)
    plt.vlines(Sspt['PeakT'],-80,40,color='r',linestyle='-')
    plt.vlines(Aspt['PeakT'],-80,40,color='k',linestyle=':')
    plt.plot(tax,R['N2'],'k')
    plt.xlabel('Time (ms)')
    plt.title('2nd node')
    plt.ylim((-80,30))
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.85, wspace=0.15, hspace=0.35)
    plt.suptitle('Membrane potential in different compartments')
    
    #Ansatz 1: average potential waveform before Axon- and Somaspike
    structures=('DA','PA','SO','PB','DB','AI','N1','N2')
    titles=('distal, apical','Neurit','Soma','proximal, basal','distal, basal','Initialsegment','Node1','Node2')
    for N_str,structure in enumerate(structures):
        plt.figure()
        plt.suptitle(titles[N_str])
        ax1=plt.subplot(121)
        plt.xlabel('Time re axon-spike (ms)')
        plt.ylabel('Vm (mV)')
        plt.title('Membrane Potential')
        ax2=plt.subplot(122)
        plt.xlabel('Vm (mV)')
        plt.ylabel('dV/dt (mV/ms)')
        plt.title('Phase plane')
        plt.xlim((-70,-40))
        plt.ylim((-10,100))
        snipdur = 4.0 #ms
        #structure = 'SO'
        nSnipSamples = int(round(snipdur/dt))
        nAxEv = len(Aspt['PeakT'])
        nSoEv = len(Sspt['PeakT'])
        wvfpreAxEv = np.zeros((nAxEv,nSnipSamples))
        wvfpreSoEv = np.zeros((nSoEv,nSnipSamples))
        avVmpreAxEv=[]
        peakVmAxEv=[]
        avVmpreSoEv=[]
        peakVmSoEv=[]
        for iev, evt in enumerate(Aspt['PeakT']):
            if np.any(np.abs(Sspt['PeakT']-evt)<0.5):
                continue
            loctax = np.linspace(-snipdur+dt,0,nSnipSamples)
            thisEndIndex = int(round(evt/dt))
            thisStartIndex = thisEndIndex-nSnipSamples
            if thisStartIndex<0:
                thisStartIndex = 0
                shortsnip = int(round(evt/dt))
                loctax = np.linspace(-evt+dt,0,shortsnip)
            ax1.plot(loctax,R[structure][thisStartIndex:thisEndIndex],color=(.8,.8,.8))
            ax2.plot(R[structure][thisStartIndex+1:thisEndIndex],np.diff(R[structure][thisStartIndex:thisEndIndex])*(1/dt),color=(.8,.8,.8))
            wvfpreAxEv[iev,:]=R[structure][thisStartIndex:thisEndIndex]
            avVmpreAxEv.append(np.mean(R[structure][thisStartIndex:thisEndIndex-20]))
            peakVmAxEv.append(Aspt['PeakV'][iev])
            
        for iev, evt in enumerate(Sspt['PeakT']):
            loctax = np.linspace(-snipdur+dt,0,nSnipSamples)
            thisEndIndex = int(round(evt/dt))
            thisStartIndex = thisEndIndex-nSnipSamples
            if thisStartIndex<0:
                thisStartIndex = 0
                shortsnip = int(round(evt/dt))
                loctax = np.linspace(-evt+dt,0,shortsnip)
            ax1.plot(loctax,R[structure][thisStartIndex:thisEndIndex],color=(0,.8,.8))
            ax2.plot(R[structure][thisStartIndex+1:thisEndIndex],np.diff(R[structure][thisStartIndex:thisEndIndex])*(1/dt),color=(0,.8,.8))
            wvfpreSoEv[iev,:]=R[structure][thisStartIndex:thisEndIndex]
            avVmpreSoEv.append(np.mean(R[structure][thisStartIndex:thisEndIndex-20]))
            peakVmSoEv.append(Sspt['PeakV'][iev])
        
        
        wvfpreAxEv=wvfpreAxEv[np.mean(wvfpreAxEv,axis=1)!=0]#remove those lines that are zeros only (Somaspiketimes)
        ax1.plot(loctax,np.mean(wvfpreAxEv,axis=0),'k-')
        ax1.plot(loctax,np.mean(wvfpreAxEv,axis=0)+np.std(wvfpreAxEv,axis=0),'k:')
        ax1.plot(loctax,np.mean(wvfpreAxEv,axis=0)-np.std(wvfpreAxEv,axis=0),'k:')
        meanVmAx = np.mean(wvfpreAxEv,axis=0)
        ax2.plot(meanVmAx[1:],np.diff(meanVmAx)*(1/dt),'k-')
    
        ax1.plot(loctax,np.mean(wvfpreSoEv,axis=0),'r-')
        ax1.plot(loctax,np.mean(wvfpreSoEv,axis=0)+np.std(wvfpreSoEv,axis=0),'r:')
        ax1.plot(loctax,np.mean(wvfpreSoEv,axis=0)-np.std(wvfpreSoEv,axis=0),'r:')
        meanVmSo = np.mean(wvfpreSoEv,axis=0)
        ax2.plot(meanVmSo[1:],np.diff(meanVmSo)*(1/dt),'r-')
        
        plt.tight_layout()
        ##AP Slope vs. membrane potential (too unclear at the moment)
#        plt.figure()
#        plt.suptitle(titles[N_str])
#        plt.plot(avVmpreAxEv,peakVmAxEv,'k.')
#        plt.plot(avVmpreSoEv,peakVmSoEv,'r.')
#        plt.xlabel('Average Vm before event (mV)')
#        plt.ylabel('Event Peak Voltage (mV)')
    plt.show()