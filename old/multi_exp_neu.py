#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on tue mar 10 2020
This is the program that generates the 
"Lead Time diff" vs. "Active Synapse count diff" 2D plots for the SCN project
#----->however this is the corrected version
@author: kuenzel
"""
#
path_on_machine = '/home/kuenzel/Dokumente/'
cores_on_machine = 20
machine_has_display = True
dt = 0.025
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
from matplotlib import ticker

def load_mechanisms():
    import Neuron_Models as nm
    nm.h('nrn_load_dll("/home/kuenzel/Documents/Python/SCN/mech/x86_64/.libs/libnrnmech.so")')

def runmodel(tstop=1000,dt=0.025,a_on=True, b_on=True,seed=32768,
             inpn=(10,10),starta=200.0,startb=200.0,indura=600.0,indurb=600.0,
             itv_a=20,itv_b=20,syntau=2,synw=0.002):
    '''
    Creates and runs the model
    ''' 
    
    #-----IMPORTS--------
    from neuron import h
    #from neuron import gui
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
    Soma = h.Section()
    AxonInit = h.Section()
    Internode1 = h.Section()
    Internode2 = h.Section()
    Node1 = h.Section()
    Node2 = h.Section()
    bDend = h.Section()
    bproxDend = h.Section()
    pNeurit = h.Section()
    aDend = h.Section()#
    
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
    anc = []
    asyn = []
    astim = []
    atv = []
    for iasyn in range(inpn[0]):
        astim.append(h.NetStim())
        asyn.append( h.ExpSyn(aDend(0.75+(iasyn/100.0))) )
        atv.append(h.Vector())
        asyn[iasyn].tau = syntau
        asyn[iasyn].e = 0
        astim[iasyn].number = (indura/itv_a)
        astim[iasyn].start = starta
        astim[iasyn].noise = 1
        astim[iasyn].interval = itv_a
        astim[iasyn].seed(seed)
        anc.append(h.NetCon(astim[iasyn],asyn[iasyn]))
        anc[iasyn].record(atv[iasyn])
        anc[iasyn].delay=0.5
        if a_on:
            anc[iasyn].weight[0]=synw
        else:
            anc[iasyn].weight[0]=0

    bstim=[]
    bsyn=[]
    bnc=[]
    btv=[]
    for ibsyn in range(inpn[1]):
        bstim.append(h.NetStim())
        bsyn.append( h.ExpSyn(bDend(0.75+(ibsyn/100.0))) )
        btv.append(h.Vector())
        bsyn[ibsyn].tau = syntau
        bsyn[ibsyn].e = 0
        bstim[ibsyn].number = (indurb/itv_b)
        bstim[ibsyn].start = startb
        bstim[ibsyn].noise = 1
        bstim[ibsyn].interval = itv_b
        bstim[ibsyn].seed(seed)
        bnc.append(h.NetCon(bstim[ibsyn],bsyn[ibsyn]))
        bnc[ibsyn].record(btv[ibsyn])
        bnc[ibsyn].delay=0.5
        if b_on:
            bnc[ibsyn].weight[0]=synw
        else:
            bnc[ibsyn].weight[0]=0
        
    # INFRASTRUCTURE
    SCN_AVm = h.Vector()
    SCN_AVm.record(Node2(0.99)._ref_v)


    #----------SIMULATE--------------
    tkt.nrngo(tstop,-66.5)
    #EXPORT DATA
    return(np.array(SCN_AVm))

def myjob(myitems):
    tstop=700.0
    Na = myitems[0]
    Nb = 20-Na
    Onset = myitems[1]
    seed=32768+int(Na)+int(np.round(Onset)+250)+int(myitems[2])#seed = 32768+(0-20)+(0-500))+(0-10)
    R=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,
               inpn=(int(round(Na)),int(round(Nb))),
               starta=250.0+Onset,startb=250.0,seed=seed,
               indura=200.0,indurb=200.0)
    print('Succesful model run for process-ID ' + str(seed))
    spA=tkt.SimpleDetectAP(R,dt=dt,thr=-50,LM=-30)
    axsp=np.array(spA['PeakT'])
    multiplier = 1000.0/tstop
    Ax_count=len(axsp)
    Ax_rate=len(axsp)*multiplier
    res={}
    res['Ax_rate']=Ax_rate
    res['Ax_count']=Ax_count
    print('All done in Process-ID ' + str(seed))
#    thisbegin = 250.0
#    if thisbegin > Onset+250.0:
#        thisbegin = Onset+250.0
#    axsp=axsp[axsp>49.9]
#    thisend = 500
#    if thisend < Onset+500:
#        thisend = Onset+500
#    axsp=axsp[axsp < thisend]
#    multiplier = 1000.0/(thisend-thisbegin)
#    Ax_count=len(axsp)
#    Ax_rate=len(axsp)*multiplier
#    res={}
#    res['Ax_rate']=Ax_rate
#    res['Ax_count']=Ax_count
    return(res)
 
if __name__ == '__main__':
    if int(sys.argv[1])==1:#do experiment
        rep = 10
        represR = 21
        represO = 41
        middleR = 10
        middleO = 20
        Na = np.linspace(0,20,represR)
        Onset = np.linspace(-250,250,represO)
        Nrep = range(rep)
        runs = rep*represR*represO
        Nrun = range(runs)
        mylist = list(itertools.product(Na,Onset,Nrep))
        #
        start = time.time()
        pool = Pool(processes=cores_on_machine)
        AR=[]
        for oneline in mylist:
            AR.append(pool.apply_async(myjob,(oneline,)))
        pool.close()
        pool.join()
        #
        end = time.time()
        #
        print "Parallel execution time: " + str((end - start)) + " s"
        print "This is " + str((end-start)/len(mylist)) + "s per execution"
        #
        Ax_rate_raw = np.zeros((len(AR),))
        Ax_count_raw = np.zeros((len(AR),))
        for indi,thisR in enumerate(AR):
            UnpackedR=thisR.get()
            Ax_rate_raw[indi] = UnpackedR['Ax_rate']
            Ax_count_raw[indi] = UnpackedR['Ax_count']
        PL=np.array(mylist)
        Ax_rate=np.zeros((represR,represO))
        Ax_rate_s=np.zeros((represR,represO))
        Ax_count=np.zeros((represR,represO))
        Ax_count_s=np.zeros((represR,represO))
        for iR in range(represR):
            for iO in range(represO):
                theseRepInds = np.logical_and(PL[:,0]==Na[iR],PL[:,1]==Onset[iO])
                Ax_rate[iR,iO] = np.mean(Ax_rate_raw[theseRepInds])
                Ax_rate_s[iR,iO] = np.std(Ax_rate_raw[theseRepInds])/np.sqrt(rep)
                Ax_count[iR,iO] = np.mean(Ax_count_raw[theseRepInds])
                Ax_count_s[iR,iO] = np.std(Ax_count_raw[theseRepInds])/np.sqrt(rep)
        s={}
        s['rep']=rep
        s['represR']=represR
        s['represO']=represO
        s['middleR']=middleR
        s['middleO']=middleO
        s['Na']=Na
        s['Nb']=20-Na
        s['Onset']=Onset
        s['Ax_rate']=Ax_rate
        s['Ax_rate_s']=Ax_rate_s
        s['Ax_count']=Ax_count
        s['Ax_count_s']=Ax_count_s
        np.save(path_on_machine + 'Python/SCN/scnres6neu.npy',s)
    else:
        try:
            s = np.load(path_on_machine + 'Python/SCN/scnres6neu.npy')
            s = s.tolist()
            middleR=s['middleR']
            middleO=s['middleO']
            represR=s['represR']
            represO=s['represO']
        except:
            print sys.exit('Data not found yet, use 1 as second argument!')

#------plot part for exp6neu begins here-----------
    #ncontours = 21#30
    gauss_sigma= 1#1.45#1

    #stuff for the "super-enhancement" contour
    #zdata = s['Ax_rate']
    zdata = ndimage.gaussian_filter(s['Ax_count']/0.25,sigma=gauss_sigma,order=0,mode='nearest')
    zdata_s = ndimage.gaussian_filter(s['Ax_count_s']/0.25,sigma=gauss_sigma,order=0,mode='nearest')
    zdata_c = s['Ax_count']
    xdata = s['Onset']
    ydata = s['Na']-s['Nb']
    alladdirate=np.zeros((s['represO'],))
    additiverate = (zdata[0,middleO]/2) + (zdata[represR-1,middleO]/2)
    for iii in range(s['represO']):
        alladdirate[iii]=(2.0*(zdata[represR-1,iii]/2))
    #zdata = ndimage.gaussian_filter(s['Ax_count']/0.25,sigma=gauss_sigma,order=0,mode='nearest')
    #zdatau = s['Ax_count']/0.25
    #zdata = ndimage.gaussian_filter(s['Ax_rate'],sigma=gauss_sigma,order=0,mode='nearest')
    #zdatau = s['Ax_rate']
    #xdata = s['Onset']
    #ydata = s['Na']-s['Nb']
    #alladdirate=np.zeros((41,))
    #additiverate = (zdata[0,20]/2) + (zdata[20,20]/2)
    #for iii in range(41):
    #    alladdirate[iii]=(zdata[0,iii]/2) + (zdata[20,iii]/2)
    
    plt.figure(figsize=(7,3))
    sp1=plt.subplot(121)
    CAx=plt.pcolormesh(zdata,cmap='inferno')
    #CAx=plt.contourf(zdata_c,cmap='inferno',shading='gouraud')
    sp1.set_ylabel('Apical - Basal active synapse count')
    sp1.set_xlabel('Apical Onset delay (ms)')
    sp1.set_title('Axonal AP rate')
    sp1.set_yticks(np.linspace(0.5,s['represR']-0.5,5))
    sp1.set_yticklabels(np.linspace(-20,20,5))
    sp1.set_xticks(np.linspace(0.5,s['represO']-0.5,5))
    sp1.set_xticklabels(np.linspace(-250,250,5))
    cbarax = plt.colorbar(CAx)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbarax.locator = tick_locator
    cbarax.update_ticks()
    cbarax.ax.set_ylabel('Rate AP/s')
    sp2=plt.subplot(122)
    Iover=np.zeros((s['represR'],s['represO']))
    for iii in range(s['represO']):
        Iover[:,iii]=zdata[:,iii]>alladdirate[iii]
    Iunder=np.logical_not(Iover)
    CAx=plt.pcolormesh(Iunder)
    sp2.set_ylabel('TEXT')
    sp2.set_xlabel('TEXT')
    sp2.set_title('TEXT')
#    #plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=3)
#    plt.subplot(121)
#    #CAx=plt.contourf(s['Onset'],s['Na']-s['Nb'],
#    #                 ndimage.gaussian_filter(s['Ax_rate'],sigma=gauss_sigma,order=0,mode='nearest'),
#    #                 ncontours,cmap='inferno')
#    CAx=plt.pcolormesh(xdata,ydata,
#                     zdata,cmap='inferno')
#    plt.ylabel('Apical - Basal active synapse count')
#    plt.xlabel('Apical Onset delay (ms)')
#    plt.title('Axonal AP rate')
#    cbarax = plt.colorbar(CAx)
#    tick_locator = ticker.MaxNLocator(nbins=5)
#    cbarax.locator = tick_locator
#    cbarax.update_ticks()
#    cbarax.ax.set_ylabel('Rate AP/s')
#
#    Iover=np.zeros((21,41))
#    for iii in range(41):
#        Iover[:,iii]=zdata[:,iii]>alladdirate[iii]
#    Iunder=np.logical_not(Iover)
#    #plt.contour(xdata,ydata,ndimage.gaussian_filter(Iover,sigma=gauss_sigma,order=0,mode='nearest'),1,colors='w')
#    plt.contour(xdata,ydata,Iover,1,colors='w')
#    
#    plt.subplot(122)
#    plt.ylabel('Auhjhjount')
#    plt.xlabel('gfhgfhgfs)')
#    plt.title('ghgfh')
    
    plt.tight_layout()
    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres6_F1_neu.pdf')

#    plt.figure()
#    #plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=3)
#    CAx=plt.contourf(s['Onset'],s['Na']-s['Nb'],
#                     ndimage.gaussian_filter(s['Ax_count'],sigma=gauss_sigma,order=0,mode='nearest'),
#                     ncontours,cmap='inferno')
#    plt.ylabel('Apical - Basal active synapse count')
#    plt.xlabel('Apical Onset delay (ms)')
#    plt.title('Axonal total AP count')
#    cbarax = plt.colorbar(CAx)
#    cbarax.ax.set_ylabel('Count AP')
#    #plt.subplot2grid((4,4),(0,3),rowspan=3)
#    #plt.errorbar(np.mean(s['Ax_rate'],1),s['Na']-s['Nb'],xerr=np.std(s['Ax_rate'],1),marker='.')
#    #plt.subplot2grid((4,4),(3,0),colspan=3)
#    #plt.errorbar(s['Onset'],np.mean(s['Ax_rate'],0),yerr=np.std(s['Ax_rate'],0),marker='.')
#    plt.tight_layout()
#    if not machine_has_display:
#        plt.savefig(path_on_machine + 'Python/SCN/scnres6_F2_neu.pdf')

    from scipy.signal import savgol_filter
    from matplotlib.cm import get_cmap 
    sgfiltL=11
    sgfiltO=3
    dim1N=s['represR']#21
    dim2N=s['represO']#31
    dim1mid=middleR#10
    dim2mid=middleO#15    
    plt.figure(figsize=(6.0,6.0))
    mycmap = get_cmap('inferno')
    cm1=np.linspace(0.2,0.8,dim1mid+1)
    cm2=np.linspace(0.2,0.8,dim2mid+1)
    #y1=savgol_filter(s['Ax_rate'][middleR,:],sgfiltL,sgfiltO,mode='nearest')
    #y1=s['Ax_rate'][middleR,:]
    #sy1=s['Ax_rate_s'][middleR,:]
    y1=zdata[middleR,:]
    sy1=zdata_s[middleR,:]
    
    #y2=savgol_filter(s['Ax_rate'][:,middleO],sgfiltL,sgfiltO,mode='nearest')
    #y2=s['Ax_rate'][:,middleO]
    #sy2=s['Ax_rate_s'][:,middleO]
    y2=zdata[:,middleO]
    sy2=zdata_s[:,middleO]
    plt.subplot(221)
    for iii in range(dim2mid+1):
        plt.errorbar(s['Onset'][iii],y1[iii],yerr=sy1[iii],marker='o',color=mycmap(cm2[iii]))
        plt.errorbar(s['Onset'][(dim2N-1)-iii],y1[(dim2N-1)-iii],
                     yerr=sy1[(dim2N-1)-iii],marker='o',color=mycmap(cm2[iii]))
    #plt.errorbar(s['Onset'],y1,yerr=sy1,color='b',marker='o')
    plt.plot(s['Onset'],zdata[0,:],'r-')
    plt.plot(s['Onset'],zdata[dim1N-1,:],'b-')
    plt.xlabel('Apical Onset delay (ms)')
    plt.ylabel('Rate AP/s')
    plt.title('Onset delays')
    plt.subplot(222)
    thisx=s['Na']-s['Nb']
    for iii in range(dim1mid+1):
        plt.errorbar(thisx[iii],y2[iii],yerr=sy2[iii],marker='o',color=mycmap(cm1[iii]))
        plt.errorbar(thisx[(dim1N-1)-iii],y2[(dim1N-1)-iii],yerr=sy2[(dim1N-1)-iii],marker='o',color=mycmap(cm1[iii]))    
    #plt.errorbar(thisx,y2,yerr=sy2,color='b',marker='o')
    plt.plot(thisx,zdata[:,0],'k-')
    plt.plot(thisx,zdata[:,dim2N-1],'k-')
    plt.xlabel('Apical - Basal active synapse count')
    plt.ylabel('Rate AP/s')
    plt.title('Input rates')
    plt.tight_layout()
    plt.subplot(223)
    for iii in range(dim2mid+1):
        plt.plot(y1[iii],y1[(dim2N-1)-iii],marker='o',color=mycmap(cm2[iii]))    
    #plt.plot(y1[0:16],np.flip(y1[15:],0),'bo')
    plt.plot((0,75),(0,75),'k-')
    plt.xlim((20,65))
    plt.ylim((20,65))
    plt.xlabel('Apical leads')
    plt.ylabel('Apical lags')
    plt.subplot(224)
    for iii in range(dim1mid+1):
        plt.plot(y2[iii],y2[(dim1N-1)-iii],marker='o',color=mycmap(cm1[iii]))    
    #plt.plot(y2[0:11],np.flip(y2[10:],0),'bo')
    plt.plot((0,80),(0,80),'k-')
    plt.xlim((20,65))
    plt.ylim((20,65))
    plt.xlabel('Basal dominates')
    plt.ylabel('Apical dominates')
    plt.tight_layout()

    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres6_F5_neu.pdf')
    
    if machine_has_display:
        plt.show()
