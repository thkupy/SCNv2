#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:01:38 2019
This is the program that generates the 
"Lead Time diff" vs. "Primary Dendrite Length" 2D plots for the SCN project
@author: kuenzel
"""
#
path_on_machine = '/home/tk/'
cores_on_machine = 20
machine_has_display = False
global_dt = 0.025
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
             inpn=(10,10),starta=0.0,startb=0.0,syntau=2,synw=0.002,
             pNeurit_L=45):
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
    pNeurit.L = pNeurit_L#from table (45 is the default)
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
    meaninterval=20
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
        astim[iasyn].number = (tstop/meaninterval)+20
        astim[iasyn].start = starta
        astim[iasyn].noise = 1
        astim[iasyn].interval = meaninterval
        astim[iasyn].seed(seed)
        anc.append(h.NetCon(astim[iasyn],asyn[iasyn]))
        anc[iasyn].record(atv[iasyn])
        anc[iasyn].delay=0.5
        if a_on:
            #anc[iasyn].weight[0]=np.abs(np.random.normal(1))*0.002
            anc[iasyn].weight[0]=synw
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
        bstim[ibsyn].number = (tstop/meaninterval)+20
        bstim[ibsyn].start = startb
        bstim[ibsyn].noise = 1
        bstim[ibsyn].interval = meaninterval
        bstim[ibsyn].seed(seed)
        bnc.append(h.NetCon(bstim[ibsyn],bsyn[ibsyn]))
        bnc[ibsyn].record(btv[ibsyn])
        bnc[ibsyn].delay=0.5
        
        if b_on:
            #bnc[ibsyn].weight[0]=np.abs(np.random.normal(1))*0.002
            bnc[ibsyn].weight[0]=synw
        else:
            bnc[ibsyn].weight[0]=0
    #print 'Na=' + str(inpn[0]) + ' + Nb=' + str(inpn[1])#debug
        
    # INFRASTRUCTURE
    SCN_SVm = h.Vector()
    SCN_SVm.record(Soma(0.5)._ref_v)
    SCN_AVm = h.Vector()
    SCN_AVm.record(Node2(0.99)._ref_v)


    #----------SIMULATE--------------
    tkt.nrngo(tstop+pretime,-66.5)
    # PACK AND EXPORT DATA
    Result = {}
    tempres_avm = np.array(SCN_AVm)
    tempres_svm = np.array(SCN_SVm)
    Result['AVm'] = tempres_avm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    Result['SVm'] = tempres_svm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    Result['atv'] = atv#packaged as hoc objects... need to unpack later on
    Result['btv'] = btv#packaged as hoc objects... need to unpack later on
    return(Result)

def myjob(myitems):
    dt = global_dt
    print "Working in Process #%d" % (os.getpid())
    Na = 10
    Nb = 10
    Onset = myitems[1]
    seed=32768+myitems[2]
    R=runmodel(tstop=1000.0,dt=dt,a_on=True,b_on=True,inpn=(int(round(Na)),int(round(Nb))),
                                    starta=200.0,startb=200.0+Onset,seed=seed,
                                    pNeurit_L=myitems[0])
    spA=tkt.SimpleDetectAP(R['AVm'],dt=dt,thr=-50,LM=-30)
    spS=tkt.SimpleDetectAP(R['SVm'],dt=dt,thr=-30,LM=-30)   
    for Ai in range(len(R['atv'])):
        if Ai == 0:
            ispta=np.array(R['atv'][0])
        else:
            ispta=np.append(ispta,np.array(R['atv'][Ai]))
    for Bi in range(len(R['btv'])):
        if Bi == 0:
            isptb=np.array(R['btv'][0])
        else:
            isptb=np.append(isptb,np.array(R['btv'][Bi]))
    A_inrate=len(ispta)
    B_inrate=len(isptb)
    Ax_rate=len(spA['PeakT'])
    So_rate=len(spS['PeakT'])
    res={}
    res['A_inrate']=A_inrate
    res['B_inrate']=B_inrate
    res['Ax_rate']=Ax_rate
    res['So_rate']=So_rate
    return res
 
if __name__ == '__main__':
    if int(sys.argv[1])==1:#do experiment
        rep = 30
        represL = 19
        represO = 21
        middleL = 9
        middleO = 10
        alllengths = np.round(np.linspace(1,125,represL))
        Onset = np.linspace(-100,100,represO)
        Nrep = range(rep)
        mylist = list(itertools.product(alllengths,Onset,Nrep))
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
        print "Parallel execution time: " + str((end - start)) + " s"
        print "This is " + str((end-start)/len(mylist)) + "s per execution"
        
        Ax_rate_raw = np.zeros((len(AR),))
        So_rate_raw = np.zeros((len(AR),))
        A_inrate_raw = np.zeros((len(AR),))
        B_inrate_raw = np.zeros((len(AR),))
        for indi,thisR in enumerate(AR):
            UnpackedR=thisR.get()
            Ax_rate_raw[indi] = UnpackedR['Ax_rate']
            So_rate_raw[indi] = UnpackedR['So_rate']
            A_inrate_raw[indi] = UnpackedR['A_inrate']
            B_inrate_raw[indi] = UnpackedR['B_inrate']
        PL=np.array(mylist)
        A_inrate=np.zeros((represL,represO))
        B_inrate=np.zeros((represL,represO))
        Ax_rate=np.zeros((represL,represO))
        Ax_rate_s=np.zeros((represL,represO))
        So_rate=np.zeros((represL,represO))
        So_rate_s=np.zeros((represL,represO))
        for iL in range(represL):
            for iO in range(represO):
                theseRepInds = np.logical_and(PL[:,0]==alllengths[iL],PL[:,1]==Onset[iO])
                A_inrate[iL,iO]= np.mean(A_inrate_raw[theseRepInds])
                B_inrate[iL,iO]= np.mean(B_inrate_raw[theseRepInds])
                Ax_rate[iL,iO] = np.mean(Ax_rate_raw[theseRepInds])
                Ax_rate_s[iL,iO] = np.std(Ax_rate_raw[theseRepInds])
                So_rate[iL,iO] = np.mean(So_rate_raw[theseRepInds])
                So_rate_s[iL,iO] = np.std(So_rate_raw[theseRepInds])
        s={}
        s['rep']=rep
        s['represL']=represL
        s['represO']=represO
        s['middleL']=middleL
        s['middleO']=middleO
        s['Na']=10
        s['Nb']=10
        s['pNeurite_L']=alllengths
        s['Onset']=Onset
        s['A_inrate']=A_inrate
        s['B_inrate']=B_inrate
        s['Ax_rate']=Ax_rate
        s['Ax_rate_s']=Ax_rate_s
        s['So_rate']=So_rate
        s['So_rate_s']=So_rate_s
        np.save(path_on_machine + 'Python/SCN/scnres11.npy',s)
    else:
        try:
            s = np.load(path_on_machine + 'Python/SCN/scnres11.npy')
            s = s.tolist()
            middleL=s['middleL']
            middleO=s['middleO']
        except:
            print sys.exit('Data not found yet, use 1 as second argument!')
        #------plot part for exp6 begins here-----------
    plt.figure()
    ncontours = 30
    gauss_sigma=1
    #plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=3)
    CAx=plt.contourf(s['Onset'],s['pNeurite_L'],ndimage.gaussian_filter(s['Ax_rate'],
                     sigma=gauss_sigma,order=0),ncontours)
    plt.ylabel(u'pNeurite L [µm]')
    plt.xlabel('Apical lead time (ms)')
    plt.title('Axonal AP rate')
    cbarax = plt.colorbar(CAx)
    cbarax.ax.set_ylabel('Rate AP/s')
    #plt.subplot2grid((4,4),(0,3),rowspan=3)
    #plt.errorbar(np.mean(s['Ax_rate'],1),s['Na']-s['Nb'],xerr=np.std(s['Ax_rate'],1),marker='.')
    #plt.subplot2grid((4,4),(3,0),colspan=3)
    #plt.errorbar(s['Onset'],np.mean(s['Ax_rate'],0),yerr=np.std(s['Ax_rate'],0),marker='.')
    #plt.tight_layout()
    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres11_F1.pdf')
       
    plt.figure()
    CSo=plt.contourf(s['Onset'],s['pNeurite_L'],ndimage.gaussian_filter(s['So_rate'],
                     sigma=gauss_sigma,order=0),ncontours)
    plt.ylabel(u'pNeurite L [µm]')
    plt.xlabel('Apical lead time (ms)')
    plt.title('Somatic AP rate (sp/s)')
    cbarso = plt.colorbar(CSo)
    cbarso.ax.set_ylabel('Rate AP/s')
    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres11_F2.pdf')
    
    plt.figure()
    CA=plt.contourf(s['Onset'],s['pNeurite_L'],ndimage.gaussian_filter(s['A_inrate'],
                    sigma=gauss_sigma,order=0),ncontours)
    plt.ylabel(u'pNeurite L [µm]')
    plt.xlabel('Apical lead time (ms)')
    plt.title('Input rate Apical (sp/s)')
    cbara = plt.colorbar(CA)
    cbara.ax.set_ylabel('Rate AP/s')
    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres11_F3.pdf')
    
    plt.figure()
    CB=plt.contourf(s['Onset'],s['pNeurite_L'],ndimage.gaussian_filter(s['B_inrate'],
                    sigma=gauss_sigma,order=0),ncontours)
    plt.ylabel(u'pNeurite L [µm]')
    plt.xlabel('Apical lead time (ms)')
    plt.title('Input rate Basal (sp/s)')
    cbarb = plt.colorbar(CB)
    cbarb.ax.set_ylabel('Rate AP/s')
    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres11_F4.pdf')
    
    plt.figure()
    plt.subplot(121)
    plt.errorbar(s['Onset'],s['Ax_rate'][middleO,:],yerr=s['Ax_rate_s'][middleO,:],color='b',marker='o')
    plt.errorbar(s['Onset'],s['So_rate'][middleO,:],yerr=s['So_rate_s'][middleO,:],color='r',marker='o')
    plt.xlabel('Apical lead time (ms)')
    plt.ylabel('Rate AP/s')
    plt.title('Onset delays')
    plt.legend(('Axonal','Somatic'))
    plt.subplot(122)
    plt.errorbar(s['pNeurite_L'],s['Ax_rate'][:,middleL],yerr=s['Ax_rate_s'][:,middleL],color='b',marker='o')
    plt.errorbar(s['pNeurite_L'],s['So_rate'][:,middleL],yerr=s['So_rate_s'][:,middleL],color='r',marker='o')
    plt.ylabel(u'pNeurite L [µm]')
    plt.ylabel('Rate AP/s')
    plt.title('Input rates')
    plt.legend(('Axonal','Somatic'))
    if not machine_has_display:
        plt.savefig(path_on_machine + 'Python/SCN/scnres11_F5.pdf')
    
    if machine_has_display:
        plt.show()
    
    
    
    
    #
#    allinv = range(100)
#    R = np.zeros((100,))
#    #
#    start = time.time()
#    #
#    pool = Pool(processes=4)
#    for thisind,oneinv in enumerate(allinv):
#        R[thisind]=pool.apply_async(mytask,(oneinv,)).get()
#    pool.close()
#    pool.join()
#    #
#    end = time.time()
#    print "Parallel execution time: " + str((end - start)) + " s"
#    #
#    
#    start = time.time()
#    for thisind,oneinv in enumerate(allinv):
#        R[thisind]=mytask(oneinv)
#    end = time.time()
#    print "serial execution time: " + str((end - start)) + " s"
    
