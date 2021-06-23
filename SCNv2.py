#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New version of the SCN model.
This module contains the necessary function for all experiment programs.

Created: Tuesday 23rd Feb 2021
@author: kuenzel
"""
#-----IMPORTS--------
from neuron import h
import numpy as np
#
pNeuritdefault = 60.0#µm
#

def runonce(x, P):
    ####hard coded settings, rarely change these:
    APdetect_thr = -75
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    thisdur = (P["dur"][x] - 5.0) / 1000.0
    N = P["Nsyn"][x] + 1
    anf_num = (N, 0, 0)
    if P["AdvZseed"]:
        thisZseed = P["Zseed"] + x
    else:
        thisZseed = P["Zseed"]
    if P["AdvEoHseed"]:
        thisEoHseed = P["EoHseed"] + x
    else:
        thisEoHseed = P["EoHseed"]
    SO = seh.create_sound(
        fs=100e3, freq=P["freq"][x], duration=thisdur, dbspl=P["dB"][x]
    )
    SP = seh.create_spk(
        SO, fs=100e3, N=N, cf=P["cf"][x], seed=thisZseed, anf_num=anf_num
    )  # generate enough spiketrains
    S = np.array(SP["spikes"] * 1000.0)
    R0 = sem.SE_3D(
        S,
        Simdur=P["dur"][x],
        dt=P["dt"][x],
        G_EoH=P["G_EoH"][x],
        StochasticEoH=P["Stochastic_EoH"][x],
        N=P["Nsyn"][x],
        G_Dend=P["gsyn"][x] * 0.0,  # set dendritic synapses to zero
        tau_Dend=P["tau_dendsyn"][x],
        gKLT_d=P["gKLT_d"][x],
        gIH_d=P["gIH_d"][x],
        gLEAK_d=P["gLEAK_d"][x],
        cell=P["cell"][x],
        EoHseed=thisEoHseed,
        somaticsynapsesonly=P["synapselocationtype"][x],
    )
    #
    RD = sem.SE_3D(
        S,
        Simdur=P["dur"][x],
        dt=P["dt"][x],
        G_EoH=P["G_EoH"][x],
        StochasticEoH=P["Stochastic_EoH"][x],
        N=P["Nsyn"][x],
        G_Dend=P["gsyn"][x],
        tau_Dend=P["tau_dendsyn"][x],
        gKLT_d=P["gKLT_d"][x],
        gIH_d=P["gIH_d"][x],
        gLEAK_d=P["gLEAK_d"][x],
        cell=P["cell"][x],
        EoHseed=thisEoHseed,
        somaticsynapsesonly=P["synapselocationtype"][x],
    )
    Ev0 = seh.SimpleDetectAP(R0["Vm"], thr=APdetect_thr, dt=P["dt"][x], LM=-20, RM=10)
    EvD = seh.SimpleDetectAP(RD["Vm"], thr=APdetect_thr, dt=P["dt"][x], LM=-20, RM=10)
    APr0 = len(Ev0["PeakT"]) / (P["dur"][x] / 1000.0)
    APrD = len(EvD["PeakT"]) / (P["dur"][x] / 1000.0)
    VS0, phi0, Ray0, phases0 = seh.vectorstrength(
        Ev0["PeakT"], P["freq"][x], [0.0, P["dur"][x]]
    )
    VSD, phiD, RayD, phasesD = seh.vectorstrength(
        EvD["PeakT"], P["freq"][x], [0.0, P["dur"][x]]
    )
    print((str(x)))
    return [VS0, phi0, Ray0, APr0, VSD, phiD, RayD, APrD]


def myMPhandler(P):
    poolmess = partial(runonce, P=P)
    if P["mp"]:
        with multiprocessing.Pool(P["cores"]) as activepool:
            r = activepool.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def nrngo(tstop,vinit):
    h.finitialize(vinit)
    h.fcurrent()
    while h.t<tstop:
        h.fadvance()


def SimpleDetectAP(V,thr=-100,dt=0.01,LM=-20,RM=10):
    """
    Detect spikes in simulated Vm without knowing the Spt or with many inputs.
    Using a dV/dt threshold of -100mV/ms usually is robust.
    """
    T = np.linspace(0,(len(V)*dt)-dt,len(V))
    dV = np.diff(V)/dt
    Ilow=np.where(dV<thr)[0]
    Ilow = np.concatenate(([0],Ilow))
    dIlow=np.diff(Ilow)
    firstIlow=np.where(dIlow>1.1)[0]
    DetectI=Ilow[firstIlow+1]
    DetectT = T[DetectI]
    PeakI = []
    PeakT = []
    PeakV = []
    for nEv,IEv in enumerate(DetectI):
        if IEv+LM < 0:
            localI=V[0:IEv+RM].argmax()-IEv
            PeakV.append(V[0:IEv+RM].max())
        elif IEv+RM > len(V):
            localI=V[IEv+LM:len(V)].argmax()+LM
            PeakV.append(V[IEv+LM:len(V)].max())
        else:
            localI=V[IEv+LM:IEv+RM].argmax()+LM
            PeakV.append(V[IEv+LM:IEv+RM].max())
        PeakI.append(IEv+localI)
        PeakT.append(T[PeakI[-1]])
            
    Res = {}
    Res['PeakI']=PeakI
    Res['PeakT']=PeakT
    Res['PeakV']=PeakV
    Res['DetectI']=DetectI
    Res['DetectT']=DetectT
    return(Res)


def runmodel(
        tstop=500.0,
        dt=0.025,
        nsyna=10,
        nsynb=10,
        hasstimulation=(False, False),
        pstimulation=(0.0, 1, 250.0, 1.0, 250.0, 1, 250.0, 1.0),
        haselectrode=False,
        pelectrode=(0,10.0, 50.0, 0.2, 0.0),
        hasinputactivity=(True, True),
        pinputactivity=(0.0, 33.0, 0.0, 33.0),
        inputstop=(500.0, 500.0),
        hasnmda=True,
        apicaltau=80.0,#new2021-06-21
        seed=32768,
        pNeurit_L=pNeuritdefault,
        pretime=100.0,
        T=25.0,
        hasfbi=False,
        hasffi=True,
        inhw=0.001,
        inhtau=75.0,#100.0
        inhdelay=5.0,
        noiseval=0.9,#0.5
        soma_na=0.2,#0.125
        soma_k=0.04,#0.027
        soma_kht=0.013,#0.013
        soma_l=0.0001,#0.0001
        dend_l=0.00001,#0.0001
        reallatency=False,
        axon_na=0.32,
        axon_k=0.04,
        axon_l=0.0001,
    ):
    """
    Creates and runs the second version of the SCN model.
    """ 
    #
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
    #
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
    pNeurit.L = pNeurit_L#from table (45 is the old default, 60 the new)
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
    #if pNeurit_L < pNeuritdefault:
    #    wehaveextradend = True
    #    extraDend = h.Section()
    #    extraDend.L = pNeuritdefault - pNeurit_L
    #    extraDend.diam = 2
    #    extraDend.nseg = 6
    #else:
    #    wehaveextradend = False
    #axial Resistance and cm
    for sec in h.allsec():
        sec.Ra = 150
        sec.insert("extracellular")
    Internode1.cm = 0.01
    Internode2.cm = 0.01
    #
    #hh
    Soma.insert("hh")
    Soma.ena = 50
    Soma.gnabar_hh = soma_na#0.12
    Soma.gkbar_hh = soma_k#0.023
    Soma.gl_hh = soma_l#hh model has own leak channel representation! [0.0001]
    AxonInit.insert("hh")
    AxonInit.ena = 50
    AxonInit.gnabar_hh = soma_na + 0.04#+0.04
    AxonInit.gkbar_hh = soma_k#0.023
    AxonInit.gl_hh = soma_l#hh model has own leak channel representation!
    pNeurit.insert("hh")
    pNeurit.ena = 50
    pNeurit.gnabar_hh = soma_na + 0.02#+0.02
    pNeurit.gkbar_hh = soma_k#0.023
    pNeurit.gl_hh = soma_l#hh model has own leak channel representation!
    #
    Node1.insert("hh")
    Node1.gnabar_hh = axon_na#0.32#0.36
    Node1.gkbar_hh = axon_k#0.04#0.04
    Node1.gl_hh = axon_l#0.0001#1e-6#hh model has own leak channel representation!
    Node2.insert("hh")
    Node2.gnabar_hh = axon_na#0.32#0.36
    Node2.gkbar_hh = axon_k#0.04#0.04
    Node2.gl_hh = axon_l#0.0001#1e-6#hh model has own leak channel representation!
    #
    #kht
    Soma.insert("kht")
    Soma.gkhtbar_kht = soma_kht
    Soma.ek = -80
    pNeurit.insert("kht")
    pNeurit.gkhtbar_kht = soma_kht
    pNeurit.ek = -80
    AxonInit.insert("kht")
    AxonInit.gkhtbar_kht = soma_kht
    AxonInit.ek = -80
    bproxDend.insert("kht")
    bproxDend.gkhtbar_kht = soma_kht
    bproxDend.ek = -80
    #
    #leak
    aDend.insert("leak")
    aDend.g_leak = dend_l#[0.0001]
    aDend.erev_leak = -65
    #if wehaveextradend:
    #    extraDend.insert("leak")
    #    extraDend.g_leak = dend_l#[0.0001]
    #    extraDend.erev_leak = -65
    bDend.insert("leak")
    bDend.g_leak = dend_l
    bDend.erev_leak = -65
    bproxDend.insert("leak")
    bproxDend.g_leak = dend_l
    bproxDend.erev_leak = -65
    Internode1.insert("leak")
    Internode1.g_leak = 1e-9
    Internode1.erev_leak = -65
    Internode2.insert("leak")
    Internode2.g_leak = 1e-9
    Internode2.erev_leak = -65
    #
    #Topology
    Node2.connect(Internode2(1))
    Internode2.connect(Node1(1))
    Node1.connect(Internode1(1))
    Internode1.connect(AxonInit(1))
    AxonInit.connect(pNeurit(1))
    #if wehaveextradend:
    #    aDend.connect(extraDend(1))
    #    extraDend.connect(pNeurit(1))
    #else:
    #    aDend.connect(pNeurit(1))
    aDend.connect(pNeurit(1))
    pNeurit.connect(Soma(1))
    bproxDend.connect(Soma(0))
    bDend.connect(bproxDend(1))
    #
    # GENERAL SETTINGS
    h.dt = dt  # simulation (or "sampling") rate
    h.celsius = T  # simulation global temperature
    #
    # MODEL INSTRUMENTATION & INPUT
    if haselectrode:
        #pelectrode=(0,10.0,50.0,0.2)
        if pelectrode[0] == 0:
            Electrode = h.IClamp(Soma(0.5))
            Electrode.delay = pelectrode[1] + pretime
            Electrode.dur = pelectrode[2]
            Electrode.amp = pelectrode[3]
            SCN_Im = []
        else:
            Electrode = h.SEClamp(Soma(0.5))
            Electrode.dur1 = pelectrode[1] + pretime
            Electrode.dur2 = pelectrode[2]
            Electrode.dur3 = tstop - (pelectrode[1] + pelectrode[2])
            Electrode.amp1 = pelectrode[3]
            Electrode.amp2 = pelectrode[4]
            Electrode.amp3 = pelectrode[3]
            Electrode.rs = 0.001
            SCN_Im = h.Vector()
            SCN_Im.record(Electrode._ref_i)
    else:
        SCN_Im = []
    # Synapses
    """
    from the new data provided by Stefan Weigel in Feb 2021 we took the
    following synaptic parameters:
    risetaua (assuming a time constant of 1/5 of the risetime): 18.2 / 5 = 3.64
    risetaub (assuming a time constant of 1/5 of the risetime): 5.8 / 5 = 1.16
    exponential decay taua @100µA = 105.9
    exponential decay taub @100µA = 50.3
    apical EPSC@100µA = 0.542nA --> 7.74nS --> w=0.00774
    basal EPSC@100µA = 0.390nA --> 5.57nS --> w=0.00557
    #
    very latest post 04.03.21 data:
    -------------------------------
    risetime 19.2 vs 6.0
    decay 130 vs 30
    ampl 600 zu 600 --> 8.6nS --> w=0.0086
    
    Even later re-correction by Stefan (Email 28.05.21)
     Die bewegen sich jetzt für die apikale Stimulation im Bereich von ~90 bis 
     110 ms (decay; würde einfach 100 nehmen ) und ~15 ms für rise time. 
     Bei der basalen Stimulation liegen die Werte jetzt bei ~20ms decay tau 
     und 6 ms rise time.

    Real latency (21.05.21)
    ------------
    Stefans conclusion from literature search is:
    minimal latency apical/visual = 50ms
    minimal latency basal/auditory = 20ms
    """
    risetaua = 3#3.84#estimated from 1/5 risetime
    risetaub = 1.2#estimated from 1/5 risetime
    decaytaua = apicaltau#80.ms(new2021-06-21)####100.0 (2021-06-07) ####130.0#measured by SW
    decaytaub = 25.0#new2021-06-21#20#30.0#measured by SW
    decaytaua_nonmda = 30.0##new2021-06-21
    synw = 0.01#0.0086 is one side!
    """
    in order for the experiments about enhancement to properly work we need to
    we able to set the input power. This can be controlled by input numbers (as
    in the previous version) or total conductance etc. But it is probably most
    biologically relevant to use the average input rate as a metric for the
    power of an input.
    In the "simulated slice experiment" we should use a percentage of the total
    available conductance.
    """
    if nsyna == 0:
        synwa = 0.0
    else:
        synwa = synw / nsyna #we distribute the total input conductance over all synapses
    if nsynb == 0:
        synwb = 0.0
    else:
        synwb = synw / nsynb #see above
    anc = []
    asyn = []
    astim = []
    atv = []
    bnc = []
    bsyn = []
    bstim = []
    btv = [] 
    for iasyn in range(nsyna):
        astim.append(h.NetStim())
        asyn.append(h.Exp2Syn(aDend((iasyn+1)*(1.0/nsyna))))#evenly distribute synapses
        atv.append(h.Vector())
        if hasnmda:#apical EPSC are longer due to more NMDA component
            asyn[iasyn].tau1 = risetaua
            asyn[iasyn].tau2 = decaytaua
        else:#NMDA "blocked" in this model run, now apical EPSC faster
            asyn[iasyn].tau1 = risetaub
            asyn[iasyn].tau2 = decaytaua_nonmda#(new 2021-06-21)decaytaub
        asyn[iasyn].e = 0
        anc.append(h.NetCon(astim[iasyn], asyn[iasyn]))
        anc[iasyn].record(atv[iasyn])
        if reallatency:
            anc[iasyn].delay = 50.0
        else:
            anc[iasyn].delay = 0.5
        if hasnmda:
            anc[iasyn].weight[0] = synwa
        else:
            anc[iasyn].weight[0] = synwb
        if hasstimulation[0]:
            #pstimulation=(0.0, 2, 50.0, 1.0, 0.0, 2, 50.0, 1.0),
            astim[iasyn].start = pstimulation[0] + pretime
            astim[iasyn].number = pstimulation[1]
            astim[iasyn].noise = 0.001#0 or 0.001
            astim[iasyn].interval = pstimulation[2]
            astim[iasyn].seed(seed + (iasyn * 54))
            anc[iasyn].weight[0] = anc[iasyn].weight[0] * pstimulation[3]
        elif hasinputactivity[0]:
            #pinputactivity=(0.0, 20.0, 0.0, 20.0),
            #inputstop=(500.0, 500.0),
            thisN = int(np.round((inputstop[0]-pinputactivity[0])/pinputactivity[1]))
            astim[iasyn].number = thisN
            astim[iasyn].start = pinputactivity[0] + pretime
            astim[iasyn].noise = noiseval
            astim[iasyn].interval = pinputactivity[1]
            astim[iasyn].seed(seed + (iasyn * 54))
        else:
            astim[iasyn].number = 0
            astim[iasyn].start = 0.0
            astim[iasyn].noise = 0
            astim[iasyn].interval = 1.0
            astim[iasyn].seed(seed + (iasyn * 54))
            anc[iasyn].weight[0] = 0.0 
    for ibsyn in range(nsynb):
        bstim.append(h.NetStim())
        bsyn.append(h.Exp2Syn(bDend((ibsyn+1)*(1.0/nsynb))))#evenly distribute synapses
        btv.append(h.Vector())
        bsyn[ibsyn].tau1 = risetaub
        bsyn[ibsyn].tau2 = decaytaub
        bsyn[ibsyn].e = 0
        bnc.append(h.NetCon(bstim[ibsyn], bsyn[ibsyn]))
        bnc[ibsyn].record(btv[ibsyn])
        if reallatency:
            bnc[ibsyn].delay = 20.0
        else:
            bnc[ibsyn].delay = 0.5
        bnc[ibsyn].weight[0] = synwb
        if hasstimulation[1]:
            #pstimulation=(0.0, 2, 50.0, 1.0, 0.0, 2, 50.0, 1.0),
            bstim[ibsyn].start = pstimulation[4] + pretime
            bstim[ibsyn].number = pstimulation[5]
            bstim[ibsyn].noise = 0.001#0 or 0.001
            bstim[ibsyn].interval = pstimulation[6]
            bnc[ibsyn].weight[0] = bnc[ibsyn].weight[0] * pstimulation[7]
            bstim[ibsyn].seed(seed + (ibsyn * 74))
        elif hasinputactivity[1]:
            #pinputactivity=(0.0, 20.0, 0.0, 20.0),
            #inputstop=(500.0, 500.0),
            thisN = int(np.round((inputstop[1]-pinputactivity[2])/pinputactivity[3]))
            bstim[ibsyn].number = thisN
            bstim[ibsyn].start = pinputactivity[2] + pretime
            bstim[ibsyn].noise = noiseval
            bstim[ibsyn].interval = pinputactivity[3]
            bstim[ibsyn].seed(seed + (ibsyn * 74))
        else:
            bstim[ibsyn].number = 0
            bstim[ibsyn].start = 0.0
            bstim[ibsyn].noise = 0
            bstim[ibsyn].interval = 1.0
            bstim[ibsyn].seed(seed + (ibsyn * 74))
            bnc[ibsyn].weight[0] = 0.0
    if hasfbi:
        isyna = h.ExpSyn(aDend(0.5))
        isyna.tau = inhtau#30.0#5.0
        isyna.e = -85
        isynb = h.ExpSyn(bDend(0.5))
        isynb.tau = inhtau#30.0#5.0
        isynb.e = -85
        inc1 = h.NetCon(Node2(0.9)._ref_v, isyna, sec=Node2)
        inc1.threshold = -35.0
        inc1.weight[0] = 0.005#0.02#0.01
        inc1.delay = 5.0#[here was 2.0ms, Stefan prefers 5ms]
        inc2 = h.NetCon(Node2(0.9)._ref_v, isynb, sec=Node2)
        inc2.threshold = -35.0
        inc2.weight[0] = 0.005#0.02#0.01
        inc2.delay = 5.0
    if hasffi:
        if hasinputactivity[0]:
            isynca = h.ExpSyn(aDend(0.41))#0.01
            isynca.tau = inhtau#50.0#5.0
            isynca.e = -85
            isyncb = h.ExpSyn(bDend(0.41))#0.01
            isyncb.tau = inhtau#50.0#5.0
            isyncb.e = -85
            inhstimc = h.NetStim()
            thisN = int(np.round((inputstop[0]-pinputactivity[0])/pinputactivity[1]))
            inhstimc.number = thisN
            inhstimc.start = pinputactivity[0] + pretime
            inhstimc.noise = 1.0
            inhstimc.interval = pinputactivity[1]
            inhstimc.seed(seed + 2944)
            inhconca = h.NetCon(inhstimc, isynca)
            inhconcb = h.NetCon(inhstimc, isyncb)
            if reallatency:
                inhconca.delay = inhdelay + 50.0
                inhconcb.delay = inhdelay + 50.0
            else:
                inhconca.delay = inhdelay
                inhconcb.delay = inhdelay
            inhconca.weight[0] = inhw
            inhconcb.weight[0] = inhw
        if hasinputactivity[1]:
            isynda = h.ExpSyn(aDend(0.4))#0.4?
            isynda.tau = inhtau#50.0#5.0
            isynda.e = -85
            isyndb = h.ExpSyn(bDend(0.4))#0.4?
            isyndb.tau = inhtau#50.0#5.0
            isyndb.e = -85
            inhstimd = h.NetStim()
            thisN = int(np.round((inputstop[1]-pinputactivity[2])/pinputactivity[3]))
            inhstimd.number = thisN
            inhstimd.start = pinputactivity[2] + pretime
            inhstimd.noise = 1.0
            inhstimd.interval = pinputactivity[3]
            inhstimd.seed(seed + 3245)
            inhconda = h.NetCon(inhstimd, isynda)
            inhcondb = h.NetCon(inhstimd, isyndb)
            if reallatency:
                inhconda.delay = inhdelay + 20.0
                inhcondb.delay = inhdelay + 20.0
            else:
                inhconda.delay = inhdelay
                inhcondb.delay = inhdelay
            inhconda.weight[0] = inhw
            inhcondb.weight[0] = inhw
    #
    # INFRASTRUCTURE
    SCN_SVm = h.Vector()
    SCN_SVm.record(Soma(0.5)._ref_v)
    SCN_AVm = h.Vector()
    SCN_AVm.record(Node2(0.99)._ref_v)
    #
    #----------SIMULATE--------------
    nrngo(tstop+pretime,-66.5)
    # PACK AND EXPORT DATA
    Result = {}
    tempres_avm = np.array(SCN_AVm)
    tempres_svm = np.array(SCN_SVm)
    tempres_im = np.array(SCN_Im)
    Result['AVm'] = tempres_avm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    Result['SVm'] = tempres_svm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    Result['Im'] = tempres_im[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    ra = []
    rb = []
    for iatv in range(len(atv)):
        ra.append(np.array(atv[iatv])-pretime)
    for ibtv in range(len(btv)):
        rb.append(np.array(btv[ibtv])-pretime)
    Result["atv"] = ra
    Result["btv"] = rb
    #Result['atv'] = atv#packaged as hoc objects... need to unpack later on
    #Result['btv'] = btv#packaged as hoc objects... need to unpack later on
    return(Result)
