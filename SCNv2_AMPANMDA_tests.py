#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This program serves the sole purpose of exploring different options for implementing
the NMDA Component from Stefans recordings more realistically in the model.

Created: 2021-06-17
Revised: 
@author: kuenzel(at)bio2.rwth-aachen.de
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from neuron import h

def nrngo(tstop,vinit):
    h.finitialize(vinit)
    h.fcurrent()
    while h.t<tstop:
        h.fadvance()


def modelcell(
        tstop=450.0,
        pretime=100.0,
        dt = 0.01,
        hasnmda=True,
        hasstimulation=True,
        pstimulation=(0.0, 1, 50.0, 1.0),
        pelectrode=(0,10.0, 50.0, -70.0, -70.0),
        hasinputactivity=False,
        pinputactivity=(0.0, 10.0),
        inputstop=150.0,
        noiseval=0.99,
        seed=23348,
        VC=True,
        T=25,
    ):
    #--------MODEL DEFINITIONS---------
    #CREATE SECTIONS
    Soma = h.Section()
    aDend1 = h.Section()#
    aDend2 = h.Section()
    #
    #Geometry & Biophysics
    Soma.L = 20#
    Soma.diam = 20#
    Soma.nseg = 7
    #
    aDend1.L = 150#
    aDend1.diam = 2
    aDend1.nseg = 15
    #
    aDend2.L = 150#
    aDend2.diam = 2
    aDend2.nseg = 15
    #
    #
    for sec in h.allsec():
            sec.Ra = 150
            sec.insert("extracellular")
    #
    #
    Soma.insert("hh")
    Soma.ena = 50
    Soma.gnabar_hh = 0.12
    Soma.gkbar_hh = 0.023
    Soma.gl_hh = 0.0001
    aDend1.insert("leak")
    aDend1.g_leak = 0.0001
    aDend1.erev_leak = -65
    aDend2.insert("leak")
    aDend2.g_leak = 0.0001
    aDend2.erev_leak = -65
    #
    #Topology
    aDend2.connect(aDend1(1))
    aDend1.connect(Soma(1))
    #
    # GENERAL SETTINGS
    h.dt = dt  # simulation (or "sampling") rate
    h.celsius = T  # simulation global temperature
    #
    #SYNAPSE
    AMPA_rise = 1.2
    NMDA_rise = 5.0
    AMPA_decay = 7.5
    NMDA_decay = 100.0
    AMPAfrac = 0.8
    synw = 0.012#
    #
    astim = h.NetStim()
    asyn_AMPA = h.Exp2Syn(aDend2(0.5))
    asyn_NMDA = h.Exp2Syn(aDend2(0.5))
    asyn_AMPA.tau1 = AMPA_rise
    asyn_AMPA.tau2 = AMPA_decay
    asyn_AMPA.e = 0.0
    asyn_NMDA.tau1 = NMDA_rise
    asyn_NMDA.tau2 = NMDA_decay
    asyn_NMDA.e = 0.0
    #
    anc_AMPA = h.NetCon(astim, asyn_AMPA)
    anc_NMDA = h.NetCon(astim, asyn_NMDA)
    anc_AMPA.delay = 0.5
    anc_NMDA.delay = 0.5

    if hasnmda:
        anc_AMPA.weight[0] = synw * AMPAfrac
        anc_NMDA.weight[0] = synw * (1 - AMPAfrac)
    else:
        anc_AMPA.weight[0] = synw * AMPAfrac
        anc_NMDA.weight[0] = 1e-12
    #
    if hasstimulation:
        #pstimulation=(0.0, 2, 50.0, 1.0),
        astim.start = pstimulation[0] + pretime
        astim.number = pstimulation[1]
        astim.noise = 0.001#0 or 0.001
        astim.interval = pstimulation[2]
        astim.seed(seed + 54)
        anc_AMPA.weight[0] = anc_AMPA.weight[0] * pstimulation[3]
        anc_NMDA.weight[0] = anc_NMDA.weight[0] * pstimulation[3]
    elif hasinputactivity:
        #pinputactivity=(0.0, 20.0, 0.0, 20.0),
        #inputstop=(500.0, 500.0),
        thisN = int(np.round((inputstop[0]-pinputactivity[0])/pinputactivity[1]))
        astim.number = thisN
        astim.start = pinputactivity[0] + pretime
        astim.noise = noiseval
        astim.interval = pinputactivity[1]
        astim.seed(seed + 54)
    else:
        astim.number = 0
        astim.start = 0.0
        astim.noise = 0
        astim.interval = 1.0
        astim.seed(seed +  54)
        anc_AMPA.weight[0] = 0.0 
        anc_NMDA.weight[0] = 0.0 
    #
    #Instrumentation
    if VC:
        Electrode = h.SEClamp(Soma(0.5))
        Electrode.dur1 = pelectrode[1] + pretime
        Electrode.dur2 = pelectrode[2]
        Electrode.dur3 = tstop - (pelectrode[1] + pelectrode[2])
        Electrode.amp1 = pelectrode[3]
        Electrode.amp2 = pelectrode[4]
        Electrode.amp3 = pelectrode[3]
        Electrode.rs = 0.001
        Im = h.Vector()
        Im.record(Electrode._ref_i)
        Vm = []
    else:
        Vm = h.Vector()
        Vm.record(Soma(0.5)._ref_v)
        Im = []
    #
    #----------SIMULATE--------------
    nrngo(tstop+pretime,-66.5)
    # PACK AND EXPORT DATA
    Result = {}
    tempres_avm = np.array(Vm)
    tempres_im = np.array(Im)
    if VC:
        Result['Vm'] = []
        Result['Im'] = tempres_im[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
        Result['taxis'] = np.linspace(0,tstop-dt, Result['Im'].size)
    else:
        Result['Vm'] = tempres_avm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
        Result['Im'] = []
        Result['taxis'] = np.linspace(0,tstop-dt, Result['Vm'].size)
    return(Result)


def runexp():
    R = modelcell()
    R2 = modelcell(hasnmda=False)
    return (R, R2)

def plotexp(R, R2):
    plt.plot(R["taxis"], R["Im"])
    plt.plot(R2["taxis"], R2["Im"])
    plt.show()

if __name__ == "__main__":
    R, R2 = runexp()
    plotexp(R, R2)















