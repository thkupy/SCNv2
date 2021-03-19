#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New version of the SCN model, including much longer synaptic time-constants and more
realistic membrane time-constants.
This is mostly a demo program.

Created: Tuesday 23rd Feb 2021
@author: kuenzel
"""
#
from progress.bar import ChargingBar
import matplotlib.pyplot as plt
import numpy as np
import SCNv2

def runmain():
    #run models
    tstop = 500.0
    Ravc = SCNv2.runmodel(
        hasstimulation=(True, False),
        pstimulation=(10.0, 1, 250.0, 1.0, 1.0, 1, 250.0, 0.0),
        hasinputactivity=(False, False),
        haselectrode=True,
        pelectrode=(1, 0.0,tstop/2, -70.0, -70.0),
        hasfbi=False,
        )
    Rbvc = SCNv2.runmodel(
        hasstimulation=(False, True),
        pstimulation=(10.0, 1, 250.0, 0.0, 10.0, 1, 250.0, 1.0),
        hasinputactivity=(False, False),
        haselectrode=True,
        pelectrode=(1, 0.0,tstop/2, -70.0, -70.0),
        hasfbi=False,
        )
    Rabvc = SCNv2.runmodel(
        hasstimulation=(True, True),
        pstimulation=(10.0, 1, 250.0, 1.0, 10.0, 1, 250.0, 1.0),
        hasinputactivity=(False, False),
        haselectrode=True,
        pelectrode=(1, 0.0,tstop/2, -70.0, -70.0),
        hasfbi=False,
        )
    Ra = SCNv2.runmodel(
        hasstimulation=(True, False),
        pstimulation=(10.0, 1, 250.0, 1.0, 1.0, 1, 250.0, 0.0),
        hasinputactivity=(False, False),
        )
    Rb = SCNv2.runmodel(
        hasstimulation=(False, True),
        pstimulation=(10.0, 1, 250.0, 0.0, 10.0, 1, 250.0, 1.0),
        hasinputactivity=(False, False),
        )
    Rab = SCNv2.runmodel(
        hasstimulation=(True, True),
        pstimulation=(10.0, 1, 250.0, 1.0, 10.0, 1, 250.0, 1.0),
        hasinputactivity=(False, False),
        )
    nv = 0.33
    R_10_10 = SCNv2.runmodel(nsyna=10,nsynb=10, hasfbi=True, noiseval=nv, seed=32768)
    R_20_0 = SCNv2.runmodel(nsyna=20,nsynb=0, hasfbi=True, noiseval=nv, seed=32769)
    R_0_20 = SCNv2.runmodel(nsyna=0,nsynb=20, hasfbi=True, noiseval=nv, seed=32770)
    T = np.linspace(0.0,500.0-0.025, int(round(500.0/0.025)))
    #Plot stuff
    sh0=plt.subplot(3,3,1)
    plt.plot(T,Ravc["Im"],"k-")
    plt.title("Apical only")
    plt.ylabel("Membrane Current (nA)")
    plt.subplot(3,3,2, sharex=sh0, sharey=sh0)
    plt.plot(T,Rbvc["Im"],"k-")
    plt.title("Basal only")
    plt.subplot(3,3,3, sharex=sh0, sharey=sh0)
    plt.plot(T,Rabvc["Im"],"k-")
    plt.title("Apical+Basal")
    #
    sh1=plt.subplot(3,3,4)
    plt.plot(T,Ra["AVm"],"k-")
    plt.plot(T,Ra["SVm"],"r-")
    plt.ylabel("Membrane Potential (mV)")
    plt.subplot(3,3,5, sharex=sh1, sharey=sh1)
    plt.plot(T,Rb["AVm"],"k-")
    plt.plot(T,Rb["SVm"],"r-")
    plt.subplot(3,3,6, sharex=sh1, sharey=sh1)
    plt.plot(T,Rab["AVm"],"k-")
    plt.plot(T,Rab["SVm"],"r-")
    #
    sh4 = plt.subplot(3,3,7)
    sh4.set_ylim((-80,40))
    plt.plot(T,R_20_0["AVm"],"k-")
    plt.plot(T,R_20_0["SVm"],"r-")
    for iii in range(len(R_20_0["atv"])):
        x=R_20_0["atv"][iii]
        y=np.ones(x.size)*iii
        plt.plot(x,y,"b.")
    for iii in range(len(R_20_0["btv"])):
        x=R_20_0["btv"][iii]
        y=np.ones(x.size)*iii
        plt.plot(x,y,"g.")
    plt.xlabel("Membrane Potential (mV)")
    plt.ylabel("Time (ms)")
    plt.title("20 Apical / 0 Basal")
    #
    plt.subplot(3,3,8, sharex=sh4, sharey=sh4)
    plt.plot(T,R_0_20["AVm"],"k-")
    plt.plot(T,R_0_20["SVm"],"r-")
    for iii in range(len(R_0_20["atv"])):
        x=R_0_20["atv"][iii]
        y=np.ones(x.size)*iii
        plt.plot(x,y,"b.")
    for iii in range(len(R_0_20["btv"])):
        x=R_0_20["btv"][iii]
        y=np.ones(x.size)*iii
        plt.plot(x,y,"g.")
    plt.ylabel("Time (ms)")
    plt.title("0 Apical / 20 Basal")
    #
    plt.subplot(3,3,9, sharex=sh4, sharey=sh4)
    plt.plot(T,R_10_10["AVm"],"k-")
    plt.plot(T,R_10_10["SVm"],"r-")
    for iii in range(len(R_10_10["atv"])):
        x=R_10_10["atv"][iii]
        y=np.ones(x.size)*iii
        plt.plot(x,y,"b.")
    for iii in range(len(R_10_10["btv"])):
        x=R_10_10["btv"][iii]
        y=np.ones(x.size)*iii
        plt.plot(x,y,"g.")
    plt.ylabel("Time (ms)")
    plt.title("10 Apical / 10 Basal")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    runmain()
