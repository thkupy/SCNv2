#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates the subpanels b of the first model figure:
    Simulated slice experiment, activate inputs deterministically and 
    demonstrate the different "threshold" for apical vs basal vs bimodal

Created: 2021-03-17
Revised: 
@author: kuenzel(at)bio2.rwth-aachen.de
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import ChargingBar
import SCNv2


def simulate_data(nconds, tstop, dt):
    bar = ChargingBar("Simulation running", max=nconds)
    vma = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vma[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmb = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmb[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmab = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmab[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    useperc = np.linspace(0.0, 1.0, nconds)
    for thiscond in range(nconds):
        thisRA = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, False),
            pstimulation=(0.0, 1, tstop, useperc[thiscond], 0.0, 1, tstop, 0.0),
            hasinputactivity=(False, False),
            hasfbi=False,
        )
        vma[thiscond + 1,:] = thisRA["AVm"]
        #
        thisRB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(False, True),
            pstimulation=(0.0, 1, tstop, 0.0, 0.0, 1, tstop, useperc[thiscond]),
            hasinputactivity=(False, False),
            hasfbi=False,
        )
        vmb[thiscond + 1,:] = thisRB["AVm"]
        #
        stimparams = (
            0.0, 1, tstop, useperc[thiscond] * 0.5,
            0.0, 1, tstop, useperc[thiscond] * 0.5,
        )
        thisRAB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, True),
            pstimulation=stimparams,
            hasinputactivity=(False, False),
            hasfbi=False,
        )
        vmab[thiscond + 1,:] = thisRAB["AVm"]
        bar.next()
    D = {"vma": vma, "vmb": vmb, "vmab": vmab}
    bar.finish()
    return(D)

def make_plots(D):
    nconds = D["vma"].shape[0] - 1
    if nconds > 10:
        plotconds = np.arange(0, nconds, int(nconds/10))
    else:
        plotconds = range(nconds)
    percs = np.linspace(0.0, 1.0, nconds) * 8.84
    fh = plt.figure()
    sh = fh.subplots(2, 2)
    mxa = []
    mxb = []
    mxab = []
    for iii in range(nconds):
        mxa.append(np.max(D["vma"][iii + 1, :]))
        mxb.append(np.max(D["vmb"][iii + 1, :]))
        mxab.append(np.max(D["vmab"][iii + 1, :]))
        if np.isin(iii, plotconds):
            sh[0, 0].plot(D["vma"][0, :], D["vma"][iii + 1, :], "b-")
            sh[0, 1].plot(D["vmb"][0, :], D["vmb"][iii + 1, :], "r-")
            sh[1, 0].plot(D["vmab"][0, :], D["vmab"][iii + 1, :], "k-")
    sh[0, 0].set_xlabel("Time (ms)")
    sh[0, 0].set_ylabel("Vm (mV)")
    sh[0, 0].set_title("Apical only")
    sh[0, 1].set_xlabel("Time (ms)")
    sh[0, 1].set_ylabel("Vm (mV)")
    sh[0, 1].set_title("Basal only")
    sh[1, 0].set_xlabel("Time (ms)")
    sh[1, 0].set_ylabel("Vm (mV)")
    sh[1, 0].set_title("Apical + Basal")
    sh[1, 1].plot(percs, mxa, "b-")
    sh[1, 1].plot(percs, mxb, "r-")
    sh[1, 1].plot(percs, mxab, "k-")
    sh[1, 1].set_xlabel("total g-syn (nS)")
    sh[1, 1].set_ylabel("max Vm (mV)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Parse command-line arguments [first: whether to load or (re-)simulate]
    inputargs = sys.argv[1:]
    myargs = [1, 10, 250.0, 0.025]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    nconds = int(myargs[1])
    tstop = float(myargs[2])
    dt = float(myargs[3])
    #------------------
    #Run the show
    if weload and os.path.isfile("./data/Figure1b.npy"):
        D = np.load("./data/Figure1b.npy", allow_pickle=True)
        D = D.tolist()
    else:
        D = simulate_data(nconds, tstop, dt)
        np.save("./data/Figure1b.npy", D, allow_pickle=True)
    make_plots(D)
