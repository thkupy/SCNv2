#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates "Figure 1" which deals with deterministic inputs, i.e. simulated 
slice experiments. It shows the different thresholds for apical and basal and combined.

Created: 2021-05-27
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
    vman = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vman[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmb = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmb[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmab = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmab[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmabn = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmabn[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    sa = []
    san = []
    sb = []
    sab = []
    sabn = []
    useperc = np.linspace(0.0, 1.0, nconds)
    for thiscond in range(nconds):
        thisRA = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, False),
            pstimulation=(0.0, 1, tstop, useperc[thiscond], 0.0, 1, tstop, 0.0),
            hasinputactivity=(False, False),
            noiseval=0.1,
        )
        vma[thiscond + 1,:] = thisRA["AVm"]
        SA = SCNv2.SimpleDetectAP(
            thisRA["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sa.append(SA["PeakT"])
        #
        thisRAN = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, False),
            pstimulation=(0.0, 1, tstop, useperc[thiscond], 0.0, 1, tstop, 0.0),
            hasinputactivity=(False, False),
            noiseval=0.1,
            hasnmda=False,
        )
        vman[thiscond + 1,:] = thisRAN["AVm"]
        SAN = SCNv2.SimpleDetectAP(
            thisRAN["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        san.append(SAN["PeakT"])
        #
        thisRB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(False, True),
            pstimulation=(0.0, 1, tstop, 0.0, 0.0, 1, tstop, useperc[thiscond]),
            hasinputactivity=(False, False),
            noiseval=0.1,
        )
        vmb[thiscond + 1,:] = thisRB["AVm"]
        SB = SCNv2.SimpleDetectAP(
            thisRB["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sb.append(SB["PeakT"])
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
            noiseval=0.1,
        )
        vmab[thiscond + 1,:] = thisRAB["AVm"]
        SAB = SCNv2.SimpleDetectAP(
            thisRAB["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sab.append(SAB["PeakT"])
        #
        thisRABN = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, True),
            pstimulation=stimparams,
            hasinputactivity=(False, False),
            noiseval=0.1,
            hasnmda=False,
        )
        vmabn[thiscond + 1,:] = thisRABN["AVm"]
        SABN = SCNv2.SimpleDetectAP(
            thisRABN["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sabn.append(SABN["PeakT"])
        #
        bar.next()
    D = {
        "vma": vma, 
        "vman": vman,
        "vmb": vmb, 
        "vmab": vmab,
        "vmabn": vmabn,
        "sa": sa,
        "san": san,
        "sb": sb,
        "sab": sab,
        "sabn": sabn,
        }
    bar.finish()
    return(D)


def mynormalize(eingang):
    ausgang = np.array(eingang)
    ausgang = ausgang - np.min(ausgang)
    ausgang = ausgang / np.max(ausgang)
    return ausgang


def make_plots(D):
    nconds = D["vma"].shape[0] - 1
    if nconds > 10:
        plotconds = np.arange(0, nconds, int(nconds/10))
    else:
        plotconds = range(nconds)
    percs = np.linspace(0.0, 1.0, nconds) * 8.84
    fh = plt.figure()
    sh = fh.subplots(2, 3)
    mxa = []
    mxan = []
    mxb = []
    mxab = []
    mxabn = []
    for iii in range(nconds):
        mxa.append(np.max(D["vma"][iii + 1, :]))
        mxan.append(np.max(D["vman"][iii + 1, :]))
        mxb.append(np.max(D["vmb"][iii + 1, :]))
        mxab.append(np.max(D["vmab"][iii + 1, :]))
        mxabn.append(np.max(D["vmabn"][iii + 1, :]))
        if np.isin(iii, plotconds):
            sh[0, 0].plot(D["vma"][0, :], D["vma"][iii + 1, :], "b-")
            sh[0, 2].plot(D["vman"][0, :], D["vman"][iii + 1, :], "b--")
            sh[0, 1].plot(D["vmb"][0, :], D["vmb"][iii + 1, :], "r-")
            sh[1, 0].plot(D["vmab"][0, :], D["vmab"][iii + 1, :], "k-")
            sh[1, 1].plot(D["vmabn"][0, :], D["vmabn"][iii + 1, :], "k--")
    mxa = mynormalize(mxa)
    mxb = mynormalize(mxb)
    mxab = mynormalize(mxab)
    mxan = mynormalize(mxan)
    mxabn = mynormalize(mxabn)
    sh[0, 0].set_xlabel("Time (ms)")
    sh[0, 0].set_ylabel("Vm (mV)")
    sh[0, 0].set_title("Apical only")
    sh[0, 1].set_xlabel("Time (ms)")
    sh[0, 1].set_ylabel("Vm (mV)")
    sh[0, 1].set_title("Basal only")
    sh[0, 2].set_xlabel("Time (ms)")
    sh[0, 2].set_ylabel("Vm (mV)")
    sh[0, 2].set_title("Apical only, no NMDA")
    sh[1, 0].set_xlabel("Time (ms)")
    sh[1, 0].set_ylabel("Vm (mV)")
    sh[1, 0].set_title("Apical + Basal")
    sh[1, 1].set_xlabel("Time (ms)")
    sh[1, 1].set_ylabel("Vm (mV)")
    sh[1, 1].set_title("Apical + Basal, no NMDA")
    sh[1, 2].plot(percs, mxa, "b-")
    sh[1, 2].plot(percs, mxan, "b--")
    sh[1, 2].plot(percs, mxb, "r-")
    sh[1, 2].plot(percs, mxab, "k-")
    sh[1, 2].plot(percs, mxabn, "k--")
    sh[1, 2].set_xlabel("total g-syn (nS)")
    sh[1, 2].set_ylabel("max Vm (mV)")
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
    if weload and os.path.isfile("./data/Figure1_new.npy"):
        D = np.load("./data/Figure1_new.npy", allow_pickle=True)
        D = D.tolist()
    else:
        D = simulate_data(nconds, tstop, dt)
        np.save("./data/Figure1_new.npy", D, allow_pickle=True)
    make_plots(D)
