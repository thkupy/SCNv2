#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates the subpanels c of the first model figure:
    Simulated slice experiment, activate inputs deterministically and 
    demonstrate the time-window for interactions

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
    vmab = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmab[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    variedparam = np.linspace(-4.0, 4.0, nconds)
    for thiscond in range(nconds):
        stimparams = (
            25.0 + variedparam[thiscond], 1, tstop-30.0, 0.33,
            25.0, 1, tstop-30.0, 0.33,
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
    bar.finish()
    return(vmab)

def make_plots(D):
    nconds = D.shape[0] - 1
    if nconds > 10:
        plotconds = np.arange(0, nconds, int(nconds/10))
    else:
        plotconds = range(nconds)
    variedparam = np.linspace(-4.0, 4.0, nconds)
    fh = plt.figure()
    sh = fh.subplots(2)
    mxa = []
    mxb = []
    mxab = []
    col1 = np.linspace(0.8, 0.2, nconds)
    col = np.vstack((col1, col1, col1))
    sh[0].plot(25.0, -70, marker=".", color="r")
    for iii in range(nconds):
        mxab.append(np.max(D[iii + 1, :]))
        if np.isin(iii, plotconds):
            sh[0].plot(25.0 + variedparam[iii], -75, marker=".", color=col[:, iii])
            sh[0].plot(D[0, :], D[iii + 1, :], color=col[:, iii])
    mxab = np.array(mxab)
    sh[0].set_ylabel("Vm (mV)")
    sh[0].set_xlabel("Time (ms)")
    sh[1].plot(variedparam, mxab, "k-")
    indmid = np.abs(variedparam).argmin()
    firstabove = np.where(mxab>-20.0)[0][0]
    lastabove = np.where(mxab>-20.0)[0][-1]
    sh[1].plot((variedparam[indmid], variedparam[indmid], ), (-70, mxab[indmid],) , "r-")
    sh[1].plot((variedparam[firstabove-1], variedparam[lastabove]), (-20, -20), "r.-")
    sh[1].annotate(str(np.round(np.abs(variedparam[firstabove]),2)) + "ms",(-1.5,-30))
    sh[1].annotate(str(np.round(np.abs(variedparam[lastabove]),2)) + "ms",(0.75,-30))
    sh[1].set_ylabel("Vmax (mV)")
    sh[1].set_xlabel("Onset delay (ms)")
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
    if weload and os.path.isfile("./data/Figure1c.npy"):
        D = np.load("./data/Figure1c.npy", allow_pickle=True)
    else:
        D = simulate_data(nconds, tstop, dt)
        np.save("./data/Figure1c.npy", D, allow_pickle=True)
    make_plots(D)
