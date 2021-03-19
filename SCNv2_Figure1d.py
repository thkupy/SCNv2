#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates the subpanels d of the first model figure:
    Show driven apical vs. spont basal, driven basal vs. spont apical
    and driven multimodal.

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


def simulate_data(freq, nreps, tstop, dt):
    thr = -50
    baserate = 100.0
    nv = 0.9
    bar = ChargingBar("Simulation running", max=nreps)
    vma = np.zeros((nreps+1, int(np.ceil(tstop / dt))))
    spa = []
    vmb = np.zeros((nreps+1, int(np.ceil(tstop / dt))))
    spb = []
    vmab = np.zeros((nreps+1, int(np.ceil(tstop / dt))))
    spab = []
    vmabn = np.zeros((nreps+1, int(np.ceil(tstop / dt))))
    spabn = []
    vma[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmb[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmab[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmabn[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    for thisrep in range(nreps):
        thisRA = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, freq, 50.0, baserate),
            inputstop=(tstop, tstop),
            hasnmda=True,
            hasfbi=True,
            seed=32768+thisrep,
            noiseval=nv,
        )
        vma[thisrep + 1,:] = thisRA["AVm"]
        Sa = SCNv2.SimpleDetectAP(thisRA["AVm"],thr=thr,dt=dt,LM=-20,RM=10)
        spa.append(Sa['PeakT'])
        #-
        thisRB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, baserate, 50.0, freq),
            inputstop=(tstop, tstop),
            hasnmda=True,
            hasfbi=True,
            seed=62768+thisrep,
            noiseval=nv,
        )
        vmb[thisrep + 1,:] = thisRB["AVm"]
        Sb = SCNv2.SimpleDetectAP(thisRB["AVm"],thr=thr,dt=dt,LM=-20,RM=10)
        spb.append(Sb['PeakT'])
        #-
        thisRAB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, freq, 50.0, freq),
            inputstop=(tstop, tstop),
            hasnmda=True,
            seed=92768+thisrep,
            hasfbi=True,
            noiseval=nv,
        )
        vmab[thisrep + 1,:] = thisRAB["AVm"]
        Sab = SCNv2.SimpleDetectAP(thisRAB["AVm"],thr=thr,dt=dt,LM=-20,RM=10)
        spab.append(Sab['PeakT'])
        #-
        thisRABN = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, freq, 50.0, freq),
            inputstop=(tstop, tstop),
            hasnmda=False,
            seed=92768+thisrep,
            hasfbi=True,
            noiseval=nv,
        )
        vmabn[thisrep + 1,:] = thisRABN["AVm"]
        Sabn = SCNv2.SimpleDetectAP(thisRABN["AVm"],thr=thr,dt=dt,LM=-20,RM=10)
        spabn.append(Sabn['PeakT'])
        #-
        bar.next()
    bar.finish()
    R = {"vma": vma, "spa": spa, 
        "vmb": vmb, "spb": spb, 
        "vmab": vmab, "spab": spab,
        "vmabn": vmabn, "spabn": spabn,
        "tstop": tstop, "dt": dt, "freq": freq, "nreps": nreps,
    }
    return(R)

def make_plots(D):
    fh = plt.figure()
    sh = []
    sh.append(fh.add_subplot(5,1,1))
    sh.append(fh.add_subplot(5,1,2, sharex=sh[0], sharey=sh[0]))
    sh.append(fh.add_subplot(5,1,3, sharex=sh[0], sharey=sh[0]))
    sh.append(fh.add_subplot(5,1,4, sharex=sh[0], sharey=sh[0]))
    sh.append(fh.add_subplot(5,1,5))
    sh[0].plot(D["vma"][0,:], D["vma"][1,:], "b-")
    sh[0].plot(D["spa"][0], np.ones(len(D["spa"][0])) * 10.0, "g|")
    sh[1].plot(D["vmb"][0,:], D["vmb"][1,:], "r-")
    sh[1].plot(D["spb"][0], np.ones(len(D["spb"][0])) * 10.0, "g|")
    sh[2].plot(D["vmab"][0,:], D["vmab"][1,:], "k-")
    sh[2].plot(D["spab"][0], np.ones(len(D["spab"][0])) * 10.0, "g|")
    sh[3].plot(D["vmabn"][0,:], D["vmabn"][1,:], "m-")
    sh[3].plot(D["spabn"][0], np.ones(len(D["spabn"][0])) * 10.0, "g|")
    #
    spa_all = np.concatenate(D["spa"], axis=0)
    spb_all = np.concatenate(D["spb"], axis=0)
    spab_all = np.concatenate(D["spab"], axis=0)
    spabn_all = np.concatenate(D["spabn"], axis=0)
    ratesa = np.zeros(D["nreps"])
    ratesb = np.zeros(D["nreps"])
    ratesab = np.zeros(D["nreps"])
    ratesabn = np.zeros(D["nreps"])
    for irep in range(D["nreps"]):
        a = np.array(D["spa"][irep])
        b = np.array(D["spb"][irep])
        ab = np.array(D["spab"][irep])
        abn = np.array(D["spabn"][irep])
        ratesa[irep] = (a[a > 50.0].size) * (1000.0 / (D["tstop"]))
        ratesb[irep] = (b[b > 50.0].size) * (1000.0 / (D["tstop"]))
        ratesab[irep] = (ab[ab > 50.0].size) * (1000.0 / (D["tstop"]))
        ratesabn[irep] = (abn[abn > 50.0].size) * (1000.0 / (D["tstop"]))
    #ratea = ((spa_all[spa_all>50.0].size) / D["nreps"]) * (1000.0 / (D["tstop"]))
    #rateb = ((spb_all[spb_all>50.0].size) / D["nreps"]) * (1000.0 / (D["tstop"]))
    #rateab = ((spab_all[spab_all>50.0].size) / D["nreps"]) * (1000.0 / (D["tstop"]))
    hista = np.histogram(spa_all,np.linspace(0, D["tstop"], int((D["tstop"] / 10)+1)))
    histb = np.histogram(spb_all,np.linspace(0, D["tstop"], int((D["tstop"] / 10)+1)))
    histab = np.histogram(spab_all,np.linspace(0, D["tstop"], int((D["tstop"] / 10)+1)))
    histabn = np.histogram(spabn_all,np.linspace(0, D["tstop"], int((D["tstop"] / 10)+1)))
    #
    atxt = str(round(np.mean(ratesa), 1)) + "+/-" + str(round(np.std(ratesa), 1)) + "AP/s"
    btxt = str(round(np.mean(ratesb), 1)) + "+/-" + str(round(np.std(ratesb), 1)) + "AP/s"
    abtxt = str(round(np.mean(ratesab), 1)) + "+/-" + str(round(np.std(ratesab), 1)) + "AP/s"
    abntxt = str(round(np.mean(ratesabn), 1)) + "+/-" + str(round(np.std(ratesabn), 1)) + "AP/s"
    sh[0].annotate(atxt, (0.0, 0))
    sh[1].annotate(btxt, (0.0, 0))
    sh[2].annotate(abtxt, (0.0, 0))
    sh[3].annotate(abntxt, (0.0, 0))
    sh[4].plot(hista[1][1:],hista[0], "b-")
    sh[4].plot(histb[1][1:],histb[0], "r-")
    sh[4].plot(histab[1][1:],hista[0] + histb[0], color=(.7,.7,.7), linewidth=2)
    sh[4].plot(histab[1][1:],histab[0], "k-", linewidth=2)
    sh[4].plot(histabn[1][1:],histabn[0], color="m", linewidth=2)
    sh[0].set_ylabel("Vm (mV)")
    sh[0].set_title("Apical driven, basal spont.")
    sh[1].set_ylabel("Vm (mV)")
    sh[1].set_title("Apical spont., basal driven")
    sh[2].set_ylabel("Vm (mV)")
    sh[2].set_title("Apical + basal driven")
    sh[3].set_ylabel("Vm (mV)")
    sh[3].set_title("Apical + basal driven, no NMDA")
    sh[4].set_ylabel("Events/bin")
    sh[4].set_xlabel("Time (ms)")
    #
    #
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Parse command-line arguments [first: whether to load or (re-)simulate]
    inputargs = sys.argv[1:]
    myargs = [1, 12.5, 10, 550.0, 0.025]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    freq = float(myargs[1])
    nreps = int(myargs[2])
    tstop = float(myargs[3])
    dt = float(myargs[4])
    #------------------
    #Run the show
    if weload and os.path.isfile("./data/Figure1d.npy"):
        D = np.load("./data/Figure1d.npy", allow_pickle=True)
        D = D.tolist()
    else:
        D = simulate_data(freq, nreps, tstop, dt)
        np.save("./data/Figure1d.npy", D, allow_pickle=True)
    make_plots(D)
