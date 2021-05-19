#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates test data on the number of output spikes generated for different
input rates (a, b, a+b) for different conductances of the "new" (as of 19.5.21)
inhibitory feed-forward model.

We will always test a given set of input rates (10,20,50,100,200,333 Hz) for N=10
synapses per "modality". The condition ninhib is the number of inibitory conductances
between weight = 0.0005 (0.5nS) and 0.01 (10nS) per inhibitory input.

The command line args are:
    SCNv2_Inhibtests.py weload ncores ninhib nrep

Created: 2021-05-19
Revised: 
@author: kuenzel(at)bio2.rwth-aachen.de
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from functools import partial
import SCNv2

# some global plot settings
plt.ioff()
plt.rc("font", family="serif", serif="Linux Libertine O")
#plt.rcParams["pdf.fonttype"] = 42
plt.rc("text", usetex=True)
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("axes", labelsize="small")
#new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
#plt.rcParams.update(new_rc_params)

def runonecondition(x, P):
    baserate = 300
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    spra = np.zeros(P["nreps"][x])
    sprb = np.zeros(P["nreps"][x])
    sprab = np.zeros(P["nreps"][x])
    spran = np.zeros(P["nreps"][x])
    sprabn = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisRA = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, baserate),
            inputstop=(950.0, 950.0),
            hasnmda=True,
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=True,
            inhw=P["inhw"][x],
        )
        Sa = SCNv2.SimpleDetectAP(
            thisRA["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spra[irep] = len(Sa["PeakT"])/0.9
        #
        thisRB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, baserate, 50.0, P["bitv"][x]),
            inputstop=(950.0, 950.0),
            hasnmda=True,
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=True,
            inhw=P["inhw"][x],
        )
        Sb = SCNv2.SimpleDetectAP(
            thisRB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprb[irep] = len(Sb["PeakT"])/0.9
        #
        thisRAB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, P["bitv"][x]),
            inputstop=(950.0, 950.0),
            hasnmda=True,
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=True,
            inhw=P["inhw"][x],
        )
        Sab = SCNv2.SimpleDetectAP(
            thisRAB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprab[irep] = len(Sab["PeakT"])/0.9
        #
        thisRAN = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, baserate),
            inputstop=(950.0, 950.0),
            hasnmda=False,
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=True,
            inhw=P["inhw"][x],
        )
        San = SCNv2.SimpleDetectAP(
            thisRAN["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spran[irep] = len(San["PeakT"])/0.9
        #
        thisRABN = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, P["bitv"][x]),
            inputstop=(950.0, 950.0),
            hasnmda=False,
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=True,
            inhw=P["inhw"][x],
        )
        Sabn = SCNv2.SimpleDetectAP(
            thisRABN["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprabn[irep] = len(Sabn["PeakT"])/0.9
        #
    print("x: " + str(x))
    return [
        np.mean(spra),
        np.std(spra),
        np.mean(sprb),
        np.std(sprb),
        np.mean(sprab),
        np.std(sprab),
        np.mean(spran),
        np.std(spran),
        np.mean(sprabn),
        np.std(sprabn),
        P["afreq"][x],
        P["inhw"][x],
    ]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonecondition, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def plotres(output, P):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats
    #
    fwidth = 18
    fheight = 21
    fhandle = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))
    #
    ra_m = output[0][:, 0]
    ra_s = output[0][:, 1]
    rb_m = output[0][:, 2]
    rb_s = output[0][:, 3]
    rab_m = output[0][:, 4]
    rab_s = output[0][:, 5]
    ran_m = output[0][:, 6]
    ran_s = output[0][:, 7]
    rabn_m = output[0][:, 8]
    rabn_s = output[0][:, 9]
    freqs = output[0][:, 10]
    inhw = output[0][:, 11]
    myweights = np.unique(inhw)
    #
    plt.subplot(3,2,1)
    plt.xscale("log")
    for iw in range(myweights.size):
        plt.errorbar(
            freqs[inhw==myweights[iw]],
            ra_m[inhw==myweights[iw]],
            yerr=ra_s[inhw==myweights[iw]],
            linewidth=0.5,
            marker='.',
            markersize=3,
        )
    plt.legend(np.round(myweights*1000.0, 2))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("AP Output Rate (Hz)")
    plt.title("Apical only")
    #
    plt.subplot(3,2,2)
    plt.xscale("log")
    for iw in range(myweights.size):
        plt.errorbar(
            freqs[inhw==myweights[iw]],
            rb_m[inhw==myweights[iw]],
            yerr=rb_s[inhw==myweights[iw]],
            linewidth=0.5,
            marker='.',
            markersize=3,
        )
    plt.legend(np.round(myweights*1000.0, 2))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("AP Output Rate (Hz)")
    plt.title("Basal only")
    #
    plt.subplot(3,2,3)
    plt.xscale("log")
    for iw in range(myweights.size):
        plt.errorbar(
            freqs[inhw==myweights[iw]],
            rab_m[inhw==myweights[iw]],
            yerr=rab_s[inhw==myweights[iw]],
            linewidth=0.5,
            marker='.',
            markersize=3,
        )
    plt.legend(np.round(myweights*1000.0, 2))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("AP Output Rate (Hz)")
    plt.title("Apical + Basal")
    #
    plt.subplot(3,2,5)
    plt.xscale("log")
    for iw in range(myweights.size):
        plt.errorbar(
            freqs[inhw==myweights[iw]],
            ran_m[inhw==myweights[iw]],
            yerr=ran_s[inhw==myweights[iw]],
            linewidth=0.5,
            marker='.',
            markersize=3,
        )
    plt.legend(np.round(myweights*1000.0, 2))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("AP Output Rate (Hz)")
    plt.title("Apical only, no NMDA")
    #
    plt.subplot(3,2,6)
    plt.xscale("log")
    for iw in range(myweights.size):
        plt.errorbar(
            freqs[inhw==myweights[iw]],
            rabn_m[inhw==myweights[iw]],
            yerr=rabn_s[inhw==myweights[iw]],
            linewidth=0.5,
            marker='.',
            markersize=3,
        )
    plt.legend(np.round(myweights*1000.0, 2))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("AP Output Rate (Hz)")
    plt.title("Apical + Basal, no NMDA")
    #
    plt.tight_layout()
    #plt.show()#debug
    return fhandle

if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_Inhibtests.py weload ncores ninhib nreps
    inputargs = sys.argv[1:]
    myargs = [1, 4, 5, 3]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    ncores = int(myargs[1])
    ninhib = int(myargs[2])
    nreps = int(myargs[3])
    #------------------
    #Run the show
    if os.path.isfile("./data/SCNv2_Inhibtests.npy") and weload:
        print("Data for SCNv2_Inhibtests found... loading!")
        output = np.load("./data/SCNv2_Inhibtests.npy", allow_pickle=True)
        P = np.load("./data/SCNv2_Inhibtests_P.npy", allow_pickle=True)
        P = P.tolist()
    else:
        #Some fixes Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.025
        dur = 1000.0
        nv = 0.9
        output = []
        P = {}
        nfreqs = 9
        P["cores"] = ncores
        P["TotalN"] = int(nfreqs * ninhib)
        P["Number"] = np.arange(P["TotalN"],dtype=int)
        P["mp"] = True
        P["Seed"] = 322453
        P["AdvSeed"] = True
        #########################################
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        P["ab_delay"] = np.repeat(0.0, P["TotalN"])
        ###########################################
        # Now define the two variable parameters. The repeated = y, the tiled = x!!
        freq = np.array((10, 20, 35, 50, 75, 100, 150, 200, 333))
        itv = np.round(1000.0 / freq)
        P["afreq"] = np.repeat(np.round(freq), ninhib) 
        P["aitv"] = np.repeat(itv, ninhib)
        P["bfreq"] = np.repeat(np.round(freq), ninhib) 
        P["bitv"] = np.repeat(itv, ninhib)
        weights = np.linspace(0.0005, 0.005, ninhib)
        P["inhw"] = np.tile(weights, nfreqs)
        # make go!
        output.append(myMPhandler(P))
        output = np.array(output)
        np.save("./data/SCNv2_Inhibtests.npy", output, allow_pickle=True)
        np.save("./data/SCNv2_Inhibtests_P.npy", P, allow_pickle=True)
    #
    fhandle = plotres(output=output, P=P)
    pp = PdfPages("./figures/SCNv2_Inhibtests.pdf")
    pp.savefig()
    pp.close()


















