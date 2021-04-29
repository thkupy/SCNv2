#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This is another approach to look at integration.
Here, we always stimulate at a total frequency of 200Hz. However, the frequency is 
summed between the a and b inputs to give this value. Different ratios (200/0 -> 0/200)
are tested.

The command line args are:
    SCNv2_RatioInput weload ncores nconds nreps

Created: 2021-04-29, TK
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
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    amin = np.max(P["aitv"])
    bmin = np.max(P["bitv"])
    #
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    spra = np.zeros(P["nreps"][x])
    spran = np.zeros(P["nreps"][x])
    sprb = np.zeros(P["nreps"][x])
    sprab = np.zeros(P["nreps"][x])
    sprabn = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisRA = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(250.0, P["aitv"][x], 250.0+P["ab_delay"][x], bmin),
            inputstop=(750.0, 750.0 + P["ab_delay"][x]),
            hasnmda=True,
            seed=thisseed + irep,
            hasfbi=True,
            noiseval=P["noiseval"][x],
        )
        Sa = SCNv2.SimpleDetectAP(
            thisRA["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spra[irep] = len(Sa["PeakT"]) * (1000.0 / P["dur"][x])
        #
        thisRAN = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(250.0, P["aitv"][x], 250.0+P["ab_delay"][x], bmin),
            inputstop=(750.0, 750.0 + P["ab_delay"][x]),
            hasnmda=False,
            seed=thisseed + irep,
            hasfbi=True,
            noiseval=P["noiseval"][x],
        )
        San = SCNv2.SimpleDetectAP(
            thisRAN["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spran[irep] = len(Sa["PeakT"]) * (1000.0 / P["dur"][x])
        #
        thisRB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(250.0, amin, 250.0+P["ab_delay"][x], P["bitv"][x]),
            inputstop=(750.0, 750.0 + P["ab_delay"][x]),
            hasnmda=True,
            seed=thisseed + irep,
            hasfbi=True,
            noiseval=P["noiseval"][x],
        )
        Sb = SCNv2.SimpleDetectAP(
            thisRB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprb[irep] = len(Sb["PeakT"]) * (1000.0 / P["dur"][x])
        #
        thisRAB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(250.0, P["aitv"][x], 250.0+P["ab_delay"][x], P["bitv"][x]),
            inputstop=(750.0, 750.0 + P["ab_delay"][x]),
            hasnmda=True,
            seed=thisseed + irep,
            hasfbi=True,
            noiseval=P["noiseval"][x],
        )
        Sab = SCNv2.SimpleDetectAP(
            thisRAB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprab[irep] = len(Sab["PeakT"]) * (1000.0 / P["dur"][x])
        #-
        thisRABN = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(250.0, P["aitv"][x], 250.0+P["ab_delay"][x], P["bitv"][x]),
            inputstop=(750.0, 750.0 + P["ab_delay"][x]),
            hasnmda=False,
            seed=thisseed + irep,
            hasfbi=True,
            noiseval=P["noiseval"][x],
        )
        Sabn = SCNv2.SimpleDetectAP(
            thisRABN["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprabn[irep] = len(Sabn["PeakT"]) * (1000.0 / P["dur"][x])
        #-
    print((str(x)))
    return [
        np.mean(spra),
        np.std(spra),
        np.mean(spran),
        np.std(spran),
        np.mean(sprb),
        np.std(sprb),
        np.mean(sprab),
        np.std(sprab),
        np.mean(sprabn),
        np.std(sprabn),
    ]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonecondition, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def plotres(output, P, x, xlabs, ylabs):
    mycmap = "bone"
    nticks = 5
    fwidth = 8 # cm
    fhandle = plt.figure(figsize=(fwidth / 2.54, (2 * fwidth) / 2.54), dpi=600)
    #
    outspra_mean = output[0][:, 0]
    outspra_std = output[0][:, 1]
    outspran_mean = output[0][:, 2]
    outspran_std = output[0][:, 3]
    outsprb_mean = output[0][:, 4]
    outsprb_std = output[0][:, 5]
    outsprab_mean = output[0][:, 6]
    outsprab_std = output[0][:, 7]
    outsprabn_mean = output[0][:, 8]
    outsprabn_std = output[0][:, 9]
    #
    midindex = int(np.floor(P["N"] / 2))
    theoretical_combined = outspra_mean[midindex] + outsprb_mean[midindex]
    theoretical_combined_n = outspran_mean[midindex] + outsprb_mean[midindex]
    #
    ax1 = plt.subplot(211)
    ax1.plot(105.0, theoretical_combined, "gx")
    ax1.plot(105.0, theoretical_combined, "mx")
    ax1.errorbar(P["afreq"], outspra_mean, yerr=outspra_std, color="b")
    ax1.errorbar(P["afreq"], outspran_mean, yerr=outspran_std, color="b", linestyle="--")
    ax1.errorbar(P["bfreq"], outsprb_mean, yerr=outsprb_std, color="r")
    #ax1.errorbar(x, outsprab_mean, yerr=outsprab_std, color="k")
    ax1.set_xlabel("Mean input rate (Hz)")
    ax1.set_ylabel(ylabs)
    plt.legend(("apical", "basal", "apical+basal"))
    #
    ax2 = plt.subplot(212)
    ax2.plot((-200, 200), (theoretical_combined, theoretical_combined), "g-")
    ax2.plot((-200, 200), (theoretical_combined_
n, theoretical_combined_n), "g", linestyle="--")
    ax2.errorbar(x, outsprab_mean, yerr=outsprab_std, color="k")
    ax2.errorbar(x, outsprabn_mean, yerr=outsprabn_std, color="m")
    ax2.set_xlabel(xlabs)
    ax2.set_ylabel(ylabs)
    plt.legend(("w/ NMDA", "w/o NMDA"))
    #
    plt.tight_layout()
    return(fhandle)


if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_RatioInputs.py weload ncores nconds nreps
    inputargs = sys.argv[1:]
    myargs = [1, 4, 7, 3]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    ncores = int(myargs[1])
    nconds = int(myargs[2])
    nreps = int(myargs[3])
    #------------------
    #Run the show
    if os.path.isfile("./data/RatioInputs.npy") and weload:
        print("Data for SCNv2_RatioInputs found... loading!")
        output = np.load("./data/RatioInputs.npy", allow_pickle=True)
        P = np.load("./data/RatioInputs_P.npy", allow_pickle=True)
        P = P.tolist()
    else:
        #Some fixes Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.025
        dur = 1000.0
        nv = 1.0#
        delay = 0.0
        output = []
        P = {}
        P["N"] = nconds
        P["cores"] = ncores
        P["TotalN"] = int(P["N"])#this is a 1d Experiment!
        P["Number"] = list(range(P["TotalN"]))
        P["mp"] = True
        P["Seed"] = 3234388
        P["AdvSeed"] = True
        #########################################
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        P["ab_delay"] = np.repeat(delay, P["TotalN"])
        ###########################################
        P["afreq"] = np.linspace(10.0, 200.0, P["TotalN"])
        P["bfreq"] = np.linspace(200.0, 10.0, P["TotalN"])
        P["aitv"] = np.round(1000.0 / P["afreq"], 1)
        P["bitv"] = np.round(1000.0 / P["bfreq"], 1)
        #
        # make go!
        output.append(myMPhandler(P))
        output = np.array(output)
        np.save("./data/RatioInputs.npy", output, allow_pickle=True)
        np.save("./data/RatioInputs_P.npy", P, allow_pickle=True)
    #
    fhandle = plotres(
        output=output,
        P=P,
        x=P["afreq"]-P["bfreq"],
        xlabs=u"Apical-Basal Frequency Diff. (Hz)",
        ylabs=u"AP Rate (Hz)",
    )
    pp = PdfPages("./figures/RatioInputs.pdf")
    pp.savefig()
    pp.close()
