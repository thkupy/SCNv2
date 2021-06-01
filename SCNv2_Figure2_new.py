#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This creates the new Figure 2... non-deterministic inputs
with realistic latencies. 

The command line args are:
    SCNv2_Figure2_new.py weload ncores nconds

Created: 2021-05-27
Revised: 
@author: kuenzel(at)bio2.rwth-aachen.de
"""

import sys
import os
import matplotlib
#matplotlib.use("Agg")
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
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    s = []
    spc = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisR = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=P["hasinputactivity"],
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x], P["bitv"][x]),
            inputstop=(P["astart"][x] + P["adur"][x], P["bstart"][x] + P["bdur"][x]),
            hasnmda=True,
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=P["hasffi"][x],
            inhw=P["inhw"][x],
            inhtau=P["inhtau"][x],
            inhdelay=P["inhdelay"][x],
            noiseval=P["noiseval"][x],
            reallatency=P["reallatency"][x],
        )
        S = SCNv2.SimpleDetectAP(
            thisR["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        s.append(S["PeakT"])
        spc[irep] = len(S["PeakT"])
        #
    print("x: " + str(x))
    return [
        np.mean(spc),
        np.std(spc),
        s,
    ]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonecondition, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def plotres(outputA, outputB, PA, PB):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats
    #
    fheight = 15  # cm
    fwidth = 20
    #contourlims = (1.0,333.0)
    #ncontours = 27
    fhandle = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    a_m = outputA[0][:, 0]
    a_s = outputA[0][:, 1]
    a_t = outputA[0][:, 2]
    b_m = outputB[0][:, 0]
    b_s = outputB[0][:, 1]
    b_t = outputB[0][:, 2]
    #
    plt.subplot(2,2,1)
    plt.errorbar(PA["afreq"], a_m, yerr=a_s / PA["nreps"][0])
    plt.subplot(2,2,2)
    plt.errorbar(PB["bfreq"], b_m, yerr=b_s / PB["nreps"][0])
    #
    plt.subplot(2,2,3)
    plt.xlim((0,PA["dur"][0]))
    plt.plot((PA["astart"][0], PA["astart"][0] + PA["adur"][0]), (-1, -1), "g-")
    for icond in range(PA["N"]):
        for irep in range(PA["nreps"][0]):
            ypos = icond + (irep / (2 * PA["nreps"][0]))
            xvals = np.array(a_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color="b", marker=".", markersize=3, linestyle=" ")
    #
    plt.subplot(2,2,4)
    plt.xlim((0,PB["dur"][0]))
    plt.plot((PB["bstart"][0], PB["bstart"][0] + PB["bdur"][0]), (-1, -1), "g-")
    for icond in range(PB["N"]):
        for irep in range(PB["nreps"][0]):
            ypos = icond + (irep / (2 * PB["nreps"][0]))
            xvals = np.array(b_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color="b", marker=".", markersize=3, linestyle=" ")
    plt.tight_layout()
    return fhandle

def getparams(
            ncores = 4,
            nconds = 7,
            nreps = 3,
            aon=True,
            astart=0.0, 
            adur=125.0, 
            afreqs=(35.0,65.0), 
            bon=False, 
            bstart=0.0, 
            bdur=125.0, 
            bfreqs=(75.0,400.0),
            reallatency=True,
        ):
        #Some fixed Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.025
        dur = 500.0
        nv = 0.9
        #ParametersA
        P = {}
        P["N"] = nconds
        P["cores"] = ncores
        P["TotalN"] = int(P["N"])#1d Experiment
        P["Number"] = np.arange(P["TotalN"],dtype=int)
        P["mp"] = True
        P["Seed"] = 36453
        P["AdvSeed"] = True
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        ###########################################
        P["hasinputactivity"] = (aon, bon)
        # Now define the variable parameters. The repeated = y, the tiled = x!!
        if aon:
            afreq = np.geomspace(afreqs[0], afreqs[1], nconds)
            aitv = np.round(1000.0 / afreq, 1)
        else:
            afreq = np.zeros(nconds)
            aitv =  np.zeros(nconds)
        if bon:
            bfreq = np.geomspace(bfreqs[0], bfreqs[1], nconds)
            bitv = np.round(1000.0 / bfreq, 1)
        else:
            bfreq = np.zeros(nconds)
            bitv =  np.zeros(nconds)
        P["nsyna"] = np.repeat(10, nconds)
        P["astart"] = np.repeat(astart, nconds)
        P["adur"] = np.repeat(adur, nconds)
        P["afreq"] = afreq
        P["aitv"] = aitv
        P["nsynb"] = np.repeat(10, nconds)
        P["bstart"] = np.repeat(bstart, nconds)
        P["bdur"] = np.repeat(bdur, nconds)
        P["bfreq"] = bfreq
        P["bitv"] = bitv
        #
        P["hasffi"] = np.repeat(True, nconds)
        P["inhw"] = np.repeat(0.00075, nconds)
        P["inhtau"] = np.repeat(120.0, nconds)
        P["inhdelay"] = np.repeat(5.0, nconds)
        P["reallatency"] = np.repeat(reallatency, nconds)
        return P


if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_Saliencetests.py weload ncores nconds nreps
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
    if os.path.isfile("./data/SCNv2_SaliencetestsA.npy") and weload:
        print("Loading Data!")
        outputA = np.load("./data/SCNv2_SaliencetestsA.npy", allow_pickle=True)
        outputB = np.load("./data/SCNv2_SaliencetestsB.npy", allow_pickle=True)
        PA = np.load("./data/SCNv2_Saliencetests_PA.npy", allow_pickle=True)
        PA = PA.tolist()
        PB = np.load("./data/SCNv2_Saliencetests_PB.npy", allow_pickle=True)
        PB = PB.tolist()
    else:
        PA = getparams(
            ncores = ncores,
            nconds = nconds,
            nreps = nreps,
            aon=True,
            bon=False,
        )
        PB = getparams(
            ncores = ncores,
            nconds = nconds,
            nreps = nreps,
            aon=False,
            bon=True,
        )
        # make go!
        outputA = []
        outputA.append(myMPhandler(PA))
        outputA = np.array(outputA, dtype=object)
        np.save("./data/SCNv2_SaliencetestsA.npy", outputA, allow_pickle=True)
        np.save("./data/SCNv2_Saliencetests_PA.npy", PA, allow_pickle=True)
        outputB = []
        outputB.append(myMPhandler(PB))
        outputB = np.array(outputB, dtype=object)
        np.save("./data/SCNv2_SaliencetestsB.npy", outputB, allow_pickle=True)
        np.save("./data/SCNv2_Saliencetests_PB.npy", PB, allow_pickle=True)
    #
    print("done")
    plotres(outputA, outputB, PA, PB)
    plt.show()
#    fhandle = plotres(
#        output=output,
#        P=P,
#        x=np.unique(P["nsyn"]),
#        y=np.unique(P["afreq"]),
#        xlabs=u"N Synapses",
#        ylabs=u"Mean Input Freq. (Hz)",
#    )
#    pp = PdfPages("./figures/SCNv2_Inputtests.pdf")
#    pp.savefig()
#    pp.close()


















