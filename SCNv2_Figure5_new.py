#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This creates the new Figure 5...
non-deterministic inputs with realistic latencies. 
Here we let a uni-/multimodal signal appear with a certain "input power" at 
different "distances from the model cell", i.e. we use the realistic latency and
add 1/343s for every m distance to the basal input.
We ask how weak/weak, weak/strong, strong/weak, strong/strong interact at different
distances.

The command line args are:
    SCNv2_Figure5_new.py weload ncores nconds

Created: 2021-06-07
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
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    xdel = P["dist"][x] * (1000.0 / 343.0)#1/343 extra s for every meter (in ms)
    sab = []
    spcab = np.zeros(P["nreps"][x])
    sa = []
    spca = np.zeros(P["nreps"][x])
    sb = []
    spcb = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisRAB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x] + xdel, P["bitv"][x]),
            inputstop=(P["astart"][x] + P["adur"][x], P["bstart"][x] + xdel + P["bdur"][x]),
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
        SAB = SCNv2.SimpleDetectAP(
            thisRAB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spcab[irep] = len(SAB["PeakT"])
        #
        thisRA = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, False),
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x] + xdel, P["bitv"][x]),
            inputstop=(P["astart"][x] + P["adur"][x], P["bstart"][x] + xdel + P["bdur"][x]),
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
        SA = SCNv2.SimpleDetectAP(
            thisRA["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spca[irep] = len(SA["PeakT"])
        #
        thisRB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(False, True),
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x] + xdel, P["bitv"][x]),
            inputstop=(P["astart"][x] + P["adur"][x], P["bstart"][x] + xdel + P["bdur"][x]),
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
        SB = SCNv2.SimpleDetectAP(
            thisRB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        spcb[irep] = len(SB["PeakT"])
        #
    print("x: " + str(x))
    return [
        np.mean(spca),
        np.std(spca),
        np.mean(spcb),
        np.std(spcb),
        np.mean(spcab),
        np.std(spcab),
        P["dist"][x],
        P["afreq"][x],
        P["bfreq"][x],
    ]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonecondition, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def plotres(outputA, PA):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats
    import scipy.ndimage as ndimage
    #
    import warnings
    warnings.filterwarnings("ignore")
    #
    responsethreshold = 2
    #
    fheight = 20  # cm
    fwidth = 14
    #contourlims = (1.0,333.0)
    #ncontours = 27
    a_m = np.array(outputA[0][:, 0], dtype=float)
    a_s = np.array(outputA[0][:, 1], dtype=float)
    b_m = np.array(outputA[0][:, 2], dtype=float)
    b_s = np.array(outputA[0][:, 3], dtype=float)
    ab_m = np.array(outputA[0][:, 4], dtype=float)
    ab_s = np.array(outputA[0][:, 5], dtype=float)
    dist = np.array(outputA[0][:, 6], dtype=float)
    afreq = np.array(outputA[0][:, 7], dtype=float)
    bfreq = np.array(outputA[0][:, 8], dtype=float)
    alldist = np.unique(dist)
    ndists = np.unique(dist).size
    nfreqs = 4
    a_m = np.reshape(a_m, (ndists, nfreqs))
    a_s = np.reshape(a_s, (ndists, nfreqs)) 
    b_m = np.reshape(b_m, (ndists, nfreqs))
    b_s = np.reshape(b_s, (ndists, nfreqs)) 
    ab_m = np.reshape(ab_m, (ndists, nfreqs))
    ab_s = np.reshape(ab_s, (ndists, nfreqs)) 
    dist = np.reshape(dist, (ndists, nfreqs)) 
    afreq = np.reshape(afreq, (ndists, nfreqs)) 
    bfreq = np.reshape(bfreq, (ndists, nfreqs)) 
    #
    filtsig = 0.66
    #a_am = ndimage.gaussian_filter(a_am, sigma=filtsig, order=0)
    #
    dotcol = ((.1, .1, .1), (.5, .5, .5))
    msize = 1
    dottype = ","
    #

    #X = np.linspace(0.0, 100.0, np.unique(freq).size)
    #cola = plt.cm.Blues((np.linspace(0.3,1,nconds)))
    #colb = plt.cm.Reds((np.linspace(0.3,1,nconds)))
    #colab =  plt.cm.gray((np.linspace(0.0,0.7,nconds)))
    #greens =  plt.cm.Greens((np.linspace(0.3,1,nconds)))
    #
    ###FIGURE VERSION 1 -line plots-
    fhandle1 = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    plt.errorbar(alldist, ab_m[:,0] - (a_m[:,0]+b_m[:,0]), yerr=ab_s[:,0], label="weak/weak")
    plt.errorbar(alldist, ab_m[:,1] - (a_m[:,1]+b_m[:,1]), yerr=ab_s[:,1], label="weak/strong")
    plt.errorbar(alldist, ab_m[:,2] - (a_m[:,2]+b_m[:,2]), yerr=ab_s[:,2], label="strong/weak")
    plt.errorbar(alldist, ab_m[:,3] - (a_m[:,3]+b_m[:,3]), yerr=ab_s[:,3], label="strong/strong")
    plt.legend()
    plt.tight_layout()
    #
    return fhandle1


def getparams(
            ncores=4,
            ndists=5,
            nreps=3,
        ):
        #Some fixed Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.01#0.025
        dur = 500.0
        nv = 0.9
        nfreqs = 4
        #
        #ParametersA
        P = {}
        P["N"] = ndists * nfreqs
        P["cores"] = ncores
        P["TotalN"] = int(P["N"])#2d Experiment
        P["Number"] = np.arange(P["TotalN"],dtype=int)
        P["mp"] = True
        P["Seed"] = 3786562
        P["AdvSeed"] = True
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        P["nsyna"] = np.repeat(25, P["TotalN"])#10 changed 2021-06-22
        P["astart"] = np.repeat(0.0, P["TotalN"])
        P["adur"] = np.repeat(125.0, P["TotalN"])
        P["nsynb"] = np.repeat(25, P["TotalN"])#10 changed 2021-06-22
        P["bstart"] = np.repeat(0.0, P["TotalN"])
        P["bdur"] = np.repeat(125.0, P["TotalN"])
        P["hasffi"] = np.repeat(True, P["TotalN"])
        P["inhw"] = np.repeat(0.001, P["TotalN"])## 0.00075 2021-06-21
        P["inhtau"] = np.repeat(75.0, P["TotalN"])##120.0 2021-06-21
        P["inhdelay"] = np.repeat(5.0, P["TotalN"])
        P["reallatency"] = np.repeat(True, P["TotalN"])
        P["hasinputactivity"] = (True, True)
        ###########################################
        #
        #P["dist"][x],
        #P["SalA"][x],
        #P["SalB"][x],
        #45/115,65/200,85/300
        dists = np.round(np.linspace(0, 100.0, ndists))
        # Now define the variable parameters. The repeated = y, the tiled = x!!
        afreq = np.array([45.0, 45.0, 85.0, 85.0])
        bfreq = np.array([115.0, 300.0, 115.0, 300.0])
        aitv = np.round(1000.0 / afreq, 1)
        bitv = np.round(1000.0 / bfreq, 1)
        P["afreq"] = np.tile(afreq, ndists)
        P["aitv"] = np.tile(aitv, ndists)
        P["bfreq"] = np.tile(bfreq, ndists)
        P["bitv"] = np.tile(bitv, ndists)
        #
        P["dist"] = np.repeat(dists, nfreqs)
        #
        return P


if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_Figure5_new.py weload ncores ndists nreps
    inputargs = sys.argv[1:]
    myargs = [1, 4, 7, 3]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    ncores = int(myargs[1])
    ndists = int(myargs[2])
    nreps = int(myargs[3])
    #------------------
    #Run the show
    if os.path.isfile("./data/SCNv2_Figure5_new.npy") and weload:
        print("Loading Data!")
        outputA = np.load("./data/SCNv2_Figure5_new.npy", allow_pickle=True)
        PA = np.load("./data/SCNv2_Figure5_new_P.npy", allow_pickle=True)
        PA = PA.tolist()
    else:
        PA = getparams(
            ncores=ncores,
            ndists=ndists,
            nreps=nreps,
        )
        outputA = []
        outputA.append(myMPhandler(PA))
        outputA = np.array(outputA, dtype=object)
        np.save("./data/SCNv2_Figure5_new.npy", outputA, allow_pickle=True)
        np.save("./data/SCNv2_Figure5_new_P.npy", PA, allow_pickle=True)
        #
    #
    print("done")
    #plotres(outputA, outputB, PA, PB)
    #plt.show()
    fhandle1 = plotres(outputA, PA)
    pp = PdfPages("./figures/SCNv2_Figure5_new.pdf")
    pp.savefig(fhandle1)
    pp.close()


















