#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This creates the new Figure 3...
varying the temporal relation of the inputs at best enhancement condition

The command line args are:
    SCNv2_Figure3_new.py weload ncores nconds salience

Created: 2021-06-06
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
    sab = []
    sa = []
    sb = []
    spcab = np.zeros(P["nreps"][x])
    spca = np.zeros(P["nreps"][x])
    spcb = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisRAB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
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
        SAB = SCNv2.SimpleDetectAP(
            thisRAB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sab.append(SAB["PeakT"])
        spcab[irep] = len(SAB["PeakT"])
        #
        thisRA = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, False),
            pinputactivity=(P["astart"][x], P["aitv_2x"][x], P["bstart"][x], P["bitv"][x]),
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
        SA = SCNv2.SimpleDetectAP(
            thisRA["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sa.append(SA["PeakT"])
        spca[irep] = len(SA["PeakT"])
        #
        thisRB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyna"][x]),
            nsynb=int(P["nsynb"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(False, True),
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x], P["bitv_2x"][x]),
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
        SB = SCNv2.SimpleDetectAP(
            thisRB["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sb.append(SB["PeakT"])
        spcb[irep] = len(SB["PeakT"])
        #
    print("x: " + str(x))
    return [
        np.mean(spcab),
        np.std(spcab),
        np.mean(spca),
        np.std(spca),
        np.mean(spcb),
        np.std(spcb),
        sab,
        sa,
        sb,
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
    #
    responsethreshold = 2
    #
    import warnings
    warnings.filterwarnings("ignore")
    #
    fheight = 15  # cm
    fwidth = 10
    #contourlims = (1.0,333.0)
    #ncontours = 27
    fhandle = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    ab_m = outputA[0][:, 0]
    ab_s = outputA[0][:, 1]
    a_m = outputA[0][:, 2]
    a_s = outputA[0][:, 3]
    b_m = outputA[0][:, 4]
    b_s = outputA[0][:, 5]
    ab_t = outputA[0][:, 6]
    a_t = outputA[0][:, 7]
    b_t = outputA[0][:, 8]
    #
    dotcol = ((.1, .1, .1), (.5, .5, .5))
    msize = 2
    dottype = ","
    #
    sp1 = plt.subplot(4,2,(1,3))
    plt.title("Unimodal, \n Salience x2")
    plt.xlim((0,PA["dur"][0]))
    plt.plot(PA["astart"], np.ones(PA["astart"].size) * -1, "bo")
    plt.plot(PA["bstart"], np.ones(PA["bstart"].size) * -2, "ro")
    a_ttfsp = np.zeros((PA["N"],2))
    b_ttfsp = np.zeros((PA["N"],2))
    for icond in range(PA["N"]):
        plt.plot([0, PA["dur"][0]], [icond, icond], "k--", linewidth=0.5)
        ttfspa = np.ones(PA["nreps"][0])
        ttfspb = np.ones(PA["nreps"][0])
        for irep in range(PA["nreps"][0]):
            ypos = icond + (irep / (PA["nreps"][0]))
            xvals_a = np.array(a_t[icond][irep])
            yvals_a = np.ones(xvals_a.size) * ypos
            xvals_b = np.array(b_t[icond][irep])
            yvals_b = np.ones(xvals_b.size) * ypos
            plt.plot(xvals_a, yvals_a, color="b", marker=dottype, markersize=msize, linestyle=" ")
            plt.plot(xvals_b, yvals_b, color="r", marker=dottype, markersize=msize, linestyle=" ")
            if xvals_a.size > 0:
                ttfspa[irep] = np.min(xvals_a)#no spontaneous activity!
            else:
                ttfspa[irep] = np.nan
            if xvals_b.size > 0:
                ttfspb[irep] = np.min(xvals_b)#no spontaneous activity!
            else:
                ttfspb[irep] = np.nan
        a_ttfsp[icond,0]=np.nanmean(ttfspa)
        a_ttfsp[icond,1]=np.nanstd(ttfspa)
        b_ttfsp[icond,0]=np.nanmean(ttfspb)
        b_ttfsp[icond,1]=np.nanstd(ttfspb)
    sp1.set_yticks(np.arange(PA["N"])+0.5)
    sp1.set_yticklabels(np.round(PA["bstart"],1))
    sp1.set_xlim(0, PA["dur"][0])
    plt.xlabel("Time (ms)")
    plt.ylabel("basal onset time (ms)")
    #
    sp2 = plt.subplot(4,2,(2,4))
    plt.title("Multimodal")
    plt.xlim((0,PA["dur"][0]))
    plt.plot(PA["astart"], np.ones(PA["astart"].size) * -1, "bo")
    plt.plot(PA["bstart"], np.ones(PA["bstart"].size) * -2, "ro")
    ab_ttfsp = np.zeros((PA["N"],2))
    for icond in range(PA["N"]):
        plt.plot([0, PA["dur"][0]], [icond, icond], "k--", linewidth=0.5)
        ttfspab = np.ones(PA["nreps"][0])
        for irep in range(PA["nreps"][0]):
            ypos = icond + (irep / (PA["nreps"][0]))
            xvals_ab = np.array(ab_t[icond][irep])
            yvals_ab = np.ones(xvals_ab.size) * ypos
            plt.plot(xvals_ab, yvals_ab, color=dotcol[icond%2], marker=dottype, markersize=msize, linestyle=" ")
            if xvals_ab.size > 0:
                ttfspab[irep] = np.min(xvals_ab)#no spontaneous activity!
            else:
                ttfspab[irep] = np.nan
        ab_ttfsp[icond,0]=np.nanmean(ttfspab)
        ab_ttfsp[icond,1]=np.nanstd(ttfspab)
    sp2.set_yticks(np.arange(PA["N"])+0.5)
    sp2.set_yticklabels(np.round(PA["bstart"],1))
    sp2.set_xlim(0, PA["dur"][0])
    plt.xlabel("Time (ms)")
    plt.ylabel("basal onset time (ms)")
    #
    sp5 = plt.subplot(4,2,5)
    plt.errorbar(
        PA["bstart"],
        a_m,
        yerr=a_s,
        marker="o",
        color="b",
        markerfacecolor="w",
        markersize=4,
    )
    plt.errorbar(
        PA["bstart"],
        b_m,
        yerr=b_s,
        marker="o",
        color="r",
        markerfacecolor="w",
        markersize=4,
    )
    plt.plot(PA["bstart"],a_m + b_m,"g-")
    sp5.set_xlim(0, PA["dur"][0])
    plt.xlabel("Onset Time (ms)")
    plt.ylabel("Mean Output (APs)")
    #
    sp6 = plt.subplot(4,2,6, sharex=sp5, sharey=sp5)
    plt.errorbar(
        PA["bstart"],
        ab_m,
        yerr=ab_s,
        marker="o",
        color="k",
        markerfacecolor="w",
        markersize=4,
    )
    plt.plot(PA["bstart"],a_m + b_m,"g-")
    sp6.set_xlim(0, PA["dur"][0])
    plt.legend(("sum of unimodal","multimodal"), fontsize=5)
    plt.xlabel("Onset Time (ms)")
    plt.ylabel("Mean Output (APs)")
    #
    sp7 = plt.subplot(4,2,7)
    plt.errorbar(
        PA["bstart"],
        a_ttfsp[:,0]-PA["astart"],
        yerr=a_ttfsp[:,1],
        marker="o",
        color="b",
        markerfacecolor="w",
        markersize=4,
    )
    plt.errorbar(
        PA["bstart"],
        b_ttfsp[:,0]-PA["bstart"],
        yerr=b_ttfsp[:,1],
        marker="o",
        color="r",
        markerfacecolor="w",
        markersize=4,
    )
    plt.xlabel("Onset Time (ms)")
    plt.ylabel("Mean FSL (ms)")
    #
    sp8 = plt.subplot(4,2,8)
    plt.errorbar(
        PA["bstart"],
        ab_ttfsp[:,0]-PA["astart"],
        yerr=ab_ttfsp[:,1],
        marker="o",
        color="k",
        markerfacecolor="w",
        markersize=4,
    )
    plt.xlabel("Onset Time (ms)")
    plt.ylabel("Mean FSL (ms)")
    #
    plt.tight_layout()
    return fhandle

def getparams1d(
            ncores = 4,
            nconds = 7,
            nreps = 3,
            a_varies=False,
            astart=250.0, 
            adur=125.0, 
            b_varies=True,
            bstart=(200.0, 625.0), 
            bdur=125.0, 
            salience = 40.0,
            reallatency=True,
        ):
        #Some fixed Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.025
        dur = 1000.0
        nv = 0.9
        #ParametersA
        P = {}
        P["N"] = nconds
        P["cores"] = ncores
        P["TotalN"] = int(P["N"])#1d Experiment
        P["Number"] = np.arange(P["TotalN"],dtype=int)
        P["mp"] = True
        P["Seed"] = 77253
        P["AdvSeed"] = True
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        ###########################################
        afreqs = np.geomspace(25.0, 70.0, nconds)
        aitvs = np.round(1000.0 / afreqs, 1)
        bfreqs = np.geomspace(65.0, 400.0, nconds)
        bitvs = np.round(1000.0 / bfreqs, 1)
        allsal = np.linspace(0.0, 100.0, nconds)
        salind = np.argmin(np.abs(allsal - salience))
        salindx2 = np.argmin(np.abs(allsal - (2 * salience)))
        if salience > 50.0:
            print("WARNING: FIXING DOUBLE SALIENCE AT 100%")
            salindx2 = nconds - 1
        P["afreq"] = np.repeat(afreqs[salind], nconds)
        P["aitv"] = np.repeat(aitvs[salind], nconds)
        P["afreq_2x"] = np.repeat(afreqs[salindx2], nconds)
        P["aitv_2x"] = np.repeat(aitvs[salindx2], nconds)
        P["bfreq"] = np.repeat(bfreqs[salind], nconds)
        P["bitv"] = np.repeat(bitvs[salind], nconds)
        P["bfreq_2x"] = np.repeat(bfreqs[salindx2], nconds)
        P["bitv_2x"] = np.repeat(bitvs[salindx2], nconds)
        P["nsyna"] = np.repeat(10, nconds)
        P["nsynb"] = np.repeat(10, nconds)
        P["hasffi"] = np.repeat(True, nconds)
        P["inhw"] = np.repeat(0.00075, nconds)
        P["inhtau"] = np.repeat(120.0, nconds)
        P["inhdelay"] = np.repeat(5.0, nconds)
        P["reallatency"] = np.repeat(reallatency, nconds)
        #
        P["adur"] = np.repeat(adur, nconds)
        P["bdur"] = np.repeat(bdur, nconds)
        if a_varies:
            P["astart"] = np.linspace(astart[0], astart[1], nconds)
        else:
            P["astart"] = np.repeat(astart, nconds)
        if b_varies:
            P["bstart"] = np.linspace(bstart[0], bstart[1], nconds)
        else:
            P["bstart"] = np.repeat(bstart, nconds)
        return P


if __name__ == "__main__":
    #Parse command-line arguments
    #weload ncores nconds nreps salience
    inputargs = sys.argv[1:]
    myargs = [1, 4, 7, 3, 37.5]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    ncores = int(myargs[1])
    nconds = int(myargs[2])
    nreps = int(myargs[3])
    salience = float(myargs[4])
    #------------------
    #Run the show
    if os.path.isfile("./data/SCNv2_Figure3A.npy") and weload:
        print("Loading Data!")
        outputA = np.load("./data/SCNv2_Figure3A.npy", allow_pickle=True)
        PA = np.load("./data/SCNv2_Figure3A_P.npy", allow_pickle=True)
        PA = PA.tolist()
    else:
        PA = getparams1d(
            ncores = ncores,
            nconds = nconds,
            nreps = nreps,
            salience = salience,
        )
        # make go!
        outputA = []
        outputA.append(myMPhandler(PA))
        outputA = np.array(outputA, dtype=object)
        np.save("./data/SCNv2_Figure3A.npy", outputA, allow_pickle=True)
        np.save("./data/SCNv2_Figure3A_P.npy", PA, allow_pickle=True)
    #
    print("done")
    plotres(outputA, PA)
    plt.show()
    #fhandle = plotres(outputA, outputB, outputC, PA, PB, PC)
    #pp = PdfPages("./figures/SCNv2_Figure2_new.pdf")
    #pp.savefig()
    #pp.close()


















