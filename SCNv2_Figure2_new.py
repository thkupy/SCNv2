#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This creates the new Figure 2... non-deterministic inputs
with realistic latencies. 

The command line args are:
    SCNv2_Figure2_new.py weload ncores nconds

Created: 2021-05-27
Revised: 2021-06-21 (checking)
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


def plotres(outputA, outputB, outputC, PA, PB, PC):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats
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
    fhandle = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    a_m = outputA[0][:, 0]
    a_s = outputA[0][:, 1]
    a_t = outputA[0][:, 2]
    b_m = outputB[0][:, 0]
    b_s = outputB[0][:, 1]
    b_t = outputB[0][:, 2]
    c_m = outputC[0][:, 0]
    c_s = outputC[0][:, 1]
    c_t = outputC[0][:, 2]
    #
    dotcol = ((.1, .1, .1), (.5, .5, .5))
    msize = 1
    dottype = ","
    #
    sp1 = plt.subplot(5,3,(1,4))
    plt.title("Apical only")
    plt.xlim((0,PA["dur"][0]))
    plt.plot((PA["astart"][0], PA["astart"][0] + PA["adur"][0]), (-1, -1), "g-")
    a_ttfsp = np.zeros((PA["N"],2))
    for icond in range(PA["N"]):
        plt.plot([0, 500], [icond, icond], "k--", linewidth=0.5)
        ttfsp = np.ones(PA["nreps"][0])
        for irep in range(PA["nreps"][0]):
            ypos = icond + (irep / (PA["nreps"][0]))
            xvals = np.array(a_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color=dotcol[icond%2], marker=dottype, markersize=msize, linestyle=" ")
            if xvals.size > 0:
                ttfsp[irep] = np.min(xvals)#no spontaneous activity!
            else:
                ttfsp[irep] = np.nan
        a_ttfsp[icond,0]=np.nanmean(ttfsp)
        a_ttfsp[icond,1]=np.nanstd(ttfsp)
    sp1.set_yticks(np.arange(PA["N"])+0.5)
    sp1.set_yticklabels(np.round(PA["afreq"],1))
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Input Frequency (Hz)")
    #
    sp2 = plt.subplot(5,3,(2,5))
    plt.title("Basal only")
    plt.xlim((0,PB["dur"][0]))
    plt.plot((PB["bstart"][0], PB["bstart"][0] + PB["bdur"][0]), (-1, -1), "g-")
    b_ttfsp = np.zeros((PB["N"],2))
    for icond in range(PB["N"]):
        ttfsp = np.ones(PB["nreps"][0])
        plt.plot([0, 500], [icond,icond], "k--", linewidth=0.5)
        for irep in range(PB["nreps"][0]):
            ypos = icond + (irep / (PB["nreps"][0]))
            xvals = np.array(b_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color=dotcol[icond%2], marker=dottype, markersize=msize, linestyle=" ")
            if xvals.size > 0:
                ttfsp[irep] = np.min(xvals)#no spontaneous activity!
            else:
                ttfsp[irep] = np.nan
        b_ttfsp[icond,0] = np.nanmean(ttfsp)
        b_ttfsp[icond,1] = np.nanstd(ttfsp)
    sp2.set_yticks(np.arange(PB["N"])+0.5)
    sp2.set_yticklabels(np.round(PB["bfreq"]))
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Input Frequency (Hz)")
    #
    sp3 = plt.subplot(5,3,(3,6))
    plt.title("Apical+Basal")
    plt.xlim((0,PC["dur"][0]))
    plt.plot((PC["bstart"][0], PC["bstart"][0] + PC["bdur"][0]), (-1, -1), "g-")
    c_ttfsp = np.zeros((PC["N"],2))
    for icond in range(PC["N"]):
        ttfsp = np.ones(PC["nreps"][0])
        plt.plot([0, 500], [icond,icond], "k--", linewidth=0.5)
        for irep in range(PC["nreps"][0]):
            ypos = icond + (irep / (PC["nreps"][0]))
            xvals = np.array(c_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color=dotcol[icond%2], marker=dottype, markersize=msize, linestyle=" ")
            if xvals.size > 0:
                ttfsp[irep] = np.min(xvals)#no spontaneous activity!
            else:
                ttfsp[irep] = np.nan
        c_ttfsp[icond,0]=np.nanmean(ttfsp)
        c_ttfsp[icond,1]=np.nanstd(ttfsp)
    sp3.set_yticks(np.arange(PC["N"])+0.5)
    sp3.set_yticklabels(np.round(np.linspace(0,100,PC["N"])))
    plt.xlabel("Time (ms)")
    plt.ylabel(r"Input Salience $\%$")
    #
    sp7 = plt.subplot(5,3,7)
    sp7.set_xscale("log")
    sp7.set_xticks((40, 50, 60, 70))
    sp7.set_xticklabels((40, 50, 60, 70))
    plt.errorbar(
        PA["afreq"],
        a_m,
        yerr=a_s, 
        #yerr=a_s / np.sqrt(PA["nreps"][0]-1),
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
    )
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("Mean Output (APs)")
    #
    sp8 = plt.subplot(5,3,8, sharey=sp7)
    sp8.set_xscale("log")
    sp8.set_xticks((100, 200, 300, 400))
    sp8.set_xticklabels((100, 200, 300, 400))
    plt.errorbar(
        PB["bfreq"],
        b_m,
        yerr=b_s, 
        #yerr=b_s / np.sqrt(PB["nreps"][0]-1),
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
    )
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("Mean Output (APs)")

    #
    sp9 = plt.subplot(5,3,9, sharey=sp7)
    plt.plot(np.linspace(0,100,PC["N"]), a_m + b_m, "g-")
    enh = np.vstack((a_m, b_m))
    plt.plot(np.linspace(0,100,PC["N"]), np.max(enh, 0), "r-")
    plt.errorbar(
        np.linspace(0,100,PC["N"]), 
        c_m, 
        yerr=c_s,
        #yerr=c_s / np.sqrt(PC["nreps"][0]-1)),
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
    )
    plt.xlabel(r"Input Salience $\%$")
    plt.ylabel("Mean Output (APs)")
    #
    sp10 = plt.subplot(5,3,10)
    plt.errorbar(
        PA["afreq"],
        a_ttfsp[:,0], 
        yerr=a_ttfsp[:,1],
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
        )
    sp10.set_xscale("log")
    sp10.set_xticks((40, 50, 60, 70))
    sp10.set_xticklabels((40, 50, 60, 70))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("Mean FSL (ms)")
    #
    sp11 = plt.subplot(5,3,11, sharey=sp10)
    plt.errorbar(
        PB["bfreq"],
        b_ttfsp[:,0], 
        yerr=b_ttfsp[:,1],
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
        )
    sp11.set_xscale("log")
    sp11.set_xticks((100, 200, 300, 400))
    sp11.set_xticklabels((100, 200, 300, 400))
    plt.xlabel("Mean Input Frequency (Hz)")
    plt.ylabel("Mean FSL (ms)")
    #
    sp12 = plt.subplot(5,3,12, sharey=sp10)
    X = np.linspace(0,100,PC["N"])
    plt.errorbar(
        X,
        c_ttfsp[:,0], 
        yerr=c_ttfsp[:,1],
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
        )
    plt.plot(X,b_ttfsp[:,0],"b-")
    plt.plot(X,a_ttfsp[:,0], "r-")
    plt.xlabel(r"Input Salience $\%$")
    plt.ylabel("Mean FSL (ms)")
    #
    sp14 = plt.subplot(5, 3, 14)
    X = np.linspace(0,100,PC["N"])
    Y = c_m - (a_m + b_m)
    plt.errorbar(
        X,
        Y, 
        yerr=c_s,
        #yerr=c_s / np.sqrt(PC["nreps"][0]-1)),
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
    )
    plt.plot(X[np.argmax(Y)], np.max(Y), "ro", markersize=10, markerfacecolor=None)
    plt.plot((X[np.argmax(Y)], X[np.argmax(Y)]), (0, np.max(Y)), "r-")
    plt.text(X[np.argmax(Y)]+1, 1, str(X[np.argmax(Y)]))
    plt.xlabel(r"Input Salience $\%$")
    plt.ylabel("AP Increase (AP) \n vs. sum")
    #
    #
    sp15 = plt.subplot(5, 3, 15)
    YA = a_m + b_m
    YM = c_m
    xx = []
    yy = []
    yys = []
    for thisx in range(int(np.floor(X.size/2))):
        xx.append(X[thisx])
        X2 = np.argwhere(X==X[thisx]*2)
        yy.append(YM[thisx]-YA[X2])
        yys.append(c_s[thisx])
    yy = np.array(yy).flatten()
    plt.errorbar(
        xx,
        yy, 
        yerr=yys,
        #yerr=c_s / np.sqrt(PC["nreps"][0]-1)),
        marker="o",
        markersize=4,
        markerfacecolor="w",
        color="k",
    )
    plt.plot(xx[np.argmax(yy)], np.max(yy), "ro", markersize=10, markerfacecolor=None)
    plt.plot((xx[np.argmax(yy)], xx[np.argmax(yy)]), (0, np.max(yy)), "r-")
    plt.xlabel(r"Input Salience $\%$")
    plt.ylabel("AP Increase (AP) \n vs. sum@2xsalience")
    plt.text(xx[np.argmax(yy)]+1, 1, str(xx[np.argmax(yy)]))
    plt.tight_layout()
    #
    #
    #adding PSTH as per SW request
    exI = (3, 6, 9)
    fhandle2 = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    mybins = np.linspace(0, PA["dur"][0], int(PA["dur"][0] / 10))
    nrep = PA["nreps"][0]
    #
    sp1 = plt.subplot(5,3,1)
    plt.title("Apical only")
    vals,bins = np.histogram(np.concatenate(a_t[exI[0]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.0, 0.0, 0.5))
    vals,bins = np.histogram(np.concatenate(a_t[exI[1]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.0, 0.0, 0.7))
    vals,bins = np.histogram(np.concatenate(a_t[exI[2]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.0, 0.0, 1.0))
    #
    sp2 = plt.subplot(5,3,2, sharex=sp1, sharey=sp1)
    plt.title("Basal only")
    vals,bins = np.histogram(np.concatenate(b_t[exI[0]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.5, 0.0, 0.0))
    vals,bins = np.histogram(np.concatenate(b_t[exI[1]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.7, 0.0, 0.0))
    vals,bins = np.histogram(np.concatenate(b_t[exI[2]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(1.0, 0.0, 0.0))
    
    sp3 = plt.subplot(5,3,3, sharex=sp1, sharey=sp1)
    plt.title("Apical+Basal only")
    vals,bins = np.histogram(np.concatenate(c_t[exI[0]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.3, 0.3, 0.3))
    vals,bins = np.histogram(np.concatenate(c_t[exI[1]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.5, 0.5, 0.5))
    vals,bins = np.histogram(np.concatenate(c_t[exI[2]]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.7, 0.7, 0.7))
    #
    sp1.set_ylabel("Mean AP/bin")
    sp1.set_xlabel("Time (binned, ms)")
    sp2.set_xlabel("Time (binned, ms)")
    sp3.set_xlabel("Time (binned, ms)")
    #
    return (fhandle, fhandle2)

def getparams(
            ncores = 4,
            nconds = 7,
            nreps = 3,
            aon=True,
            astart=0.0, 
            adur=125.0, 
            afreqs=(35.0,90.0), ###40/120#35/70 changed 2021-06-21
            bon=False, 
            bstart=0.0, 
            bdur=125.0, 
            bfreqs=(75.0,333.0),#60/250#75/400 changed 2021-06-21
            reallatency=True,
        ):
        #Some fixed Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.01#0.025
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
        P["nsyna"] = np.repeat(25, nconds)#10 changed 2021-06-22
        P["astart"] = np.repeat(astart, nconds)
        P["adur"] = np.repeat(adur, nconds)
        P["afreq"] = afreq
        P["aitv"] = aitv
        P["nsynb"] = np.repeat(25, nconds)#10 changed 2021-06-22
        P["bstart"] = np.repeat(bstart, nconds)
        P["bdur"] = np.repeat(bdur, nconds)
        P["bfreq"] = bfreq
        P["bitv"] = bitv
        #
        P["hasffi"] = np.repeat(True, nconds)
        P["inhw"] = np.repeat(0.001, nconds)##0.00075 changed 2021-06-21
        P["inhtau"] = np.repeat(75.0, nconds)##120.0 changed 2021-06-21
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
        outputC = np.load("./data/SCNv2_SaliencetestsC.npy", allow_pickle=True)
        PA = np.load("./data/SCNv2_Saliencetests_PA.npy", allow_pickle=True)
        PA = PA.tolist()
        PB = np.load("./data/SCNv2_Saliencetests_PB.npy", allow_pickle=True)
        PB = PB.tolist()
        PC = np.load("./data/SCNv2_Saliencetests_PC.npy", allow_pickle=True)
        PC = PC.tolist()
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
        PC = getparams(
            ncores = ncores,
            nconds = nconds,
            nreps = nreps,
            aon=True,
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
        outputC = []
        outputC.append(myMPhandler(PC))
        outputC = np.array(outputC, dtype=object)
        np.save("./data/SCNv2_SaliencetestsC.npy", outputC, allow_pickle=True)
        np.save("./data/SCNv2_Saliencetests_PC.npy", PC, allow_pickle=True)
    #
    print("done")
    #plotres(outputA, outputB, outputC, PA, PB, PC)
    #plt.show()
    fhandle, fhandle2 = plotres(outputA, outputB, outputC, PA, PB, PC)
    pp = PdfPages("./figures/SCNv2_Figure2_new.pdf")
    pp.savefig(fhandle)
    pp.savefig(fhandle2)
    pp.close()


















