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
Revised: 2021-06-23 (completion)
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
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rc("text", usetex=False)
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("axes", labelsize="small")
def runonecondition(x, P):
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    xdel = P["dist"][x] * (1000.0 / 343.0)#1/343 extra s for every meter (in ms)
    fslab = np.zeros(P["nreps"][x])*np.nan
    spcab = np.zeros(P["nreps"][x])
    fsla = np.zeros(P["nreps"][x])*np.nan
    spca = np.zeros(P["nreps"][x])
    fslb = np.zeros(P["nreps"][x])*np.nan
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
        if spcab[irep] > 0:
            fslab[irep] = np.min(SAB["PeakT"])
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
        if spca[irep] > 0:
            fsla[irep] = np.min(SA["PeakT"])
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
        if spcb[irep] > 0:
            fslb[irep] = np.min(SB["PeakT"])
        #
    print("x: " + str(x))
    return [
        np.mean(spca),
        np.std(spca),
        np.mean(spcb),
        np.std(spcb),
        np.mean(spcab),
        np.std(spcab),
        np.nanmean(fsla),
        np.nanstd(fsla),
        np.nanmean(fslb),
        np.nanstd(fslb),
        np.nanmean(fslab),
        np.nanstd(fslab),
        P["dist"][x],
        P["afreq"][x],
        P["bfreq"][x],
        spca,
        spcb,
        spcab,
        fsla,
        fslb,
        fslab,
    ]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonecondition, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def mymultitest(a,b,c,d, thr):
    from scipy import stats
    s1,p1 = stats.ttest_ind(a,b)
    b1 = p1<(thr/6)
    T = np.array((1, 2, s1, p1, b1))
    s2,p2 = stats.ttest_ind(a,c)
    b2 = p2<(thr/6)
    T = np.vstack((T, np.array((1, 3, s2, p2, b2))))
    s3,p3 = stats.ttest_ind(a,d)
    b3 = p3<(thr/6)
    T = np.vstack((T, np.array((1, 4, s3, p3, b3))))
    s4,p4 = stats.ttest_ind(b,c)
    b4 = p4<(thr/6)
    T = np.vstack((T, np.array((2, 3, s4, p4, b4))))
    s5,p5 = stats.ttest_ind(b,d)
    b5 = p5<(thr/6)
    T = np.vstack((T, np.array((2, 4, s5, p5, b5))))
    s6,p6 = stats.ttest_ind(c,d)
    b6 = p6<(thr/6)
    T = np.vstack((T, np.array((3, 4, s6, p6, b6))))
    return(T)


def mymultitest2(a, b1, b2, b3, b4, thr):
    from scipy import stats
    s1,p1 = stats.ttest_ind(a,b1)
    b1 = p1<(thr/4)
    T = np.array((1, 2, s1, p1, b1))
    s2,p2 = stats.ttest_ind(a,b2)
    b2 = p2<(thr/4)
    T = np.vstack((T, np.array((1, 3, s2, p2, b2))))
    s3,p3 = stats.ttest_ind(a,b3)
    b3 = p3<(thr/4)
    T = np.vstack((T, np.array((1, 4, s3, p3, b3))))
    s4,p4 = stats.ttest_ind(a,b4)
    b4 = p4<(thr/4)
    T = np.vstack((T, np.array((2, 3, s4, p4, b4))))
    return(T)
    
    

    
    
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
    fheight = 8  # cm
    fwidth = 8
    #contourlims = (1.0,333.0)
    #ncontours = 27
    a_m = np.array(outputA[0][:, 0], dtype=float)
    a_s = np.array(outputA[0][:, 1], dtype=float)
    b_m = np.array(outputA[0][:, 2], dtype=float)
    b_s = np.array(outputA[0][:, 3], dtype=float)
    ab_m = np.array(outputA[0][:, 4], dtype=float)
    ab_s = np.array(outputA[0][:, 5], dtype=float)
    fa_m = np.array(outputA[0][:, 6], dtype=float)
    fa_s = np.array(outputA[0][:, 7], dtype=float)
    fb_m = np.array(outputA[0][:, 8], dtype=float)
    fb_s = np.array(outputA[0][:, 9], dtype=float)
    fab_m = np.array(outputA[0][:, 10], dtype=float)
    fab_s = np.array(outputA[0][:, 11], dtype=float)
    dist = np.array(outputA[0][:, 12], dtype=float)
    afreq = np.array(outputA[0][:, 13], dtype=float)
    bfreq = np.array(outputA[0][:, 14], dtype=float)
    alldist = np.unique(dist)
    ndists = np.unique(dist).size
    nfreqs = 4
    #
    spa = np.vstack(outputA[0][:,15])
    spb = np.vstack(outputA[0][:,16])
    spab = np.vstack(outputA[0][:,17])
    fsla = np.vstack(outputA[0][:,18])
    fslb = np.vstack(outputA[0][:,19])
    fslab = np.vstack(outputA[0][:,20])
    #
    filtsig = 1.33
    a_m = ndimage.gaussian_filter1d(np.reshape(a_m, (ndists, nfreqs)), sigma=filtsig, axis=0)
    a_s = ndimage.gaussian_filter1d(np.reshape(a_s, (ndists, nfreqs)), sigma=filtsig, axis=0) 
    b_m = ndimage.gaussian_filter1d(np.reshape(b_m, (ndists, nfreqs)), sigma=filtsig, axis=0)
    b_s = ndimage.gaussian_filter1d(np.reshape(b_s, (ndists, nfreqs)), sigma=filtsig, axis=0) 
    ab_m = ndimage.gaussian_filter1d(np.reshape(ab_m, (ndists, nfreqs)), sigma=filtsig, axis=0)
    ab_s = ndimage.gaussian_filter1d(np.reshape(ab_s, (ndists, nfreqs)), sigma=filtsig, axis=0)
    fa_m = ndimage.gaussian_filter1d(np.reshape(fa_m, (ndists, nfreqs)), sigma=filtsig, axis=0)
    fa_s = ndimage.gaussian_filter1d(np.reshape(fa_s, (ndists, nfreqs)), sigma=filtsig, axis=0)
    fb_m = ndimage.gaussian_filter1d(np.reshape(fb_m, (ndists, nfreqs)), sigma=filtsig, axis=0)
    fb_s = ndimage.gaussian_filter1d(np.reshape(fb_s, (ndists, nfreqs)), sigma=filtsig, axis=0)
    fab_m = ndimage.gaussian_filter1d(np.reshape(fab_m, (ndists, nfreqs)), sigma=filtsig, axis=0)
    fab_s = ndimage.gaussian_filter1d(np.reshape(fab_s, (ndists, nfreqs)), sigma=filtsig, axis=0)
    dist = np.reshape(dist, (ndists, nfreqs)) 
    afreq = np.reshape(afreq, (ndists, nfreqs)) 
    bfreq = np.reshape(bfreq, (ndists, nfreqs)) 
    #
    filtsig = 0.66
    #a_am = ndimage.gaussian_filter1d(a_am, sigma=filtsig, order=0)
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
    plt.subplot(2,2,1)
    plt.errorbar(alldist, a_m[:,0], yerr=a_s[:,0], label="apical weak", color=(.0, .0, .5))
    plt.errorbar(alldist, a_m[:,2], yerr=a_s[:,2], label="apical strong", color=(.0, .0, 1.0))
    plt.errorbar(alldist, b_m[:,0], yerr=a_s[:,0], label="basal weak", color=(.5, .0, .0))
    plt.errorbar(alldist, b_m[:,1], yerr=a_s[:,2], label="basal strong", color=(1.0, .0, .0))
    plt.errorbar(alldist, ab_m[:,0], yerr=ab_s[:,0], label="weak/weak", color="b")
    plt.errorbar(alldist, ab_m[:,1], yerr=ab_s[:,1], label="weak/strong", color="y")
    plt.errorbar(alldist, ab_m[:,2], yerr=ab_s[:,2], label="strong/weak", color="g")
    plt.errorbar(alldist, ab_m[:,3], yerr=ab_s[:,3], label="strong/strong", color="r")
    plt.xlabel("Distance to sound/light source (m)")
    plt.ylabel("Unimodal response (AP)")    
    plt.legend()
    #
    plt.subplot(2,2,2)
    plt.errorbar(alldist, ab_m[:,0] - (a_m[:,0]+b_m[:,0]), yerr=ab_s[:,0], label="weak/weak")
    plt.errorbar(alldist, ab_m[:,1] - (a_m[:,1]+b_m[:,1]), yerr=ab_s[:,1], label="weak/strong")
    plt.errorbar(alldist, ab_m[:,2] - (a_m[:,2]+b_m[:,2]), yerr=ab_s[:,2], label="strong/weak")
    plt.errorbar(alldist, ab_m[:,3] - (a_m[:,3]+b_m[:,3]), yerr=ab_s[:,3], label="strong/strong")
    plt.legend()
    plt.xlabel("Distance to sound/light source (m)")
    plt.ylabel("Enhancement vs. sum of unimodal (AP diff)")
    #
    #Some Statistics... we are going to compare for each distance the 4 conditions
    pthres = 0.01
    for idist in range(ndists):
        weak_weak = spab[(idist * 4)]-(spa[(idist * 4)] + spb[(idist * 4)])
        weak_strong = spab[(idist * 4)+1]-(spa[(idist * 4)+1] + spb[(idist * 4)+1])
        strong_weak = spab[(idist * 4)+2]-(spa[(idist * 4)+2] + spb[(idist * 4)+2])
        strong_strong = spab[(idist * 4)+3]-(spa[(idist * 4)+3] + spb[(idist * 4)+3])
        AN=stats.f_oneway(weak_weak, weak_strong, strong_weak, strong_strong)
        print("Dist: " + str(alldist[idist]) + ":")
        print(AN)
        if AN.pvalue < pthres:
            plt.plot(alldist[idist], 60, "k*", markersize=2)
            T = mymultitest(weak_weak, weak_strong, strong_weak, strong_strong, pthres)
            if T[2,4]:
                plt.plot(alldist[idist], ab_m[idist,0] - (a_m[idist,0]+b_m[idist,0]), "bo", markersize=2)
            if T[4,4]:
                plt.plot(alldist[idist], ab_m[idist,1] - (a_m[idist,1]+b_m[idist,1]), "yo", markersize=2)
            if T[5,4]:
                plt.plot(alldist[idist], ab_m[idist,2] - (a_m[idist,2]+b_m[idist,2]), "go", markersize=2)
            #3/5/6 is vs. strong_strong
    #
    plt.subplot(2,2,3)
    plt.errorbar(alldist, fa_m[:,0], yerr=fa_s[:,0], label="apical weak", color=(.0, .0, .5))
    plt.errorbar(alldist, fa_m[:,2], yerr=fa_s[:,2], label="apical strong", color=(.0, .0, 1.0))
    plt.errorbar(alldist, fb_m[:,0], yerr=fa_s[:,0], label="basal weak", color=(.5, .0, .0))
    plt.errorbar(alldist, fb_m[:,1], yerr=fa_s[:,2], label="basal strong", color=(1.0, .0, .0))
    plt.xlabel("Distance to sound/light source (m)")
    plt.ylabel("Unimodal FSL (ms)")    
    plt.legend()
    #
    plt.subplot(2,2,4)
    plt.errorbar(alldist, fab_m[:,0] - fa_m[:,0], yerr=fab_s[:,0], label="weak/weak")
    plt.errorbar(alldist, fab_m[:,1] - fa_m[:,1], yerr=fab_s[:,1], label="weak/strong")
    plt.errorbar(alldist, fab_m[:,2] - fa_m[:,2], yerr=fab_s[:,2], label="strong/weak")
    plt.errorbar(alldist, fab_m[:,3] - fa_m[:,3], yerr=fab_s[:,3], label="strong/strong")   
    plt.legend()
    plt.xlabel("Distance to sound/light source (m)")
    plt.ylabel("FSL vs. visual only (ms diff)")
    print("weak/weak:")
    print(fab_m[:,0] - fa_m[:,0])
    print(fab_s[:,0])
    print("----")
    print("weak/strong:")
    print(fa_m[:,1])
    print(fab_m[:,1])
    print(fab_m[:,1] - fa_m[:,1])
    print(fab_s[:,1])
    print("----")
    print("strong/weak:")
    print(fab_m[:,2] - fa_m[:,2])
    print(fab_s[:,2])
    print("----")
    print("strong/strong:")
    print(fab_m[:,3] - fa_m[:,3])
    print(fab_s[:,3])
    print("----")
    #Some Statistics... we are going to compare for each distance the 4 conditions
    pthres = 0.01
    for idist in range(ndists):
        apical_strong = fsla[(idist * 4)+3][np.isfinite(fsla[(idist * 4)+3])]
        weak_weak = fslab[(idist * 4)][np.isfinite(fslab[(idist * 4)])]
        weak_strong = fslab[(idist * 4)+1][np.isfinite(fslab[(idist * 4)+1])]
        strong_weak = fslab[(idist * 4)+2][np.isfinite(fslab[(idist * 4)+2])]
        strong_strong = fslab[(idist * 4)+3][np.isfinite(fslab[(idist * 4)+3])]
        AN = stats.f_oneway(apical_strong, weak_weak, weak_strong, strong_weak, strong_strong)
        print("Dist: " + str(alldist[idist]) + ":")
        print(AN)
        if AN.pvalue < pthres:
            plt.plot(alldist[idist], 80, "k*", markersize=2)
            T = mymultitest2(apical_strong, weak_weak, weak_strong, strong_weak, strong_strong, pthres)
            if T[0,4]:
                plt.plot(alldist[idist], fab_m[idist,0] - fa_m[idist,0],  "bo", markersize=2)
            if T[1,4]:
                plt.plot(alldist[idist], fab_m[idist,1] - fa_m[idist,1], "yo", markersize=2)
            if T[2,4]:
                plt.plot(alldist[idist], fab_m[idist,2] - fa_m[idist,2], "go", markersize=2)
            if T[3,4]:
                plt.plot(alldist[idist], fab_m[idist,3] - fa_m[idist,3], "ro", markersize=2)
            #3/5/6 is vs. strong_strong



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
        dt = 0.025
        dur = 1000.0#500.0
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
        P["Seed"] = 77253#3786562
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
        # 2021-11-17: the reviewer2 of the J Neurosci submission raised the issue
        #  about the difference between these results (weak/weak) here and those 
        #  in Figure 8e (-->SCNv3_Figure3_new.py)... the latter were calculated
        #  "at 20% input power" (->21.4% for 15 conds), which corresponds to a
        #  afreq of 41 and a bfreq of 103, which explains the differences.
        #  A revised version of this figure will be recalculated with these new
        #  weak/weak values.
        afreq = np.array([41.0, 41.0, 60.0, 60.0])#45/60#45/80#45/85
        bfreq = np.array([103.0, 300.0, 103.0, 300.0])#150/300#120/300
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
    fhandle1.savefig("./figures/SCNv2_Figure5_new_p1.svg")

















