#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This creates the new Figure 4...
non-deterministic inputs with realistic latencies. 
Here, we vary either the pNeurite-Length or the NMDA contribution (i.e. the
tau-decay of the apical inputs) gradually and check simple (>sum of unimodal inputs
at the same salience) or "real" (>sum of unimodal inputs as 2x salience) enhancement.

The command line args are:
    SCNv2_Figure4_new.py weload ncores nconds

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
    sa = []
    spca = np.zeros(P["nreps"][x])
    sb = []
    spcb = np.zeros(P["nreps"][x])
    sab = []
    spcab = np.zeros(P["nreps"][x])
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
            apicaltau=P["apicaltau"][x],
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=P["hasffi"][x],
            inhw=P["inhw"][x],
            inhtau=P["inhtau"][x],
            inhdelay=P["inhdelay"][x],
            noiseval=P["noiseval"][x],
            reallatency=P["reallatency"][x],
            pNeurit_L=P["pNeuritL"][x],
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
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x], P["bitv"][x]),
            inputstop=(P["astart"][x] + P["adur"][x], P["bstart"][x] + P["bdur"][x]),
            hasnmda=True,
            apicaltau=P["apicaltau"][x],
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=P["hasffi"][x],
            inhw=P["inhw"][x],
            inhtau=P["inhtau"][x],
            inhdelay=P["inhdelay"][x],
            noiseval=P["noiseval"][x],
            reallatency=P["reallatency"][x],
            pNeurit_L=P["pNeuritL"][x],
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
            pinputactivity=(P["astart"][x], P["aitv"][x], P["bstart"][x], P["bitv"][x]),
            inputstop=(P["astart"][x] + P["adur"][x], P["bstart"][x] + P["bdur"][x]),
            hasnmda=True,
            apicaltau=P["apicaltau"][x],
            seed=thisseed+irep,
            hasfbi=False,
            hasffi=P["hasffi"][x],
            inhw=P["inhw"][x],
            inhtau=P["inhtau"][x],
            inhdelay=P["inhdelay"][x],
            noiseval=P["noiseval"][x],
            reallatency=P["reallatency"][x],
            pNeurit_L=P["pNeuritL"][x],
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
        #
        #
    print("x: " + str(x))
    return [
        np.mean(spca),
        np.std(spca),
        sa,
        np.mean(spcb),
        np.std(spcb),
        sb,
        np.mean(spcab),
        np.std(spcab),
        sab,
        P["afreq"][x],
        P["pNeuritL"][x],
        P["apicaltau"][x],
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
    a_am = np.array(outputA[0][:, 0], dtype=float)
    a_as = np.array(outputA[0][:, 1], dtype=float)
    a_bm = np.array(outputA[0][:, 3], dtype=float)
    a_bs = np.array(outputA[0][:, 4], dtype=float)
    a_abm = np.array(outputA[0][:, 6], dtype=float)
    a_abs = np.array(outputA[0][:, 7], dtype=float)
    #a_t = outputA[0][:, 2]
    b_am = np.array(outputB[0][:, 0], dtype=float)
    b_as = np.array(outputB[0][:, 1], dtype=float)
    b_bm = np.array(outputB[0][:, 3], dtype=float)
    b_bs = np.array(outputB[0][:, 4], dtype=float)
    b_abm = np.array(outputB[0][:, 6], dtype=float)
    b_abs = np.array(outputB[0][:, 7], dtype=float)
    #b_t = outputB[0][:, 2]
    freq = np.array(outputA[0][:, 9], dtype=float)
    pnl = np.array(outputA[0][:, 10], dtype=float)
    tau = np.array(outputB[0][:, 11], dtype=float)
    #
    filtsig = 0.5#0.66
    mymode = "constant"
    a_am = ndimage.gaussian_filter(a_am, sigma=filtsig, order=0, mode=mymode)
    a_bm = ndimage.gaussian_filter(a_bm, sigma=filtsig, order=0, mode=mymode)
    a_abm = ndimage.gaussian_filter(a_abm, sigma=filtsig, order=0, mode=mymode)
    b_am = ndimage.gaussian_filter(b_am, sigma=filtsig, order=0, mode=mymode)
    b_bm = ndimage.gaussian_filter(b_bm, sigma=filtsig, order=0, mode=mymode)
    b_abm = ndimage.gaussian_filter(b_abm, sigma=filtsig, order=0, mode=mymode)
    #
    dotcol = ((.1, .1, .1), (.5, .5, .5))
    msize = 1
    dottype = ","
    #
    nconds = np.unique(pnl).size
    nfreqs = np.unique(freq).size
    X = np.linspace(0.0, 100.0, np.unique(freq).size)
    cola = plt.cm.Blues((np.linspace(0.3,1,nconds)))
    colb = plt.cm.Reds((np.linspace(0.3,1,nconds)))
    colab =  plt.cm.gray((np.linspace(0.0,0.7,nconds)))
    greens =  plt.cm.Greens((np.linspace(0.3,1,nconds)))
    #
    ###FIGURE VERSION 1 -line plots-
    fhandle1 = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    for ncond, thisL in enumerate(np.unique(pnl)):
        sp1 = plt.subplot(5,2,1)
        sp1.set_title("pNeurite Length")
        plt.errorbar(
            X, 
            a_am[pnl==thisL],
            yerr=a_as[pnl==thisL],
            color=cola[ncond],
            marker="o",
            markersize=4,
        )
        sp3 = plt.subplot(5,2,3, sharey=sp1)
        plt.errorbar(
            X, 
            a_bm[pnl==thisL],
            yerr=a_bs[pnl==thisL],
            color=colb[ncond],
            marker="o",
            markersize=4,
        )
        sp5 = plt.subplot(5,2,5)
        plt.errorbar(
            X, 
            a_abm[pnl==thisL],
            yerr=a_abs[pnl==thisL],
            color=colab[ncond],
            marker="o",
            markersize=4,
        )
        sp7 = plt.subplot(5,2,7, sharey=sp5)
        plt.plot(X,a_am[pnl==thisL] + a_bm[pnl==thisL],"-",color=greens[ncond])
        sp9 = plt.subplot(5,2,9)
        Y = a_abm[pnl==thisL] - (a_am[pnl==thisL] + a_bm[pnl==thisL])
        plt.errorbar(
            X,
            Y,
            yerr=a_abs[pnl==thisL],
            color=colab[ncond],
            marker="o",
            markersize=4,
        )
    for ncond, thist in enumerate(np.unique(tau)):
        sp2 = plt.subplot(5,2,2)
        sp2.set_title("NMDA component \n (apical tau-decay)")
        plt.errorbar(
            X, 
            b_am[tau==thist],
            yerr=b_as[tau==thist],
            color=cola[ncond],
            marker="o",
            markersize=4,
        )
        sp4 = plt.subplot(5,2,4)
        plt.errorbar(
            X, 
            b_bm[tau==thist],
            yerr=b_bs[tau==thist],
            color=colb[ncond],
            marker="o",
            markersize=4,
        )
        sp6 = plt.subplot(5,2,6)
        plt.errorbar(
            X, 
            b_abm[tau==thist],
            yerr=b_abs[tau==thist],
            color=colab[ncond],
            marker="o",
            markersize=4,
        )
        sp8 = plt.subplot(5,2,8)
        plt.plot(X,b_am[tau==thist] + b_bm[tau==thist],"-",color=greens[ncond])
        sp10 = plt.subplot(5,2,10)
        plt.errorbar(
            X, 
            b_abm[tau==thist] - (b_am[tau==thist] + b_bm[tau==thist]),
            yerr=b_abs[tau==thist],
            color=colab[ncond],
            marker="o",
            markersize=4,
        )
    sp1.set_ylabel("Apical only response (AP)")
    sp3.set_ylabel("Basal only response (AP)")
    sp5.set_ylabel("Multimodal response (AP)")
    sp7.set_ylabel("Sum of unimodal (AP)")
    sp9.set_ylabel("Multimodal benefit (AP)")
    sp9.set_ylabel("Input Power (\%)")
    sp10.set_ylabel("Input Power (\%)")
    #
    plt.tight_layout()
    #
    ####FIGURE VERSION 2 -colormesh version-
    fhandle2 = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54))#, dpi=600)
    Y1 = np.unique(pnl)
    Y2 = np.unique(tau)
    #
    mycontours= np.arange(11) * 10
    sp21 = plt.subplot(5,2,1)
    sp21.set_title("pNeurite Length")
    z21 = np.reshape(a_am, (nconds,nfreqs))
    c21 = sp21.pcolormesh(X,Y1, z21, cmap="Blues", shading="gouraud", vmin=0, vmax=80)
    plt.colorbar(c21, ax=sp21, extend="both")
    sp21.set_ylabel("Apical only \n pNeuritL (µm)")
    cl21 = sp21.contour(X,Y1,z21,mycontours, colors="k", linewidths=0.5)
    sp21.clabel(cl21, cl21.levels, fmt="%1.0f", fontsize=6)
    #
    sp23 = plt.subplot(5,2,3)
    z23 = np.reshape(a_bm, (nconds,nfreqs))
    c23 = sp23.pcolormesh(X,Y1, z23, cmap="Reds", shading="gouraud", vmin=0, vmax=80)
    plt.colorbar(c23, ax=sp23, extend="both")
    sp23.set_ylabel("Basal only \n pNeuritL (µm)")
    cl23 = sp23.contour(X,Y1,z23,mycontours, colors="k", linewidths=0.5)
    sp23.clabel(cl23, cl23.levels, fmt="%1.0f", fontsize=6)
    #
    sp25 = plt.subplot(5,2,5)
    z25 = np.reshape(a_abm, (nconds,nfreqs))
    c25 = sp25.pcolormesh(X,Y1, z25, cmap="bone", shading="gouraud", vmin=0, vmax=100)
    plt.colorbar(c25, ax=sp25, extend="both")
    sp25.set_ylabel("Multimodal \n pNeuritL (µm)")
    cl25 = sp25.contour(X,Y1,z25,mycontours, colors="k", linewidths=0.5)
    sp25.clabel(cl25, cl25.levels, fmt="%1.0f", fontsize=6)
    #
    sp27 = plt.subplot(5,2,7)
    z27 = np.reshape((a_am + a_bm), (nconds,nfreqs))
    c27 = sp27.pcolormesh(X,Y1, z27, cmap="Greens", shading="gouraud", vmin=0, vmax=100)
    plt.colorbar(c27, ax=sp27, extend="both")
    sp27.set_ylabel("Sum of unimodal \n NeuritL (µm)")
    cl27 = sp27.contour(X,Y1,z27,mycontours, colors="k", linewidths=0.5)
    sp27.clabel(cl27, cl27.levels, fmt="%1.0f", fontsize=6)
    #
    sp29 = plt.subplot(5,2,9)
    z29 = np.reshape(a_abm - (a_am + a_bm), (nconds,nfreqs))
    c29 = sp29.pcolormesh(X,Y1, z29, cmap="seismic", shading="gouraud", vmin=-70, vmax=70)
    plt.colorbar(c29, ax=sp29, extend="both")
    sp29.set_ylabel("Multimodal enhancement \n pNeuritL (µm)")
    sp29.set_xlabel("Input Power (\%)")
    cl29 = sp29.contour(X,Y1,z29,(-10,0,10,20,30,40,50,60,70), colors="k", linewidths=0.5)
    sp29.clabel(cl29, cl29.levels, fmt="%1.0f", fontsize=6)
    #
    sp22 = plt.subplot(5,2,2)
    sp22.set_title("NMDA component \n (apical tau-decay)")
    z22 = np.reshape(b_am, (nconds,nfreqs))
    c22 = sp22.pcolormesh(X, Y2, z22, cmap="Blues", shading="gouraud", vmin=0, vmax=80)
    plt.colorbar(c22, ax=sp22, extend="both")
    sp22.set_ylabel(" \n Apical tau-decay (ms)")
    cl22 = sp22.contour(X,Y2,z22,mycontours, colors="k", linewidths=0.5)
    sp22.clabel(cl22, cl22.levels, fmt="%1.0f", fontsize=6)
    #
    sp24 = plt.subplot(5,2,4)
    z24 = np.reshape(b_bm, (nconds,nfreqs))
    c24 = sp24.pcolormesh(X, Y2, z24, cmap="Reds", shading="gouraud", vmin=0, vmax=80)
    plt.colorbar(c24, ax=sp24, extend="both")
    sp24.set_ylabel(" \n Apical tau-decay (ms)")
    cl24 = sp24.contour(X,Y2,z24,mycontours, colors="k", linewidths=0.5)
    sp24.clabel(cl24, cl24.levels, fmt="%1.0f", fontsize=6)
    #
    sp26 = plt.subplot(5,2,6)
    z26 = np.reshape(b_abm, (nconds,nfreqs))
    c26 = sp26.pcolormesh(X, Y2, z26, cmap="bone", shading="gouraud", vmin=0, vmax=100)
    plt.colorbar(c26, ax=sp26, extend="both")
    sp26.set_ylabel(" \n Apical tau-decay (ms)")
    cl26 = sp26.contour(X,Y2,z26,mycontours, colors="k", linewidths=0.5)
    sp26.clabel(cl26, cl26.levels, fmt="%1.0f", fontsize=6)
    #
    sp28 = plt.subplot(5,2,8)
    z28 = np.reshape((b_am + b_bm), (nconds,nfreqs))
    c28 = sp28.pcolormesh(X, Y2, z28, cmap="Greens", shading="gouraud", vmin=0, vmax=100)
    plt.colorbar(c28, ax=sp28, extend="both")
    sp28.set_ylabel(" \n Apical tau-decay (ms)")
    cl28 = sp28.contour(X,Y2,z28,mycontours, colors="k", linewidths=0.5)
    sp28.clabel(cl28, cl28.levels, fmt="%1.0f", fontsize=6)
    #
    sp210 = plt.subplot(5,2,10)
    z210 = np.reshape(b_abm - (b_am + b_bm), (nconds,nfreqs))
    c210 = sp210.pcolormesh(X, Y2, z210, cmap="seismic", shading="gouraud", vmin=-70, vmax=70)
    plt.colorbar(c210, ax=sp210, extend="both")
    sp210.set_ylabel(" \n Apical tau-decay (ms)")
    sp210.set_xlabel("Input Power (\%)")
    cl210 = sp210.contour(X,Y2,z210,(-10, 0, 10,20,30,40,50,60,70), colors="k", linewidths=0.5)
    sp210.clabel(cl210, cl210.levels, fmt="%1.0f", fontsize=6)
    #
    plt.tight_layout()
    #
    return (fhandle1, fhandle2)


def getparams(
            ncores=4,
            nconds=5,
            nfreqs=7,
            nreps=3,
            vary_neurite=True,
            vary_nmda=False,
        ):
        if vary_neurite == vary_nmda:
            print("ERROR: Either vary_neurite or vary_nmda need to be 'True'")
            raise SystemExit(-1)
        #
        #Some fixed Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.01#0.025
        dur = 500.0
        nv = 0.9
        #
        #ParametersA
        P = {}
        P["N"] = nconds * nfreqs
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
        # Now define the variable parameters. The repeated = y, the tiled = x!!
        afreq = np.geomspace(35.0, 90.0, nfreqs)#(35.0,90.0), ###40/120#35/70 changed 2021-06-22
        aitv = np.round(1000.0 / afreq, 1)
        bfreq = np.geomspace(75.0, 333.0, nfreqs)##(75.0,333.0),#60/250#75/400 changed 2021-06-22
        bitv = np.round(1000.0 / bfreq, 1)
        P["afreq"] = np.tile(afreq, nconds)
        P["aitv"] = np.tile(aitv, nconds)
        P["bfreq"] = np.tile(bfreq, nconds)
        P["bitv"] = np.tile(bitv, nconds)
        #
        allL = np.linspace(0.1, 120.0, nconds)
        alltau = np.linspace(20.0, 120.0, nconds)##20/100 2021-06-21
        #
        if vary_neurite:
            P["pNeuritL"] = np.repeat(allL, nfreqs)
        else:
            P["pNeuritL"] = np.repeat(60.0, P["TotalN"])
        if vary_nmda:
            P["apicaltau"] = np.repeat(alltau, nfreqs)
        else:
            P["apicaltau"] = np.repeat(80.0, P["TotalN"])##100.0 2021-06-21
        #
        return P


if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_Figure4.py weload ncores nfreqs nconds nreps
    inputargs = sys.argv[1:]
    myargs = [1, 4, 7, 5, 3]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    ncores = int(myargs[1])
    nfreqs = int(myargs[2])
    nconds = int(myargs[3])
    nreps = int(myargs[4])
    #------------------
    #Run the show
    if os.path.isfile("./data/SCNv2_Figure4A.npy") and weload:
        print("Loading Data!")
        outputA = np.load("./data/SCNv2_Figure4A.npy", allow_pickle=True)
        PA = np.load("./data/SCNv2_Figure4A_P.npy", allow_pickle=True)
        PA = PA.tolist()
        outputB = np.load("./data/SCNv2_Figure4B.npy", allow_pickle=True)
        PB = np.load("./data/SCNv2_Figure4B_P.npy", allow_pickle=True)
        PB = PB.tolist()
    else:
        PA = getparams(
            ncores=ncores,
            nfreqs=nfreqs,
            nconds=nconds,
            nreps=nreps,
            vary_neurite=True,
            vary_nmda=False,
        )
        outputA = []
        outputA.append(myMPhandler(PA))
        outputA = np.array(outputA, dtype=object)
        np.save("./data/SCNv2_Figure4A.npy", outputA, allow_pickle=True)
        np.save("./data/SCNv2_Figure4A_P.npy", PA, allow_pickle=True)
        #
        PB = getparams(
            ncores=ncores,
            nfreqs=nfreqs,
            nconds=nconds,
            nreps=nreps,
            vary_neurite=False,
            vary_nmda=True,
        )
        outputB = []
        outputB.append(myMPhandler(PB))
        outputB = np.array(outputB, dtype=object)
        np.save("./data/SCNv2_Figure4B.npy", outputB, allow_pickle=True)
        np.save("./data/SCNv2_Figure4B_P.npy", PB, allow_pickle=True)
    #
    print("done")
    #plotres(outputA, outputB, PA, PB)
    #plt.show()
    fhandle1, fhandle2 = plotres(outputA, outputB, PA, PB)
    pp = PdfPages("./figures/SCNv2_Figure4_new.pdf")
    pp.savefig(fhandle2)
    pp.savefig(fhandle1)
    pp.close()


















