#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates "Figure 1" which deals with deterministic inputs, i.e. simulated 
slice experiments. It shows the different thresholds for apical and basal and combined.

Created: 2021-05-27
Revised: 2021-06-23 (Phaseplaneplots etc)
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
from progress.bar import ChargingBar
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

def simulate_data(nconds, tstop, dt):
    bar = ChargingBar("Simulation running", max=nconds)
    vma = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vma[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vman = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vman[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmb = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmb[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmab = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmab[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    vmabn = np.zeros((nconds+1, int(np.ceil(tstop / dt))))
    vmabn[0,:] = np.linspace(0.0,tstop-dt,int(np.ceil(tstop / dt)))
    sa = []
    san = []
    sb = []
    sab = []
    sabn = []
    useperc = np.linspace(0.0, 1.0, nconds)
    for thiscond in range(nconds):
        thisRA = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, False),
            pstimulation=(0.0, 1, tstop, useperc[thiscond], 0.0, 1, tstop, 0.0),
            hasinputactivity=(False, False),
            noiseval=0.1,
        )
        vma[thiscond + 1,:] = thisRA["AVm"]
        SA = SCNv2.SimpleDetectAP(
            thisRA["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sa.append(SA["PeakT"])
        #
        thisRAN = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, False),
            pstimulation=(0.0, 1, tstop, useperc[thiscond], 0.0, 1, tstop, 0.0),
            hasinputactivity=(False, False),
            noiseval=0.1,
            hasnmda=False,
        )
        vman[thiscond + 1,:] = thisRAN["AVm"]
        SAN = SCNv2.SimpleDetectAP(
            thisRAN["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        san.append(SAN["PeakT"])
        #
        thisRB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(False, True),
            pstimulation=(0.0, 1, tstop, 0.0, 0.0, 1, tstop, useperc[thiscond]),
            hasinputactivity=(False, False),
            noiseval=0.1,
        )
        vmb[thiscond + 1,:] = thisRB["AVm"]
        SB = SCNv2.SimpleDetectAP(
            thisRB["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sb.append(SB["PeakT"])
        #
        stimparams = (
            0.0, 1, tstop, useperc[thiscond] * 0.5,
            0.0, 1, tstop, useperc[thiscond] * 0.5,
        )
        thisRAB = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, True),
            pstimulation=stimparams,
            hasinputactivity=(False, False),
            noiseval=0.1,
        )
        vmab[thiscond + 1,:] = thisRAB["AVm"]
        SAB = SCNv2.SimpleDetectAP(
            thisRAB["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sab.append(SAB["PeakT"])
        #
        thisRABN = SCNv2.runmodel(
            tstop=tstop,
            dt=dt,
            hasstimulation=(True, True),
            pstimulation=stimparams,
            hasinputactivity=(False, False),
            noiseval=0.1,
            hasnmda=False,
        )
        vmabn[thiscond + 1,:] = thisRABN["AVm"]
        SABN = SCNv2.SimpleDetectAP(
            thisRABN["AVm"],
            thr=-50.0,
            dt=dt,
            LM=-20,
            RM=10,
        )
        sabn.append(SABN["PeakT"])
        #
        bar.next()
    D = {
        "vma": vma, 
        "vman": vman,
        "vmb": vmb, 
        "vmab": vmab,
        "vmabn": vmabn,
        "sa": sa,
        "san": san,
        "sb": sb,
        "sab": sab,
        "sabn": sabn,
        }
    bar.finish()
    return(D)


def mynormalize(eingang):
    ausgang = np.array(eingang)
    ausgang = ausgang - np.min(ausgang)
    ausgang = ausgang / np.max(ausgang)
    return ausgang


def make_plots(D):
    nconds = D["vma"].shape[0] - 1
    #if nconds > 10:
    #    plotconds = np.arange(0, nconds, int(nconds/10))
    #else:
    #    plotconds = range(nconds)
    plotconds = (40, 55, 65, 95)
    xl = (0, 100.0)
    mygray = (.3, .3, .3)
    percs = np.linspace(0.0, 1.0, nconds) * 8.84
    fh = plt.figure(figsize=(8,10))
    sh11 = plt.subplot(5,2,1)
    sh12 = plt.subplot(5,2,3, sharex=sh11, sharey=sh11)
    sh13 = plt.subplot(5,2,5, sharex=sh11, sharey=sh11)
    sh14 = plt.subplot(5,2,7, sharex=sh11, sharey=sh11)
    sh15 = plt.subplot(5,2,9, sharex=sh11, sharey=sh11)
    sh16 = plt.subplot(5,4,4)
    mxa = []
    mxan = []
    mxb = []
    mxab = []
    mxabn = []
    fsl_a = np.zeros(nconds) * np.nan
    fsl_b = np.zeros(nconds) * np.nan
    fsl_an = np.zeros(nconds) * np.nan
    fsl_ab = np.zeros(nconds) * np.nan
    fsl_abn = np.zeros(nconds) * np.nan
    for iii in range(nconds):
        mxa.append(np.max(D["vma"][iii + 1, :]))
        mxan.append(np.max(D["vman"][iii + 1, :]))
        mxb.append(np.max(D["vmb"][iii + 1, :]))
        mxab.append(np.max(D["vmab"][iii + 1, :]))
        mxabn.append(np.max(D["vmabn"][iii + 1, :]))
        if D["sa"][iii]:
            fsl_a[iii] = np.min(D["sa"][iii])
        if D["sb"][iii]:
            fsl_b[iii] = np.min(D["sb"][iii])
        if D["san"][iii]:
            fsl_an[iii] = np.min(D["san"][iii])
        if D["sab"][iii]:
            fsl_ab[iii] = np.min(D["sab"][iii])
        if D["sabn"][iii]:
            fsl_abn[iii] = np.min(D["sabn"][iii])
        if np.isin(iii, plotconds):
            sh11.plot(D["vma"][0, :], D["vma"][iii + 1, :], "b-", linewidth=0.5)
            sh13.plot(D["vman"][0, :], D["vman"][iii + 1, :], "m-", linewidth=0.5)
            sh12.plot(D["vmb"][0, :], D["vmb"][iii + 1, :], "r-", linewidth=0.5)
            sh14.plot(D["vmab"][0, :], D["vmab"][iii + 1, :], "k-", linewidth=0.5)
            sh15.plot(D["vmabn"][0, :], D["vmabn"][iii + 1, :], color=mygray, linewidth=0.5)
    mxa = mynormalize(mxa)
    mxb = mynormalize(mxb)
    mxab = mynormalize(mxab)
    mxan = mynormalize(mxan)
    mxabn = mynormalize(mxabn)
    sh11.set_xlabel("Time (ms)")
    sh11.set_ylabel("Vm (mV)")
    sh11.set_title("Apical only")
    sh12.set_xlabel("Time (ms)")
    sh12.set_ylabel("Vm (mV)")
    sh12.set_title("Basal only")
    sh13.set_xlabel("Time (ms)")
    sh13.set_ylabel("Vm (mV)")
    sh13.set_title("Apical only, no NMDA")
    sh14.set_xlabel("Time (ms)")
    sh14.set_ylabel("Vm (mV)")
    sh14.set_title("Apical + Basal")
    sh15.set_xlabel("Time (ms)")
    sh15.set_ylabel("Vm (mV)")
    sh15.set_title("Apical + Basal, no NMDA")
    sh11.set_xlim(xl)
    #
    sh16.plot(percs, fsl_a, "b-", label="apical")
    sh16.plot(percs, fsl_an, "m-", label="basal")
    sh16.plot(percs, fsl_b, "r-", label="apical/noNMDA")
    sh16.plot(percs, fsl_ab, "k-", label="apical+basal")
    sh16.plot(percs, fsl_abn, color=mygray, label="apical+basal/noNMDA")
#    sh[1, 2].plot(percs, mxa, "b-")
#    sh[1, 2].plot(percs, mxan, "b--")
#    sh[1, 2].plot(percs, mxb, "r-")
#    sh[1, 2].plot(percs, mxab, "k-")
#    sh[1, 2].plot(percs, mxabn, "k--")
    sh16.set_xlabel("total g-syn (nS)")
    #sh[1, 2].set_ylabel("max Vm (norm.)")
    sh16.set_ylabel("AP Latency (ms)")
    #sh[1,2].set_xlim((3.5,6.5))
    sh16.legend(fontsize=4)
    #
    #get Vthres for every condition
    dt = np.mean(np.diff(D["vma"][0]))
    dtthr = 0.075 / dt
    sp21 = plt.subplot(5,4,3)
    sp22 = plt.subplot(5,4,7, sharex=sp21, sharey=sp21)
    sp23 = plt.subplot(5,4,11, sharex=sp21, sharey=sp21)
    sp24 = plt.subplot(5,4,15, sharex=sp21, sharey=sp21)
    sp25 = plt.subplot(5,4,19, sharex=sp21, sharey=sp21)
    sp21.set_xlabel("V (mV)")
    sp21.set_ylabel("dV (mV/ms)")
    sp22.set_xlabel("V (mV)")
    sp22.set_ylabel("dV (mV/ms)")
    sp23.set_xlabel("V (mV)")
    sp23.set_ylabel("dV (mV/ms)")
    sp24.set_xlabel("V (mV)")
    sp24.set_ylabel("dV (mV/ms)")
    sp25.set_xlabel("V (mV)")
    sp25.set_ylabel("dV (mV/ms)")
    Vthres_a = np.zeros(nconds) * np.nan
    dVthres_a = np.zeros(nconds) * np.nan
    Vthres_an = np.zeros(nconds) * np.nan
    dVthres_an = np.zeros(nconds) * np.nan
    Vthres_b = np.zeros(nconds) * np.nan
    dVthres_b = np.zeros(nconds) * np.nan
    Vthres_ab = np.zeros(nconds) * np.nan
    dVthres_ab = np.zeros(nconds) * np.nan
    Vthres_abn = np.zeros(nconds) * np.nan
    dVthres_abn = np.zeros(nconds) * np.nan
    for thiscond in range(nconds):
        iii = thiscond
        xa = D["vma"][iii+1][0:-1]
        ya = np.diff(D["vma"][iii+1]) / dt
        if np.isin(thiscond, plotconds):
            sp21.plot(xa,ya,color="b", linewidth=0.5)
        if np.any(ya > dtthr):
            thresI = np.argwhere(xa==np.min(xa[ya>dtthr]))
            Vthres_a[iii] = xa[thresI]
            dVthres_a[iii] = ya[thresI]
        #
        xb = D["vmb"][iii+1][0:-1]
        yb = np.diff(D["vmb"][iii+1]) / dt
        if np.isin(iii, plotconds):
            sp22.plot(xb,yb,color="r", linewidth=0.5)
        if np.any(yb > dtthr):
            thresI = np.argwhere(xb==np.min(xb[yb>dtthr]))
            Vthres_b[iii] = xb[thresI]
            dVthres_b[iii] = yb[thresI]
        #
        xan = D["vman"][iii+1][0:-1]
        yan = np.diff(D["vman"][iii+1]) / dt
        if np.isin(iii, plotconds):
            sp23.plot(xan,yan,color="m", linewidth=0.5)
        if np.any(yan > dtthr):
            thresI = np.argwhere(xan==np.min(xan[yan>dtthr]))
            Vthres_an[iii] = xan[thresI]
            dVthres_an[iii] = yan[thresI]
        #
        xab = D["vmab"][iii+1][0:-1]
        yab = np.diff(D["vmab"][iii+1]) / dt
        if np.isin(iii, plotconds):
            sp24.plot(xab,yab,color="k", linewidth=0.5)
        if np.any(yab > dtthr):
            thresI = np.argwhere(xab==np.min(xab[yab>dtthr]))
            Vthres_ab[iii] = xab[thresI]
            dVthres_ab[iii] = yab[thresI]
        #
        xabn = D["vmabn"][iii+1][0:-1]
        yabn = np.diff(D["vmabn"][iii+1]) / dt
        if np.isin(iii, plotconds):
            sp25.plot(xabn,yabn,color=mygray, linewidth=0.5)
        if np.any(yabn > dtthr):
            thresI = np.argwhere(xabn==np.min(xabn[yabn>dtthr]))
            Vthres_abn[iii] = xabn[thresI]
            dVthres_abn[iii] = yabn[thresI]
    sp21.errorbar(
        np.nanmean(Vthres_a),
        np.nanmean(dVthres_a),
        xerr=np.nanstd(Vthres_a),
        yerr=np.nanstd(dVthres_a),
        color="g",
        marker=".",
        linewidth=2,
    )
    sp22.errorbar(
        np.nanmean(Vthres_b),
        np.nanmean(dVthres_b),
        xerr=np.nanstd(Vthres_b),
        yerr=np.nanstd(dVthres_b),
        color="g",
        marker=".",
        linewidth=2,
    )
    sp23.errorbar(
        np.nanmean(Vthres_an),
        np.nanmean(dVthres_an),
        xerr=np.nanstd(Vthres_an),
        yerr=np.nanstd(dVthres_an),
        color="g",
        marker=".",
        linewidth=2,
    )
    sp24.errorbar(
        np.nanmean(Vthres_ab),
        np.nanmean(dVthres_ab),
        xerr=np.nanstd(Vthres_ab),
        yerr=np.nanstd(dVthres_ab),
        color="g",
        marker=".",
        linewidth=2,
    )
    sp25.errorbar(
        np.nanmean(Vthres_abn),
        np.nanmean(dVthres_abn),
        xerr=np.nanstd(Vthres_abn),
        yerr=np.nanstd(dVthres_abn),
        color="g",
        marker=".",
        linewidth=2,
    )
    plt.subplot(5,4,8)
    plt.plot(percs, Vthres_a, "b-", label="apical")
    plt.plot(percs, Vthres_b, "r-", label="basal")
    plt.plot(percs, Vthres_an, "m-", label="apical/noNMDA")
    plt.plot(percs, Vthres_ab, "k-", label="apical+basal")
    plt.plot(percs, Vthres_abn, color=mygray, label="apical+basal/noNMDA")
    plt.legend(fontsize=4)
    plt.xlabel("Total gsyn (nS)")
    plt.ylabel("Vthres (mV)")
    #
    plt.tight_layout()
    #
    return fh

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
    if weload and os.path.isfile("./data/Figure1_new.npy"):
        D = np.load("./data/Figure1_new.npy", allow_pickle=True)
        D = D.tolist()
    else:
        D = simulate_data(nconds, tstop, dt)
        np.save("./data/Figure1_new.npy", D, allow_pickle=True)
    fh = make_plots(D)
    pp = PdfPages("./figures/SCNv2_Figure1_new.pdf")
    pp.savefig(fh)
    pp.close()
