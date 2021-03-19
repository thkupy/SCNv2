#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
With this program we test for every cell (N = 18 as of feb 2021) at which CF/Stimfreq-
combination the best effect of dendritic synapses occurs.

You can pass arguments to this function as in:
    python3 moredend_CFtest.py weload ncores nconds cell

    weload: 1 = load data instead of calculating new (use 0 or anything to recalc)
    ncores: number of core to run on (default = 4)
    nconds: number of conditions (in a nconds x nconds matrix!) to test (default = 5)
    dur: number of milliseconds per condition (default = 2000)
    cell: which cell (1 - 18 (see semodels.BS_3D)) (default = 1)

Created on 17 feb 2021
@author: Thomas Kuenzel (kuenzel{at}bio2.rwth-aachen.de)
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import semodels as sem
import sehelper as seh
import multiprocessing
from functools import partial

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

def runonce(x, P):
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    thisdur = (P["dur"][x] - 5.0) / 1000.0
    N = P["Nsyn"][x]
    anf_num = (N + 1, 0, 0)
    if P["AdvZseed"]:
        thisZseed = P["Zseed"] + x
    else:
        thisZseed = P["Zseed"]
    if P["AdvEoHseed"]:
        thisEoHseed = P["EoHseed"] + x
    else:
        thisEoHseed = P["EoHseed"]
    SO = seh.create_sound(
        fs=100e3, freq=P["freq"][x], duration=thisdur, dbspl=P["dB"][x]
    )
    SP = seh.create_spk(
        SO, fs=100e3, N=P["Nsyn"][x] + 1, cf=P["cf"][x], seed=thisZseed, anf_num=anf_num
    )  # generate enough spiketrains
    S = np.array(SP["spikes"] * 1000.0)
    R0 = sem.SE_3D(
        S,
        Simdur=P["dur"][x],
        dt=P["dt"][x],
        N=P["Nsyn"][x],
        G_Dend=0.0,
        EoHseed=thisEoHseed,
        cell=P["cell"][x],
    )
    RD = sem.SE_3D(
        S,
        Simdur=P["dur"][x],
        dt=P["dt"][x],
        N=P["Nsyn"][x],
        G_Dend=P["gsyn"][x],
        EoHseed=thisEoHseed,
        cell=P["cell"][x],
    )
    Ev0 = seh.SimpleDetectAP(R0["Vm"], thr=-100, dt=P["dt"][x], LM=-20, RM=10)
    EvD = seh.SimpleDetectAP(RD["Vm"], thr=-100, dt=P["dt"][x], LM=-20, RM=10)
    APr0 = len(Ev0["PeakT"]) / (P["dur"][x] / 1000.0)
    APrD = len(EvD["PeakT"]) / (P["dur"][x] / 1000.0)
    VS0, phi0, Ray0, phases0 = seh.vectorstrength(
        Ev0["PeakT"], P["freq"][x], [0.0, P["dur"][x]]
    )
    VSD, phiD, RayD, phasesD = seh.vectorstrength(
        EvD["PeakT"], P["freq"][x], [0.0, P["dur"][x]]
    )
    print((str(x)))
    return [VS0, phi0, Ray0, APr0, VSD, phiD, RayD, APrD]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonce, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def plotres(output, P, x, y, xlabs, ylabs):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats

    titsize = 9
    mycmap = "bone"
    nticks = 5
    r_crit = 0.001  # 0.001
    outVS0 = output[0][:, 0]
    outPH0 = output[0][:, 1]
    outRC0 = output[0][:, 2]
    outAP0 = output[0][:, 3]
    outVSD = output[0][:, 4]
    outPHD = output[0][:, 5]
    outRCD = output[0][:, 6]
    outAPD = output[0][:, 7]

    outVS0[
        np.logical_or(outRC0 > r_crit, outRCD > r_crit)
    ] = 0.0  # set VS of conditions to zero that fail the rayleigh test
    outPH0[
        np.logical_or(outRC0 > r_crit, outRCD > r_crit)
    ] = 0.0  # set phi of conditions to zero that fail the rayleigh test
    outVSD[np.logical_or(outRC0 > r_crit, outRCD > r_crit)] = 0.0
    outPHD[np.logical_or(outRC0 > r_crit, outRCD > r_crit)] = 0.0

    pshift = seh.get_anglediff(outPH0, outPHD)
    pshift = np.reshape(pshift, (P["N"], P["N"]))
    APdiff = outAPD - outAP0
    APdiff = np.reshape(APdiff, (P["N"], P["N"]))
    VSdiff = outVSD - outVS0
    VSdiff = np.reshape(VSdiff, (P["N"], P["N"]))

    #added some upsampling --- that is only ok for the poster stage of this project
    APdiff = ndimage.zoom(APdiff, 5, order=3)
    VSdiff = ndimage.zoom(VSdiff, 5, order=3)
    pshift = ndimage.zoom(pshift, 5, order=3)
    x = np.interp(np.linspace(0,x.size,x.size*5),np.linspace(0,x.size,x.size),x)
    y = np.interp(np.linspace(0,y.size,y.size*5),np.linspace(0,y.size,y.size),y)

    filtsig = 1.66#0.5
    APdiff = ndimage.gaussian_filter(APdiff, sigma=filtsig, order=0)
    VSdiff = ndimage.gaussian_filter(VSdiff, sigma=filtsig, order=0)

    fwidth = 5.5  # cm
    fhandle = plt.figure(figsize=(fwidth / 2.54, (fwidth * 2.0) / 2.54), dpi=600)

    ax1 = plt.subplot(311)
    CS1 = plt.contourf(
        x,
        y,
        APdiff,
        np.linspace(-50, 50, 27),
        cmap="seismic",
        extend="both",
    )  # repeated = y, tiled = x!!
    ax1.set_title("AP Rate change")
    ax1.set_xlabel(xlabs)
    ax1.set_ylabel(ylabs)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xticks((0.2, 0.5, 1.0, 2.0))
    ax1.set_yticks((0.2, 0.5, 1.0, 2.0))
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #
    # Linear regression: Reviewer request
    yl = ax1.get_ylim()
    xl = ax1.get_xlim()
    ax1.plot([0.125, 2.5], [0.125, 2.5], "k--", linewidth=0.5)
    ax1.plot(x, y[np.argmax(APdiff, 0)], "k.", markersize=2)
    s, intc, r, p, std = stats.linregress(x, y[np.argmax(APdiff, 0)])
    ax1.plot(x, s * x + intc, "y-", linewidth=0.5)
    ax1.set_ylim(yl)
    ax1.set_xlim(xl)
    outtext = (
        "bestAPdiff regression, y="
        + str(round(s, 3))
        + "*x+"
        + str(round(intc, 3))
        + ", r="
        + str(round(r, 3))
        + ", p="
        + str(round(p, 3))
    )
    #print(outtext)
    #
    cbar1 = plt.colorbar(CS1, use_gridspec=True)
    cbar1.ax.set_ylabel(r"$\Delta$ Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar1.locator = tl
    cbar1.update_ticks()
    #
    #
    ax2 = plt.subplot(312)
    CS2 = plt.contourf(
        x,
        y,
        VSdiff,
        np.linspace(-0.1, 0.1, 21),
        cmap="seismic",
        extend="both",
    )
    ax2.set_title(r"Vectorstrength change")
    ax2.set_xlabel(xlabs)
    ax2.set_ylabel(ylabs)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xticks((0.2, 0.5, 1.0, 2.0))
    ax2.set_yticks((0.2, 0.5, 1.0, 2.0))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #
    # Linear regression: Reviewer request
    yl = ax2.get_ylim()
    xl = ax2.get_xlim()
    ax2.plot([0.125, 2.5], [0.125, 2.5], "k--", linewidth=0.5)
    ax2.plot(x, y[np.argmax(VSdiff, 0)], "k.", markersize=2)
    s, intc, r, p, std = stats.linregress(x, y[np.argmax(VSdiff, 0)])
    ax2.plot(x, s * x + intc, "y-", linewidth=0.5)
    ax2.set_ylim(yl)
    ax2.set_xlim(xl)
    outtext = (
        "best VSdiff regression, y="
        + str(round(s, 3))
        + "*x+"
        + str(round(intc, 3))
        + ", r="
        + str(round(r, 3))
        + ", p="
        + str(round(p, 3))
    )
    #print(outtext)
    cbar2 = plt.colorbar(CS2, use_gridspec=True)
    cbar2.ax.set_ylabel(r"$\Delta$ Vectorstrength")
    tl = MaxNLocator(nbins=5)
    cbar2.locator = tl
    cbar2.update_ticks()
    #
    #
    ax3 = plt.subplot(313)
    CS3 = plt.contourf(
        x,
        y,
        pshift,
        np.linspace(-0.1, 0.1, 21),
        cmap="seismic",
        extend="both",
    )  # repeated = y, tiled = x!!
    ax3.set_title("Phase shift")
    ax3.set_xlabel(xlabs)
    ax3.set_ylabel(ylabs)
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_xticks((0.2, 0.5, 1.0, 2.0))
    ax3.set_yticks((0.2, 0.5, 1.0, 2.0))
    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #
    # Linear regression: Reviewer request
    yl = ax3.get_ylim()
    xl = ax3.get_xlim()
    ax3.plot([0.0, 2.5], [0.0, 2.5], "k--", linewidth=0.5)
    ax3.plot(x, y[np.argmax(np.abs(VSdiff), 0)], "k.", markersize=2)
    s, intc, r, p, std = stats.linregress(x, y[np.argmax(np.abs(VSdiff), 0)])
    ax3.plot(x, s * x + intc, "y-", linewidth=0.5)
    ax3.set_ylim(yl)
    ax3.set_xlim(xl)
    outtext = (
        "best phidiff regression, y="
        + str(round(s, 3))
        + "*x+"
        + str(round(intc, 3))
        + ", r="
        + str(round(r, 3))
        + ", p="
        + str(round(p, 3))
    )
    #print(outtext)
    #
    cbar3 = plt.colorbar(CS3, use_gridspec=True)
    cbar3.ax.set_ylabel(r"$\Delta \varphi$ (cycles)")
    tl = MaxNLocator(nbins=5)
    cbar3.locator = tl
    cbar3.update_ticks()

    #generate output about best conditions
    val1=np.max(APdiff,axis=0)
    arg1=np.argmax(APdiff,axis=0)
    val2=np.max(val1)
    arg1=arg1[np.argmax(val1)]
    arg2=np.argmax(val1)
    bestAPdiff = (x[arg2], y[arg1], val2)
    val1=np.max(VSdiff,axis=0)
    arg1=np.argmax(VSdiff,axis=0)
    val2=np.max(val1)
    arg1=arg1[np.argmax(val1)]
    arg2=np.argmax(val1)
    bestVSdiff = (x[arg2], y[arg1], val2)
    #
    ax1.plot(bestAPdiff[0],bestAPdiff[1], "ko", markersize=7, markerfacecolor="none")
    ax2.plot(bestVSdiff[0],bestVSdiff[1], "ko", markersize=7, markerfacecolor="none")
    plt.tight_layout()
    return(fhandle, bestAPdiff, bestVSdiff)


if __name__ == "__main__":
    addedargs = sys.argv[1:]  # args are (fname, weload, ncores, nconds, dur, cell)
    myargs = [1, 4, 5, 2000, 1]
    for iarg, thisarg in enumerate(addedargs):
        myargs[iarg] = int(thisarg)
    ncores = myargs[1]
    nconds = myargs[2]
    dur = float(myargs[3])
    cell = myargs[4]
    if myargs[0] == 1:
        weload = True
    else:
        weload = False
    #
    ####LOAD DATA FOR moredend_CFtest (IF IT EXISTS)
    if os.path.isfile("./results/moredend_CF_cell" + str(cell) + ".npy"):
        print("Data for moredend_CF_cell" + str(cell) + " found... loading!")
        output = np.load("./results/moredend_CF_cell" + str(cell) + ".npy", allow_pickle=True)
        P = np.load("./results/moredend_CF_cell" + str(cell) + "_P.npy", allow_pickle=True)
        P = P.tolist()
    else:
        #fixed model parameters#
        N = 32
        gsyn = 0.064
        dB = 70.0
        conditions = nconds
        cores = ncores
        dt = 0.01
        output = []
        # Model Parameters (all in a linearly aranged fashion, so that a minimal
        # amount of programming is required to change the experiment).
#        S,
#        Simdur=P["dur"][x],
#        N=P["Nsyn"][x],
#        G_Dend=0.0,
#        EoHseed=thisEoHseed,
#        cell=P["cell"][x],
        P = {}
        P["N"] = conditions
        P["cores"] = cores
        P["TotalN"] = int(P["N"] ** 2)
        P["Number"] = list(range(P["TotalN"]))
        P["mp"] = True
        P["Zseed"] = 45453
        P["AdvZseed"] = True
        P["EoHseed"] = 34823
        P["AdvEoHseed"] = True
        #########################################
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["dB"] = np.repeat(dB, P["TotalN"])
        P["Nsyn"] = np.repeat(int(N), P["TotalN"])
        P["gsyn"] = np.repeat(gsyn, P["TotalN"])
        P["cell"] = np.repeat(cell, P["TotalN"])
        # Now define the two variable parameters. The repeated = y, the tiled = x!!
        allcf = np.round(np.geomspace(125.1, 2501.0, P["N"]))
        allfreq = np.round(np.geomspace(125.1, 2501.0, P["N"]))
        P["freq"] = np.repeat(allfreq, P["N"])
        P["cf"] = np.tile(allcf, P["N"])
        # make go!
        output.append(myMPhandler(P))
        output = np.array(output)
        np.save("./results/moredend_CF_cell" + str(cell) + ".npy", output, allow_pickle=True)
        np.save("./results/moredend_CF_cell" + str(cell) + "_P.npy", P, allow_pickle=True)

    ###Plot results
    fhandle, bestAP, bestVS = plotres(
        output=output,
        P=P,
        x=np.unique(P["cf"]) / 1000.0,
        y=np.unique(P["freq"]) / 1000.0,
        xlabs=r"$CF (kHz)$",
        ylabs=r"$Frequency (kHz)$",
    )
    pp = PdfPages("./figs/moredend_CF_cell" + str(cell) + ".pdf")
    pp.savefig()
    pp.close()
    if os.path.isfile("./results/moredend_allbest.npy"):
        #column is cellnr, row is: bestap-x, bestap-y, bestvs-x, bestvs-y, dur in s, nconds
        allbest = np.load("./results/moredend_allbest.npy")
    else:
        allbest = np.ones((19,8))*-1
    allbest[cell,:] = [
        bestAP[0],
        bestAP[1],
        bestAP[2],
        bestVS[0],
        bestVS[1],
        bestVS[2],
        dur/1000.0,
        nconds,
    ]
    np.save("./results/moredend_allbest.npy",allbest, allow_pickle=True)
