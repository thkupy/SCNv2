#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates the subpanels A-C of the second model figure:
Grid of many apical and basal rates vs. onset delays

The command line args are:
    SCNv2_Figure2ABC.py weload ncores nconds nreps

Created: 2021-03-18
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
    baserate = np.max(np.unique(P["bitv"]))
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    sprb = np.zeros(P["nreps"][x])
    sprab = np.zeros(P["nreps"][x])
    sprabn = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisRB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=10,
            nsynb=10,
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(250.0, baserate, 250.0+P["ab_delay"][x], P["bitv"][x]),
            inputstop=(750.0, 750.0 + P["ab_delay"][x]),
            hasnmda=True,
            seed=thisseed,
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
        sprb[irep] = len(Sb["PeakT"])
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
            seed=thisseed,
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
        sprab[irep] = len(Sab["PeakT"])
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
            seed=thisseed,
            hasfbi=True,
            noiseval=P["noiseval"][x],
        )
        Sab = SCNv2.SimpleDetectAP(
            thisRABN["AVm"],
            thr=P["thr"][x],
            dt=P["dt"][x],
            LM=-20,
            RM=10,
        )
        sprabn[irep] = len(Sab["PeakT"]) * (1000.0 / P["dur"][x])
        #-
    print((str(x)))
    return [
        np.mean(sprab),
        np.std(sprab),
        np.mean(sprabn),
        np.std(sprabn),
        np.mean(sprb),
        np.std(sprb),
    ]


def myMPhandler(P):
    p = multiprocessing.Pool(P["cores"])
    poolmess = partial(runonecondition, P=P)
    if P["mp"]:
        r = p.map(poolmess, P["Number"])
    else:  # for debug
        r = list(map(poolmess, P["Number"]))  # for debug
    return r


def plotres(output, P, x, y, xlabs, ylabs):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats
    #
    mycmap = "bone"
    nticks = 5
    fwidth = 6  # cm
    fhandle = plt.figure(figsize=(fwidth / 2.54, (fwidth * 3.0) / 2.54), dpi=600)
    #
    outsprab_mean = output[0][:, 0]
    outsprab_std = output[0][:, 1]
    outsprabn_mean = output[0][:, 2]
    outsprabn_std = output[0][:, 3]
    outsprb_mean = output[0][:, 4]
    outsprb_std = output[0][:, 5]
    sprab_mean = np.reshape(outsprab_mean, (P["N"], P["N"]))
    sprab_std = np.reshape(outsprab_std, (P["N"], P["N"]))
    sprabn_mean = np.reshape(outsprabn_mean, (P["N"], P["N"]))
    sprabn_std = np.reshape(outsprabn_std, (P["N"], P["N"]))
    sprb_mean = np.reshape(outsprb_mean, (P["N"], P["N"]))
    sprb_std = np.reshape(outsprb_std, (P["N"], P["N"]))
    nmda_diff =  np.reshape(outsprab_mean - outsprabn_mean, (P["N"], P["N"]))

    aplusb = np.reshape(np.tile(sprab_mean[0],P["N"]), (P["N"],P["N"]))
    aplusbn = np.reshape(np.tile(sprabn_mean[0],P["N"]), (P["N"],P["N"]))
    Ienhance = sprab_mean > aplusb
    Ienhance = Ienhance.astype(int)
    Ienhancen = sprabn_mean > aplusbn
    Ienhancen = Ienhancen.astype(int)
    #
    ax1 = plt.subplot(411)
    CS1 = plt.contourf(
        x,
        y,
        sprab_mean,
        np.linspace(0.0, 150.0, 27),
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    #plt.contour(x,y,Ienhance, (0.9,1.1,))
    ax1.set_title("NMDA")
    ax1.set_xlabel(xlabs)
    ax1.set_ylabel(ylabs)
    #
    cbar1 = plt.colorbar(CS1, use_gridspec=True)
    cbar1.ax.set_ylabel(u"Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar1.locator = tl
    cbar1.update_ticks()
    #
    ax2 = plt.subplot(412)
    CS2 = plt.contourf(
        x,
        y,
        sprabn_mean,
        np.linspace(0.0, 150.0, 27),
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    #plt.contour(x,y,Ienhancen, (0.9,1.1,))
    ax2.set_title("no NMDA")
    ax2.set_xlabel(xlabs)
    ax2.set_ylabel(ylabs)
    #
    cbar2 = plt.colorbar(CS2, use_gridspec=True)
    cbar2.ax.set_ylabel(u"Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar2.locator = tl
    cbar2.update_ticks()
    #
    ax3 = plt.subplot(413)
    CS3 = plt.contourf(
        x,
        y,
        nmda_diff,
        np.linspace(-50.0, 100.0, 27),
        cmap="seismic",
        extend="both",
    )  # repeated = y, tiled = x!!
    ax3.set_title("diff")
    ax3.set_xlabel(xlabs)
    ax3.set_ylabel(ylabs)
    #
    cbar3 = plt.colorbar(CS3, use_gridspec=True)
    cbar3.ax.set_ylabel(r"$\Delta Rate (Hz)$")
    tl = MaxNLocator(nbins=5)
    cbar3.locator = tl
    cbar3.update_ticks()
    #
    ax4 = plt.subplot(414)
    CS4 = plt.contourf(
        x,
        y,
        sprb_mean,
        np.linspace(0.0, 150.0, 27),
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    ax4.set_title("basal vs spont-apical")
    ax4.set_xlabel(xlabs)
    ax4.set_ylabel(ylabs)
    #
    cbar4 = plt.colorbar(CS4, use_gridspec=True)
    cbar4.ax.set_ylabel(u"Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar4.locator = tl
    cbar4.update_ticks()
    #
    plt.tight_layout()
    return fhandle

if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_Figure2ABC.py weload ncores nconds nreps
    inputargs = sys.argv[1:]
    myargs = [1, 4, 5, 3]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    ncores = int(myargs[1])
    nconds = int(myargs[2])
    nreps = int(myargs[3])
    #------------------
    #Run the show
    if os.path.isfile("./data/Figure2ABC.npy") and weload:
        print("Data for SCNv2 Figure2ABC found... loading!")
        output = np.load("./data/Figure2ABC.npy", allow_pickle=True)
        P = np.load("./data/Figure2ABC_P.npy", allow_pickle=True)
        P = P.tolist()
    else:
        #Some fixes Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.025
        dur = 1000.0
        aitv = 12.5#this is the fixed interval of the apical input
        nv = 0.9
        output = []
        P = {}
        P["N"] = nconds
        P["cores"] = ncores
        P["TotalN"] = int(P["N"] ** 2)
        P["Number"] = list(range(P["TotalN"]))
        P["mp"] = True
        P["Seed"] = 322453
        P["AdvSeed"] = True
        #########################################
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["aitv"] = np.repeat(aitv, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        ###########################################
        # Now define the two variable parameters. The repeated = y, the tiled = x!!
        bfreq = np.geomspace(10.0, 200.0, P["N"])
        bitv = np.round(1000.0 / bfreq)
        alldelays = np.round(np.linspace(-250.0, 250.0, P["N"]))
        P["bfreq"] = np.repeat(np.round(bfreq), P["N"]) 
        P["bitv"] = np.repeat(bitv, P["N"])
        P["ab_delay"] = np.tile(alldelays, P["N"])
        # make go!
        output.append(myMPhandler(P))
        output = np.array(output)
        np.save("./data/Figure2ABC.npy", output, allow_pickle=True)
        np.save("./data/Figure2ABC_P.npy", P, allow_pickle=True)
    #
    fhandle = plotres(
        output=output,
        P=P,
        x=np.unique(P["ab_delay"]),
        y=np.unique(P["bfreq"]),
        xlabs=u"Basal Delay (ms)",
        ylabs=u"Basal Mean Input Freq. (Hz)",
    )
    pp = PdfPages("./figures/Figure2ABC.pdf")
    pp.savefig()
    pp.close()


















