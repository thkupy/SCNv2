#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This generates test data on the number of output spikes generated for
N synapses at a rate of M Hz (both basal, apical and combined (N/2) per side).

The command line args are:
    SCNv2_Figure2ABC.py weload ncores nconds

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
    baserate = 200#np.max(np.unique(P["aitv"][x]))
    ####GENERATE SPIKE INPUTS (the spiketimes are always the same to improve comparability)
    if P["AdvSeed"]:
        thisseed = P["Seed"] + (x * 50)
    else:
        thisseed = P["Seed"]
    spra = np.zeros(P["nreps"][x])
    sprb = np.zeros(P["nreps"][x])
    sprab = np.zeros(P["nreps"][x])
    spran = np.zeros(P["nreps"][x])
    sprabn = np.zeros(P["nreps"][x])
    for irep in range(P["nreps"][x]):
        thisRA = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyn"][x]),
            nsynb=int(P["nsyn"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, baserate),
            inputstop=(950.0, 950.0),
            hasnmda=True,
            seed=thisseed+irep,
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
        spra[irep] = len(Sa["PeakT"])/0.9
        #
        thisRB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyn"][x]),
            nsynb=int(P["nsyn"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, baserate, 50.0, P["bitv"][x]),
            inputstop=(950.0, 950.0),
            hasnmda=True,
            seed=thisseed+irep,
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
        sprb[irep] = len(Sb["PeakT"])/0.9
        #
        thisRAB = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(np.round(P["nsyn"][x])),
            nsynb=int(np.round(P["nsyn"][x])),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, P["bitv"][x]),
            inputstop=(950.0, 950.0),
            hasnmda=True,
            seed=thisseed+irep,
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
        sprab[irep] = len(Sab["PeakT"])/0.9
        #
        thisRAN = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyn"][x]),
            nsynb=int(P["nsyn"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, baserate),
            inputstop=(950.0, 950.0),
            hasnmda=False,
            seed=thisseed+irep,
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
        spran[irep] = len(Sa["PeakT"])/0.9
        #
        thisRABN = SCNv2.runmodel(
            tstop=P["dur"][x],
            dt=P["dt"][x],
            nsyna=int(P["nsyn"][x]),
            nsynb=int(P["nsyn"][x]),
            hasstimulation=(False, False),
            hasinputactivity=(True, True),
            pinputactivity=(50.0, P["aitv"][x], 50.0, P["aitv"][x]),
            inputstop=(950.0, 950.0),
            hasnmda=False,
            seed=thisseed+irep,
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
        sprabn[irep] = len(Sa["PeakT"])/0.9
        #
    print("x: " + str(x))
    return [
        np.mean(spra),
        np.std(spra),
        np.mean(sprb),
        np.std(sprb),
        np.mean(sprab),
        np.std(sprab),
        np.mean(spran),
        np.std(spran),
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


def plotres(output, P, x, y, xlabs, ylabs):
    from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MaxNLocator
    import scipy.ndimage as ndimage
    from scipy import stats
    #
    mycmap = "bone"
    nticks = 5
    fheight = 15  # cm
    fwidth = 20
    #contourlims = (1.0,333.0)
    #ncontours = 27
    fhandle = plt.figure(figsize=(fwidth / 2.54, fheight / 2.54), dpi=600)
    #
    outspra_mean = output[0][:, 0]
    outsprb_mean = output[0][:, 2]
    outsprab_mean = output[0][:, 4]
    #
    outspran_mean = output[0][:, 6]
    outsprabn_mean = output[0][:, 8]
    #
    outaplusb = outspra_mean + outsprb_mean
    outaplusb_n = outspran_mean + outsprb_mean
    spra_mean = np.reshape(outspra_mean, (P["N"], P["N"]))
    sprb_mean = np.reshape(outsprb_mean, (P["N"], P["N"]))
    sprab_mean = np.reshape(outsprab_mean, (P["N"], P["N"]))
    #
    spran_mean = np.reshape(outspran_mean, (P["N"], P["N"]))
    #
    sprabn_mean = np.reshape(outsprabn_mean, (P["N"], P["N"]))
    #
    aplusb = np.reshape(outaplusb,(P["N"],P["N"]))
    aplusb_n = np.reshape(outaplusb_n,(P["N"],P["N"]))
    Ienhance = sprab_mean > aplusb
    Ienhance_n = sprabn_mean > aplusb_n
    Ienhance = Ienhance.astype(int)
    Ienhance_n = Ienhance_n.astype(int)
    #
    ax1 = plt.subplot(321)
    CS1 = plt.contourf(
        x,
        y,
        spra_mean,
        #np.geomspace(contourlims[0], contourlims[1], ncontours),
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    #plt.contour(x,y,Ienhance, (0.9,1.1,))
    ax1.set_title("Apical only")
    ax1.set_xlabel(xlabs)
    ax1.set_ylabel(ylabs)
    #
    cbar1 = plt.colorbar(CS1, use_gridspec=True)
    cbar1.ax.set_ylabel(u"Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar1.locator = tl
    cbar1.update_ticks()
    #
    ax2 = plt.subplot(323)
    CS2 = plt.contourf(
        x,
        y,
        sprb_mean,
        #np.linspace(0, 10, 10),
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    #plt.contour(x,y,Ienhancen, (0.9,1.1,))
    ax2.set_title("Basal only")
    ax2.set_xlabel(xlabs)
    ax2.set_ylabel(ylabs)
    #
    cbar2 = plt.colorbar(CS2, use_gridspec=True)
    cbar2.ax.set_ylabel(u"Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar2.locator = tl
    cbar2.update_ticks()
    #
    ax3 = plt.subplot(325)
    CS3 = plt.contourf(
        x,
        y,
        sprab_mean,
        #np.geomspace(contourlims[0], contourlims[1], ncontours),
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    ax3.set_title("Apical+Basal")
    ax3.set_xlabel(xlabs)
    ax3.set_ylabel(ylabs)
    #
    cbar3 = plt.colorbar(CS3, use_gridspec=True)
    cbar3.ax.set_ylabel("Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar3.locator = tl
    cbar3.update_ticks()
    plt.contour(x,y,Ienhance,1)
    #
    ax4 = plt.subplot(322)
    CS4 = plt.contourf(
        x,
        y,
        spran_mean,
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    ax4.set_title("Apical only, no-NMDA")
    ax4.set_xlabel(xlabs)
    ax4.set_ylabel(ylabs)
    #
    cbar4 = plt.colorbar(CS4, use_gridspec=True)
    cbar4.ax.set_ylabel("Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar4.locator = tl
    cbar4.update_ticks()
    #
    ax5 = plt.subplot(324)
    CS5 = plt.contourf(
        x,
        y,
        sprabn_mean,
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    ax5.set_title("Apical + Basal, no-NMDA")
    ax5.set_xlabel(xlabs)
    ax5.set_ylabel(ylabs)
    #
    cbar5 = plt.colorbar(CS5, use_gridspec=True)
    cbar5.ax.set_ylabel("Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar5.locator = tl
    cbar5.update_ticks()
    plt.contour(x,y,Ienhance_n,1)
    #
    #
    ax6 = plt.subplot(326)
    CS6 = plt.contourf(
        x,
        y,
        sprab_mean-sprabn_mean,
        cmap="inferno",
    )  # repeated = y, tiled = x!!
    ax6.set_title("NMDA effect")
    ax6.set_xlabel(xlabs)
    ax6.set_ylabel(ylabs)
    #
    cbar6 = plt.colorbar(CS5, use_gridspec=True)
    cbar6.ax.set_ylabel("Rate (Hz)")
    tl = MaxNLocator(nbins=5)
    cbar6.locator = tl
    cbar6.update_ticks()
    #
    plt.tight_layout()
    return fhandle

if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_Inputtests.py weload ncores nconds nreps
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
    if os.path.isfile("./data/SCNv2_Inputtests.npy") and weload:
        print("Data for SCNv2_Inputtests found... loading!")
        output = np.load("./data/SCNv2_Inputtests.npy", allow_pickle=True)
        P = np.load("./data/SCNv2_Inputtests_P.npy", allow_pickle=True)
        P = P.tolist()
    else:
        #Some fixes Parameters, could be exposed to user later
        apthr = -50.0
        dt = 0.025
        dur = 1000.0
        nv = 0.9
        output = []
        P = {}
        P["N"] = nconds
        P["cores"] = ncores
        P["TotalN"] = int(P["N"] ** 2)
        P["Number"] = np.arange(P["TotalN"],dtype=int)
        P["mp"] = True
        P["Seed"] = 322453
        P["AdvSeed"] = True
        #########################################
        P["thr"]  = np.repeat(apthr, P["TotalN"]) 
        P["dur"] = np.repeat(dur, P["TotalN"])
        P["dt"] = np.repeat(dt, P["TotalN"])
        P["nreps"] = np.repeat(nreps, P["TotalN"])
        P["noiseval"] = np.repeat(nv, P["TotalN"])
        P["ab_delay"] = np.repeat(0.0, P["TotalN"])
        ###########################################
        # Now define the two variable parameters. The repeated = y, the tiled = x!!
        freq = np.geomspace(10.0, 100.0, P["N"])
        itv = np.round(1000.0 / freq)
        P["afreq"] = np.repeat(np.round(freq), P["N"]) 
        P["aitv"] = np.repeat(itv, P["N"])
        P["bfreq"] = np.repeat(np.round(freq), P["N"]) 
        P["bitv"] = np.repeat(itv, P["N"])
        synnums = np.ceil(np.geomspace(3,120,nconds))
        P["nsyn"] = np.tile(synnums, P["N"])
        # make go!
        output.append(myMPhandler(P))
        output = np.array(output)
        np.save("./data/SCNv2_Inputtests.npy", output, allow_pickle=True)
        np.save("./data/SCNv2_Inputtests_P.npy", P, allow_pickle=True)
    #
    fhandle = plotres(
        output=output,
        P=P,
        x=np.unique(P["nsyn"]),
        y=np.unique(P["afreq"]),
        xlabs=u"N Synapses",
        ylabs=u"Mean Input Freq. (Hz)",
    )
    pp = PdfPages("./figures/SCNv2_Inputtests.pdf")
    pp.savefig()
    pp.close()


















