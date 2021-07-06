#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This takes precalculated data and generates an illustration with raster- and psth-plots
that can actually be imported in inkscape without problems.

Created: 2021-07-01
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
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rc("text", usetex=False)
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("axes", labelsize="small")


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
    mybins = np.linspace(0, PA["dur"][0], int(PA["dur"][0] / 10))
    nrep = PA["nreps"][0]
    icond = 6
    #
    sp1 = plt.subplot(3,1,1)
    plt.title("Apical only")
    plt.xlim((0,PA["dur"][0]))
    plt.plot((PA["astart"][0], PA["astart"][0] + PA["adur"][0]), (-1, -1), "g-")
    for irep in range(PA["nreps"][0]):
        if irep%2 == 0:
            ypos = -2 + ((2*irep) / (PA["nreps"][0]))
            xvals = np.array(a_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color="b", marker=dottype, markersize=msize, linestyle=" ")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Input Frequency (Hz)")
    #
    vals,bins = np.histogram(np.concatenate(a_t[icond]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.0, 0.0, 0.5))
    #
    #
    sp2 = plt.subplot(3,1,2, sharex=sp1, sharey=sp1)
    plt.title("Basal only")
    plt.xlim((0,PB["dur"][0]))
    plt.plot((PB["bstart"][0], PB["bstart"][0] + PB["bdur"][0]), (-1, -1), "g-")
    plt.plot([0, 500], [icond,icond], "k--", linewidth=0.5)
    for irep in range(PB["nreps"][0]):
        if irep%2 == 0:
            ypos = -2 + ((2*irep) / (PB["nreps"][0]))
            xvals = np.array(b_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color="r", marker=dottype, markersize=msize, linestyle=" ")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Input Frequency (Hz)")
    #
    vals,bins = np.histogram(np.concatenate(b_t[icond]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.5, 0.0, 0.0))
    #
    #
    sp3 = plt.subplot(3,1,3, sharex=sp1, sharey=sp1)
    plt.title("Apical+Basal")
    plt.xlim((0,PC["dur"][0]))
    plt.plot((PC["bstart"][0], PC["bstart"][0] + PC["bdur"][0]), (-1, -1), "g-")
    plt.plot([0, 500], [icond,icond], "k--", linewidth=0.5)
    for irep in range(PC["nreps"][0]):
        if irep%2 == 0:
            ypos = -2 + ((2*irep) / (PC["nreps"][0]))
            xvals = np.array(c_t[icond][irep])
            yvals = np.ones(xvals.size) * ypos
            plt.plot(xvals, yvals, color="k", marker=dottype, markersize=msize, linestyle=" ")
    plt.xlabel("Time (ms)")
    plt.ylabel(r"Input Salience $\%$")
    #
    vals,bins = np.histogram(np.concatenate(c_t[icond]), mybins)
    plt.plot(bins[1:], vals / nrep, color=(0.3, 0.3, 0.3))
    #
    return (fhandle)


if __name__ == "__main__":
    if os.path.isfile("./data/SCNv2_SaliencetestsA.npy"):
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
        print("No data found, please calculate first")
    fhandle = plotres(outputA, outputB, outputC, PA, PB, PC)
    pp = PdfPages("./figures/SCNv2_Figure2_PSTHillu.pdf")
    pp.savefig(fhandle)
    pp.close()
    fhandle.savefig("./figures/SCNv2_Figure2_PSTHillu.svg")


















