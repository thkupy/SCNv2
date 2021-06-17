#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function loads the EPSC/VC data from one cell (in Stefans Export-Format in this
case) and analyzes the EPSC therein.
It will save the data and results in a more handy format [not implemented yet]
This was for comparison between different fit-methods.

The command line args are:
    SCNv2_EPSCfit_single.py filename.xlsx

Created: 2021-06-16
Revised: 
@author: kuenzel(at)bio2.rwth-aachen.de
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import curve_fit

#some global parameters
dt = 0.05#Stefan says 20kHz is sampling rate
begin = 203.0#artefact rejection
ibegin = int(203.0 / dt)



def loadEPSCdata(fdfn):
    #this is just hacked together to load the one file I have, needs to be adjusted for 
    #more productive use
    #a future load-function should be more generic. Columnheaders would be good though.
    D = pd.read_excel(fdfn, engine="openpyxl")
    D.columns = [11,12,13,14,15,21,22,23,24,25,31,32,33,34,35,41,42,43,44,45,51,52,53,54,55,61,62,63,64,65]
    D.drop(np.arange(ibegin), inplace=True)#discard evertyhing beforw ibegin
    D["Actrl"] = D.loc[:, (11,12,13,14,15)].mean(axis=1) * 1e12#in pA
    D["Ablck"] = D.loc[:, (21,22,23,24,25)].mean(axis=1) * 1e12
    D["Arecv"] = D.loc[:, (31,32,33,34,35)].mean(axis=1) * 1e12
    D["Bctrl"] = D.loc[:, (41,42,43,44,45)].mean(axis=1) * 1e12
    D["Bblck"] = D.loc[:, (51,52,53,54,55)].mean(axis=1) * 1e12
    D["Brecv"] = D.loc[:, (61,62,63,64,65)].mean(axis=1) * 1e12
    D["Actrl_s"] = D.loc[:, (11,12,13,14,15)].std(axis=1) * 1e12
    D["Ablck_s"] = D.loc[:, (21,22,23,24,25)].std(axis=1) * 1e12
    D["Arecv_s"] = D.loc[:, (31,32,33,34,35)].std(axis=1) * 1e12
    D["Bctrl_s"] = D.loc[:, (41,42,43,44,45)].std(axis=1) * 1e12
    D["Bblck_s"] = D.loc[:, (51,52,53,54,55)].std(axis=1) * 1e12
    D["Brecv_S"] = D.loc[:, (61,62,63,64,65)].std(axis=1) * 1e12
    D["tx"] = np.linspace(0, (D["Actrl"].size * dt) - dt, D["Actrl"].size)#mk timeaxis
    return(D)


def singleexp(t, s1, s2):
    #fitting function... gets called A LOT
    return s1 * np.exp(-t / s2)


def fitsingle(oldtx, oldT, S):
    T = oldT[S[2]:]
    T = T[~np.isnan(T)]
    tx = np.linspace(0.0, (T.size * dt) - dt, T.size)#mk timeaxis
    #s1*(exp(-t/s2))
    popt, pcov = curve_fit(singleexp, tx, T, bounds=([-1000.0, 1],[0, 1000.0]), p0=[-300.0, 30.0])
    #
    ss_res = np.sum((T - singleexp(tx, *popt)) ** 2)
    ss_tot = np.sum((T - np.mean(singleexp(tx, *popt))) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    #
    #DEBUG
    #plt.plot(tx, T)
    #plt.plot(tx, singleexp(tx, *popt), 'g--')
    #print(str(popt[0]) + "/" + str(popt[1]))
    #print(r2)
    #plt.show()
    return(popt, r2)


def doubleexp(t, s1, s2, s3, s4):
    #fitting function... gets called A LOT
    return (s1 * (np.exp(-t / s2))) + (s3 * (np.exp(-t / s4)))


def fitdouble(oldtx, oldT, S):
    T = oldT[S[2]:]
    T = T[~np.isnan(T)]
    tx = np.linspace(0.0, (T.size * dt) - dt, T.size)#mk timeaxis
    #
    popt, pcov = curve_fit(
        doubleexp,
        tx, 
        T, 
        bounds=([-500.0, 1, -500, 1],[0, 100.0, 0, 2000.0]), 
        p0=[-150.0, 20.0, -150.0, 200],
    )
    #
    ss_res = np.sum((T - doubleexp(tx, *popt)) ** 2)
    ss_tot = np.sum((T - np.mean(doubleexp(tx, *popt))) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    #
    #DEBUG
    #plt.plot(tx, T)
    #plt.plot(tx, doubleexp(tx, *popt), 'g--')
    #print(str(popt[0]) + "/" + str(popt[1]) + "/" + str(popt[2]) + "/" + str(popt[3]))
    #print(r2)
    #lt.show()
    return(popt, r2)


def analyzeEPSCdata(D):
    #this is specific for the dataformat in Stefans Export file.
    R={}
    R["Actrl_start"] = [D.tx.iloc[D.Actrl.argmin()], D.Actrl.min(), D.Actrl.argmin()]
    R["Ablck_start"] = [D.tx.iloc[D.Ablck.argmin()], D.Ablck.min(), D.Ablck.argmin()]
    R["Arecv_start"] = [D.tx.iloc[D.Arecv.argmin()], D.Arecv.min(), D.Arecv.argmin()]
    R["Bctrl_start"] = [D.tx.iloc[D.Bctrl.argmin()], D.Bctrl.min(), D.Bctrl.argmin()]
    R["Bblck_start"] = [D.tx.iloc[D.Bblck.argmin()], D.Bblck.min(), D.Bblck.argmin()]
    R["Brecv_start"] = [D.tx.iloc[D.Brecv.argmin()], D.Brecv.min(), D.Brecv.argmin()]
    #
    p, r2 = fitsingle(D.tx, D.Actrl, R["Actrl_start"])
    R["Actrl_single_p"] = p
    R["Actrl_single_r2"] = r2
    p, r2 = fitdouble(D.tx, D.Actrl, R["Actrl_start"])
    R["Actrl_double_p"] = p
    R["Actrl_double_r2"] = r2
    #
    p, r2 = fitsingle(D.tx, D.Ablck, R["Ablck_start"])
    R["Ablck_single_p"] = p
    R["Ablck_single_r2"] = r2
    p, r2 = fitdouble(D.tx, D.Ablck, R["Ablck_start"])
    R["Ablck_double_p"] = p
    R["Ablck_double_r2"] = r2
    #
    p, r2 = fitsingle(D.tx, D.Arecv, R["Arecv_start"])
    R["Arecv_single_p"] = p
    R["Arecv_single_r2"] = r2
    p, r2 = fitdouble(D.tx, D.Arecv, R["Arecv_start"])
    R["Arecv_double_p"] = p
    R["Arecv_double_r2"] = r2
    #
    #
    p, r2 = fitsingle(D.tx, D.Bctrl, R["Bctrl_start"])
    R["Bctrl_single_p"] = p
    R["Bctrl_single_r2"] = r2
    p, r2 = fitdouble(D.tx, D.Bctrl, R["Bctrl_start"])
    R["Bctrl_double_p"] = p
    R["Bctrl_double_r2"] = r2
    #
    p, r2 = fitsingle(D.tx, D.Bblck, R["Bblck_start"])
    R["Bblck_single_p"] = p
    R["Bblck_single_r2"] = r2
    p, r2 = fitdouble(D.tx, D.Bblck, R["Bblck_start"])
    R["Bblck_double_p"] = p
    R["Bblck_double_r2"] = r2
    #
    p, r2 = fitsingle(D.tx, D.Brecv, R["Brecv_start"])
    R["Brecv_single_p"] = p
    R["Brecv_single_r2"] = r2
    p, r2 = fitdouble(D.tx, D.Brecv, R["Brecv_start"])
    R["Brecv_double_p"] = p
    R["Brecv_double_r2"] = r2
    return(R)


def plottheresults(tx, dn, sh):
    #adds fits and startpoints to existing plots
    sh.plot(R[dn + "_start"][0], R[dn + "_start"][1], "go")
    #mkfittimeaxis
    tx = tx[R[dn + "_start"][2]:]
    fitx = np.linspace(0.0, (tx.size * dt) - dt, tx.size)#mk timeaxis
    fitvalssingle = singleexp(fitx, *R[dn + "_single_p"])
    fitvalsdouble = doubleexp(fitx, *R[dn + "_double_p"])
    sh.plot(
        fitx + R[dn + "_start"][0],
        fitvalssingle,
        color="m",
        linestyle="-",
        label='tau_single=%5.3f, r2_single=%5.3f' % (R[dn + "_single_p"][1], R[dn + "_single_r2"]),
    )
    Afast = R[dn + "_double_p"][0] / (R[dn + "_double_p"][0] + R[dn + "_double_p"][2])
    Aslow = R[dn + "_double_p"][2] / (R[dn + "_double_p"][0] + R[dn + "_double_p"][2])
    lbltuple = (Afast, R[dn + "_double_p"][1], Aslow, R[dn + "_double_p"][3], R[dn + "_double_r2"])
    lbldbltxt = "Afast=%5.3f, tau_fast=%5.3f, Aslow=%5.3f, tau_slow=%5.3f, r2_single=%5.3f" % lbltuple
    sh.plot(
        fitx + R[dn + "_start"][0],
        fitvalsdouble, 
        color="g",
        linestyle="-",
        label=lbldbltxt,
    )
    sh.legend(fontsize=7)


def plotEPSCtraces(D, R):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    repcol = (.7, .7, .7)
    for irep in range(5):
        ax[0, 0].plot(D.tx, D[10 + (irep + 1)] * 1e12, color=repcol)
        ax[0, 1].plot(D.tx, D[20 + (irep + 1)] * 1e12, color=repcol)
        ax[0, 2].plot(D.tx, D[30 + (irep + 1)] * 1e12, color=repcol)
        ax[1, 0].plot(D.tx, D[40 + (irep + 1)] * 1e12, color=repcol)
        ax[1, 1].plot(D.tx, D[50 + (irep + 1)] * 1e12, color=repcol)
        ax[1, 2].plot(D.tx, D[60 + (irep + 1)] * 1e12, color=repcol)
    ax[0, 0].plot(D.tx, D.Actrl, color="b", linewidth=2)
    #uncomment to show the std of the trace... not so nice
    #ax[0, 0].plot(D.tx, D.Actrl + D.Actrl_s, color="b", linewidth=2, linestyle = "--")
    #ax[0, 0].plot(D.tx, D.Actrl - D.Actrl_s, color="b", linewidth=2, linestyle = "--")
    ax[0, 0].plot(D.tx, D.Actrl, color="b", linewidth=2)
    plottheresults(D.tx, "Actrl", ax[0, 0])
    ax[0, 0].set_xlim((-5, 400))
    ax[0, 1].plot(D.tx, D.Ablck, color="b", linewidth=2)
    plottheresults(D.tx, "Ablck", ax[0, 1])
    ax[0, 2].plot(D.tx, D.Arecv, color="b", linewidth=2)
    plottheresults(D.tx, "Arecv", ax[0, 2])
    ax[1, 0].plot(D.tx, D.Bctrl, color="r", linewidth=2)
    plottheresults(D.tx, "Bctrl", ax[1, 0])
    ax[1, 1].plot(D.tx, D.Bblck, color="r", linewidth=2)
    plottheresults(D.tx, "Bblck", ax[1, 1])
    ax[1, 2].plot(D.tx, D.Brecv, color="r", linewidth=2)
    plottheresults(D.tx, "Brecv", ax[1, 2])
    #
    #decorations
    ax[0, 0].set_title("Control")
    ax[0, 1].set_title("APV")
    ax[0, 2].set_title("Recovery")
    ax[0, 0].set_ylabel("APICAL \n Membrane Current (pA)")
    ax[1, 0].set_ylabel("BASAL \n Membrane Current (pA)")
    ax[1,0].set_xlabel("Time re Stim (ms)")
    ax[1,1].set_xlabel("Time re Stim (ms)")
    ax[1,2].set_xlabel("Time re Stim (ms)")
    plt.tight_layout()
    plt.show()


def saveEPSCresults(R):
    print("not yet")


if __name__ == "__main__":
    #Parse command-line argument (at the moment only one, but plan for more)
    inputargs = sys.argv[1:]
    myargs = ["/home/kuenzel/Daten/SCN_onecell_20210616.xlsx",]
    for iarg, thisarg in enumerate(inputargs):
        defaultarg[iarg] = thisarg
    fdfn = myargs[0]
    D = loadEPSCdata(fdfn)#this is to easily incorporate different import methods later  
    R = analyzeEPSCdata(D)
    plotEPSCtraces(D, R)
    saveEPSCresults(R)
