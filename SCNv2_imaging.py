#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This program is about a comment by the reviewers at J Neurosci: does the model
also replicate the apical to basal spread of excitation seen in the imaging 
experiments?

Created: 2021-21-06
@author: kuenzel(at)bio2.rwth-aachen.de
"""

import sys
import os
import SCNv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def runexp(nconds, thisdt):
    startstim = 0.0
    w = 1.0
    tstop = 7.0
    AR = SCNv2.runmodel_prop(
        tstop=tstop,
        dt=thisdt,
        hasstimulation=(True, True),
        pstimulation=(startstim, 1, 5.0, w*0.8, startstim, 1, 5.0, w*0.2),#start, number, interval, weight
        nvecs=nconds,
        )
    BR = SCNv2.runmodel_prop(
        tstop=tstop,
        dt=thisdt,
        hasstimulation=(True, True),
        pstimulation=(startstim, 1, 5.0, w*0.2, startstim, 1, 5.0, w*0.8),#start, number, interval, weight
        nvecs=nconds,
        )
    MR = SCNv2.runmodel_prop(
        tstop=tstop,
        dt=thisdt,
        hasstimulation=(True, True),
        pstimulation=(startstim, 1, 5.0, w*0.5, startstim, 1, 5.0, w*0.5),#start, number, interval, weight
        nvecs=nconds,
        )
    R = {}
    R["dt"] = thisdt
    R["nconds"] = nconds
    R["AR"] = AR
    R["BR"] = BR
    R["MR"] = MR
    return(R)

def makeplots(D):
    #
    apical_l = 230.0#um
    neurite_l = 60.0#um
    soma_l = 20#um
    basal_l = 25#um
    total_l = neurite_l + soma_l + basal_l
    #
    xn = np.linspace(0.0, neurite_l, D["nconds"])#distance from top of neurite
    xs = np.linspace(neurite_l, soma_l+neurite_l, D["nconds"])#distance from top of neurite
    xb = np.linspace(neurite_l+soma_l, total_l, D["nconds"])#distance from top of neurite
    x = np.concatenate((xn,xs,xb))
    #
    fh1 = plt.figure()
    ya = np.hstack((
        np.fliplr(D["AR"]["vm_neurite"]),
        np.fliplr(D["AR"]["vm_soma"]),
        D["AR"]["vm_basal"],
        ))
    tmxa = np.argmax(ya, 0)
    tmxa_min = ((tmxa - tmxa[np.argmin(tmxa)]) * D["dt"]) * 1000.0
    tmxa = ((tmxa - tmxa[0]) * D["dt"]) * 1000.0
    yb = np.hstack((
        np.fliplr(D["BR"]["vm_neurite"]),
        np.fliplr(D["BR"]["vm_soma"]),
        D["BR"]["vm_basal"],
        ))
    tmxb = np.argmax(yb, 0)
    tmxb_min = ((tmxb - tmxb[np.argmin(tmxb)]) * D["dt"]) * 1000.0
    tmxb = ((tmxb - tmxb[0]) * D["dt"]) * 1000.0
    ym = np.hstack((
        np.fliplr(D["MR"]["vm_neurite"]),
        np.fliplr(D["MR"]["vm_soma"]),
        D["MR"]["vm_basal"],
        ))
    tmxm = np.argmax(ym, 0)
    tmxm_min = ((tmxm - tmxm[np.argmin(tmxm)]) * D["dt"]) * 1000.0
    tmxm = ((tmxm - tmxm[0]) * D["dt"]) * 1000.0
    #
    plt.plot(x,tmxa,"b.")
    plt.plot(x,tmxb,"r.")
    plt.plot(x,tmxm,"k.")
    
    #
    plt.legend(("unimodal apical", "unimodal basal", "multimodal"))
    yl = plt.ylim()
    plt.plot((neurite_l, neurite_l), yl, "k--")
    plt.text(21.0, yl[0], "pNeurite")
    plt.plot((neurite_l+soma_l, neurite_l+soma_l), yl, "k--")
    plt.text(65.0, yl[0], "Soma")
    plt.text(85.0, yl[0], "bDend")
    plt.ylabel("AP Peak Delay re AxonInit (µs)")
    plt.xlabel("Distance from AxonInit (µm)")
    #
    #
    fh2 = plt.figure(figsize=(5 / 2.54, 8 / 2.54))
    plt.plot(tmxa_min,-x,"b-")
    plt.plot(tmxb_min,-x,"r-")
    plt.plot(tmxm_min,-x,"k-")
    plt.xlim((-1, 10))
    plt.ylim((-60.5, 0.5))
    plt.xlabel("AP Peak Delay (ms)")
    plt.ylabel("Distance from AxonInit (µm)")
    #
    #plt.show()
    plt.tight_layout()
    #
    return fh1, fh2
    

if __name__ == "__main__":
    #Parse command-line arguments
    #SCNv2_imaging.py weload nconds thisdt
    inputargs = sys.argv[1:]
    myargs = [1, 11, 0.001]
    for iarg, thisarg in enumerate(inputargs):
        myargs[iarg] = float(thisarg)
    weload = bool(myargs[0])
    nconds = int(myargs[1])
    thisdt = float(myargs[2])
    #------------------
    #Run the show
    if os.path.isfile("./data/SCNv2_imaging_cache.npy") and weload:
        print("Cache for SCNv2_imaging found... loading!")
        output = np.load("./data/SCNv2_imaging_cache.npy", allow_pickle=True)
        output = output.tolist()
    else:
        output = runexp(nconds, thisdt)
        np.save("./data/SCNv2_imaging_cache.npy", output, allow_pickle=True)
    fh1, fh2 = makeplots(output)
    pp = PdfPages("./figures/SCNv2_imaging.pdf")
    pp.savefig(fh2)
    pp.savefig(fh1)
    pp.close()
    fh2.savefig("./figures/SCNv2_imaging_p2.svg")
    fh1.savefig("./figures/SCNv2_imaging_p1.svg")
