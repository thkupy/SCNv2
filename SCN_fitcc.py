#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New version of the SCN model, including much longer synaptic time-constants and more
realistic membrane time-constants.
This is mostly a demo program.

Created: Tuesday 23rd Feb 2021
@author: kuenzel
"""
#
import matplotlib.pyplot as plt
import numpy as np
import SCNv2

def runmain():
    weload = True
    if weload:
        D = np.load("./fitres.npz")
        g1s = D["g1s"]
        g2s = D["g2s"]
        nAP = D["nAP"]
        nSpo = D["nSpo"]
        aAP1 = D["aAP1"]
    else:
        #run models
        tstop = 250.0
        dt = 0.025
        n1 = 15
        n2 = 16
        T = np.linspace(0,tstop - dt, int(round(tstop / dt)))
        g1s = np.linspace(0.001,0.05, n1)
        g2s = np.linspace(0.08, 0.2, n2)
        nAP=np.zeros((n2,n1))*np.nan
        aAP1=np.zeros((n2,n1))*np.nan
        nSpo=np.zeros((n2,n1))*np.nan
        for i1,s1 in enumerate(g1s):
            plt.figure()
            for i2,s2 in enumerate(g2s):
                R = SCNv2.runmodel(
                        tstop=tstop,
                        hasstimulation=(False, False),
                        hasinputactivity=(False, False),
                        haselectrode=True,
                        pelectrode=(0, 50.0,100.00, 1.2, 0.0),
                        hasfbi=False,
                        soma_na=s2,
                        soma_k=s1,
                        soma_kht=0.013,
                        soma_l=0.0001,
                        dend_l=0.00001,
                    )
                E = SCNv2.SimpleDetectAP(R['SVm'], dt=dt, thr=-50, LM=-30)
                plt.plot(T, R["SVm"], "k")
                plt.plot(E["PeakT"], E["PeakV"], "gx")
                if len(E["PeakT"]) > 0:
                    allAPT = np.array(E["PeakT"])
                    allAPV = np.array(E["PeakV"])
                    dAPT = allAPT[np.logical_and(allAPT<149.0,allAPT>50.0)]
                    dAPV = allAPV[np.logical_and(allAPT<149.0,allAPT>50.0)]
                    sAPT = allAPT[np.logical_or(allAPT>160.0,allAPT<49.0)]
                    sAPV = allAPV[np.logical_or(allAPT>160.0,allAPT<49.0)]
                    nAP[i2,i1] = np.size(dAPT)
                    if np.size(dAPT) > 0:
                        aAP1[i2,i1] = dAPV[0]-np.mean(R["SVm"][int(10.0/dt):int(49.0/dt)])
                    nSpo[i2,i1] = np.size(sAPT)
        np.savez(
            "./fitres.npz",
            g1s=g1s,
            g2s=g2s,
            nAP=nAP,
            nSpo=nSpo,
            aAP1=aAP1,
        )
    plt.figure()
    plt.subplot(131)
    cs = plt.contourf(
        g2s,
        g1s,
        np.flipud(np.rot90(nAP)),
        levels=(1,2,5,10,20,50,100),
        cmap="bone",
        extend="both",
    )
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.colorbar()
    plt.xlabel("gnabar_hh")
    plt.ylabel("gkbar_hh")
    plt.title("N driven Spikes @ 1200pA")
    plt.subplot(132)
    cs = plt.contourf(
        g2s,
        g1s,
        np.flipud(np.rot90(nSpo)),
        levels=(1,2,5,10),
        cmap="bone",
        extend="both",
    )
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.colorbar()
    plt.xlabel("gnabar_hh")
    plt.ylabel("gkbar_hh")
    plt.title("N spont Spikes @ 1200pA")
    plt.subplot(133)
    plt.contourf(g2s,g1s, np.flipud(np.rot90(aAP1)), cmap="bone")
    plt.colorbar()
    plt.ylabel("gnabar_hh")
    plt.xlabel("gkbar_hh")
    plt.title("AP1 V (mV)")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    runmain()
