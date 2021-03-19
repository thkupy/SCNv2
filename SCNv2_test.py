#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised version of the SCN model (2021).

This is a test function. If this displays a plot all is set up correctly.

Created: 2021-03-17
Revised: 
@author: kuenzel(at)bio2.rwth-aachen.de
"""

import matplotlib.pyplot as plt
import numpy as np
import SCNv2
from pathlib import Path

def check_folders():
    pd = Path("data/")
    pf = Path("figures/")
    if pd.exists() and pd.is_dir():
        print("subfolder data ok")
    else:
        pd.mkdir(parents=True, exist_ok=True)
    if pf.exists() and pf.is_dir():
        print("subfolder figures ok")
    else:
        pf.mkdir(parents=True, exist_ok=True)

def make_exampleplot():
    tstop = 500.0
    print("Trying to simulate...")
    R = SCNv2.runmodel(
        haselectrode=True,
        pelectrode=(0,10.0, 50.0, 0.5, 0.0),
        hasinputactivity=(False, False),
        hasfbi=True,
    )
    print("Simulation worked ok")
    T = np.linspace(0.0,500.0-0.025, int(round(500.0/0.025)))
    fh = plt.figure()
    sp = fh.add_subplot(111)
    sp.plot(T, R["AVm"],"k-")
    sp.set_title("Apical only")
    sp.set_ylabel("Membrane Potential (mV)")
    sp.set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_folders()
    make_exampleplot()
