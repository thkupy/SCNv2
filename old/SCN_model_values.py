#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on tue mar 10 2020
This is the program that generates the 
"Lead Time diff" vs. "Active Synapse count diff" 2D plots for the SCN project
#----->however this is the corrected version
@author: kuenzel
"""
#
path_on_machine = '/home/kuenzel/Dokumente/'
cores_on_machine = 20
machine_has_display = True
dt = 0.025
#
#
from multiprocessing import Pool
import numpy as np
import time
import sys
import os
sys.path.append(path_on_machine + 'Python/tools')
sys.path.append(path_on_machine + 'Python/models')
sys.path.append(path_on_machine + 'Python/pxko')
import matplotlib
if not machine_has_display:
    matplotlib.use('Agg')
import tk_tools as tkt
import smallexc as se
import pandas as pd
import matplotlib.pyplot as plt
#se.load_mechanisms()
import itertools
import scipy.ndimage as ndimage
from matplotlib import ticker


tstop=1000
dt=0.025
a_on=True
b_on=True
seed=32768
inpn=(10,10)
starta=200.0
startb=200.0
indura=600.0
indurb=600.0
itv_a=20
itv_b=20
syntau=2
synw=0.002
'''
Creates and runs the model
''' 

#-----IMPORTS--------
from neuron import h
#from neuron import gui
#load_mechanisms()

pretime = 0

T = 21

#CREATE CONDUCTANCE TRACES
#E_temp = tkt.g_template('e',2,dt,0.15,0.2,geval)
#gvE=[]
#for gE in range(Nsyn):
#    thisg = tkt.make_g_trace()
#    gvE.append(h.Vector(thisg))

#--------MODEL DEFINITIONS---------
#CREATE SECTIONS
Soma = h.Section()
AxonInit = h.Section()
Internode1 = h.Section()
Internode2 = h.Section()
Node1 = h.Section()
Node2 = h.Section()
bDend = h.Section()
bproxDend = h.Section()
pNeurit = h.Section()
aDend = h.Section()#

#Geometry & Biophysics
bDend.L = 175#from tableq
bDend.diam = 2#assumed, averaged over length
bDend.nseg = 175
bDend_area = h.area(.5,sec=bDend)*bDend.nseg
#
bproxDend.L = 25#inferred
bproxDend.diam = 2#assumed
bproxDend.nseg = 25
bproxDend_area = h.area(.5,sec=bproxDend)*bproxDend.nseg
#
Soma.L = 20#from table
Soma.diam = 20#assumed'5d
Soma.nseg = 7
Soma_area = h.area(.5,sec=Soma)*Soma.nseg#in um^2
#
pNeurit.L = 45#from table
pNeurit.diam = 3#assumed
pNeurit.nseg = 7
pNeurit_area = h.area(.5,sec=pNeurit)*pNeurit.nseg#in um^2
#
AxonInit.L = 50#from table
AxonInit.diam = 3#assumed
AxonInit.nseg = 9
AxonInit_area = h.area(.5,sec=AxonInit)*AxonInit.nseg#in um^2
#
Internode1.L = 1000
Internode1.diam = 2
Internode1.nseg = 51
Internode2.L = 1000
Internode2.diam = 2
Internode2.nseg = 51
Node1.L = 3
Node1.diam = 2
Node1.nseg = 3
Node2.L = 3
Node2.diam = 2
Node2.nseg = 3
Internode_area = h.area(.5,sec=Internode1)*Internode1.nseg#in um^2
Node_area = h.area(.5,sec=Node1)*Node1.nseg#in um^2
#
aDend.L = 230#from table
aDend.diam = 2
aDend.nseg = 21
aDend_area = h.area(.5,sec=aDend)*aDend.nseg#in um^2
#
#axial Resistance and cm
for sec in h.allsec():
    sec.Ra = 150
    sec.insert('extracellular')
Internode1.cm = 0.01
Internode2.cm = 0.01
#
#hh
Soma.insert('hh')
Soma.ena = 50
Soma.gnabar_hh = 0.12
Soma.gkbar_hh = 0.023
AxonInit.insert('hh')
AxonInit.ena = 50
AxonInit.gnabar_hh = 0.12
AxonInit.gkbar_hh = 0.023
pNeurit.insert('hh')
pNeurit.ena = 50
pNeurit.gnabar_hh = 0.12
pNeurit.gkbar_hh = 0.023
#
Node1.insert('hh')
Node1.gnabar_hh = 0.36
Node2.insert('hh')
Node2.gnabar_hh = 0.36
Internode1.insert('leak')
Internode1.g_leak = 1e-6    
Internode2.insert('leak')
Internode2.g_leak = 1e-6 
#
#kht
Soma.insert('kht')
Soma.gkhtbar_kht = 0.013
Soma.ek = -80
pNeurit.insert('kht')
pNeurit.gkhtbar_kht = 0.013
pNeurit.ek = -80
AxonInit.insert('kht')
AxonInit.gkhtbar_kht = 0.013
AxonInit.ek = -80
bproxDend.insert('kht')
bproxDend.gkhtbar_kht = 0.013
bproxDend.ek = -80
#
#leak
aDend.insert('leak')
aDend.g_leak = 0.0001    
bDend.insert('leak')
bDend.g_leak = 0.0001    
bproxDend.insert('leak')
bproxDend.g_leak = 0.0001
#
#Topology
Node2.connect(Internode2(1))
Internode2.connect(Node1(1))
Node1.connect(Internode1(1))
Internode1.connect(AxonInit(1))
AxonInit.connect(pNeurit(1))
aDend.connect(pNeurit(1))
pNeurit.connect(Soma(1))
bproxDend.connect(Soma(0))
bDend.connect(bproxDend(1))

# GENERAL SETTINGS
h.dt = dt  # simulation (or "sampling") rate
h.celsius = T  # simulation global temperature

# MODEL INSTRUMENTATION & INPUT
Electrode = h.IClamp(Soma(0.5))
Electrode.delay = 10
Electrode.amp =0.1
Electrode.dur = 100
anc = []
asyn = []
astim = []
atv = []
for iasyn in range(inpn[0]):
    astim.append(h.NetStim())
    asyn.append( h.ExpSyn(aDend(0.75+(iasyn/100.0))) )
    atv.append(h.Vector())
    asyn[iasyn].tau = syntau
    asyn[iasyn].e = 0
    astim[iasyn].number = (indura/itv_a)
    astim[iasyn].start = starta
    astim[iasyn].noise = 1
    astim[iasyn].interval = itv_a
    astim[iasyn].seed(seed)
    anc.append(h.NetCon(astim[iasyn],asyn[iasyn]))
    anc[iasyn].record(atv[iasyn])
    anc[iasyn].delay=0.5
    if a_on:
        anc[iasyn].weight[0]=synw
    else:
        anc[iasyn].weight[0]=0

bstim=[]
bsyn=[]
bnc=[]
btv=[]
for ibsyn in range(inpn[1]):
    bstim.append(h.NetStim())
    bsyn.append( h.ExpSyn(bDend(0.75+(ibsyn/100.0))) )
    btv.append(h.Vector())
    bsyn[ibsyn].tau = syntau
    bsyn[ibsyn].e = 0
    bstim[ibsyn].number = (indurb/itv_b)
    bstim[ibsyn].start = startb
    bstim[ibsyn].noise = 1
    bstim[ibsyn].interval = itv_b
    bstim[ibsyn].seed(seed)
    bnc.append(h.NetCon(bstim[ibsyn],bsyn[ibsyn]))
    bnc[ibsyn].record(btv[ibsyn])
    bnc[ibsyn].delay=0.5
    if b_on:
        bnc[ibsyn].weight[0]=synw
    else:
        bnc[ibsyn].weight[0]=0
    
# INFRASTRUCTURE
SCN_Vm = h.Vector()
SCN_Vm.record(Soma(0.5)._ref_v)

#----------SIMULATE--------------
tkt.nrngo(tstop,-66.5)
#EXPORT DATA
Vm=np.array(SCN_Vm)



itv = []
for iii in range(10):
    tv=np.array(atv[iii])
    itv.append(np.mean(np.diff(tv)))
    tv2=np.array(btv[iii])
    itv.append(np.mean(np.diff(tv2)))
itv=np.array(itv)
print('mean ISI= ' + str(np.mean(itv)) + 'ms +/- ' + str(np.std(itv)) + 'ms')

plt.figure()
tax=np.linspace(0,tstop,(tstop/dt)+1)
plt.plot(tax,Vm)
plt.plot(109.9,Vm[int(round(109.9/dt))],'ro')
plt.plot(195.0,Vm[int(round(195.0/dt))],'ro')
Vdiff=Vm[int(round(109.9/dt))]-Vm[int(round(195.0/dt))]#in mV 1e-3
print('Rm=' + str(Vdiff/Electrode.amp) + 'MO')


#100um2 have 1pF capacitance, so the cell has...
Total_area = Soma_area+pNeurit_area+AxonInit_area+bproxDend_area+aDend_area+aDend_area
print('Cm=' + str(Total_area/100.0) + 'pF (not counting the axon)')
print('taum=' + str((Total_area/100.0)*(Vdiff/Electrode.amp)/1000.0) + 'ms')
#spit out conductances per segment in nS
#distributed conductances are in S/cm2. If you multiply this with with the area,
#you get conductance in S. Multiplication with 1e9 gives nS.
#from um2 to cm2 --> * 1e-8
#
#Soma
print('Soma na_hh ' + str( ((Soma_area*1e-8)*Soma.gnabar_hh)*1e9 ) +' nS')
print('Soma k_hh ' + str( ((Soma_area*1e-8)*Soma.gkbar_hh)*1e9 ) + 'nS')
print('Soma leak ' + str( ((Soma_area*1e-8)*Soma.gl_hh)*1e9 ) + 'nS')
print('Soma kht ' + str( ((Soma_area*1e-8)*Soma.gkhtbar_kht)*1e9 ) + 'nS')
#pNeurit
print('pNeurit na_hh ' + str( ((pNeurit_area*1e-8)*pNeurit.gnabar_hh)*1e9 ) +' nS')
print('pNeurit k_hh ' + str( ((pNeurit_area*1e-8)*pNeurit.gkbar_hh)*1e9 ) + 'nS')
print('pNeurit leak ' + str( ((pNeurit_area*1e-8)*pNeurit.gl_hh)*1e9 ) + 'nS')
print('pNeurit kht ' + str( ((pNeurit_area*1e-8)*pNeurit.gkhtbar_kht)*1e9 ) + 'nS')
#AxonInit
print('AxonInit na_hh ' + str( ((AxonInit_area*1e-8)*AxonInit.gnabar_hh)*1e9 ) +' nS')
print('AxonInit k_hh ' + str( ((AxonInit_area*1e-8)*AxonInit.gkbar_hh)*1e9 ) + 'nS')
print('AxonInit leak ' + str( ((AxonInit_area*1e-8)*AxonInit.gl_hh)*1e9 ) + 'nS')
print('AxonInit kht ' + str( ((AxonInit_area*1e-8)*AxonInit.gkhtbar_kht)*1e9 ) + 'nS')
#Internode
print('Internode leak ' + str( ((Internode_area*1e-8)*Internode1.g_leak)*1e9 ) + 'nS')
#Node
print('Node na_hh ' + str( ((AxonInit_area*1e-8)*AxonInit.gnabar_hh)*1e9 ) +' nS')
print('Node k_hh ' + str( ((AxonInit_area*1e-8)*AxonInit.gkbar_hh)*1e9 ) + 'nS')
print('Node leak ' + str( ((AxonInit_area*1e-8)*AxonInit.gl_hh)*1e9 ) + 'nS')
#bproxDend
print('bproxDend leak ' + str( ((bproxDend_area*1e-8)*bproxDend.g_leak)*1e9 ) + 'nS')
print('bproxDendt kht ' + str( ((bproxDend_area*1e-8)*bproxDend.gkhtbar_kht)*1e9 ) + 'nS')
#adistaldend
print('aDend leak ' + str( ((aDend_area*1e-8)*aDend.g_leak)*1e9 ) + 'nS')
#bdistaldend
print('bDend leak ' + str( ((bDend_area*1e-8)*bDend.g_leak)*1e9 ) + 'nS')


plt.show()