# -*- coding: utf-8 -*-
# Filename: scn_play.py
# indentation: 4 spaces
# run this as a function within ipython
#
#Beware: this is made to run in a Linux environment

import sys
import os
sys.path.append('/home/kuenzel/Documents/Python/tools')
sys.path.append('/home/kuenzel/Documents/Python/models')
sys.path.append('/home/kuenzel/Documents/Python/pxko')
import matplotlib.pyplot as plt
import numpy as np
import tk_tools as tkt
from datetime import datetime
from scipy import stats

def smoothy(x,windowlen=9,windowtype='hamming'):
    if not windowtype in ['flat','hanning','hamming','bartlett','blackman']:
        print 'windowtype must be flat,hanning,hamming,bartlett or blackman' 
        raise ValueError, "windowtype"
    #
    s=np.r_[x[windowlen-1:0:-1],x,x[-2:-windowlen-1:-1]]
    if windowtype=='flat':
        w=np.ones(windowlen,'d')
    else:
        w=eval('np.'+windowtype+'(windowlen)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    Ncut=(len(y)-len(x))/2
    y=y[Ncut-1:-Ncut-1]
    return(y)
    
def load_mechanisms():
    import Neuron_Models as nm
    nm.h('nrn_load_dll("/home/kuenzel/Documents/Python/SCN/mech/x86_64/.libs/libnrnmech.so")')

def runmodel(tstop=1000,dt=0.025,a_on=True, b_on=True,seed=32768,
             inpn=(10,10),starta=0.0,startb=0.0,syntau=2,synw=0.002,
             hasrandomness=True):
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
    #
    bproxDend.L = 25#inferred
    bproxDend.diam = 2#assumed
    bproxDend.nseg = 25
    #
    Soma.L = 20#from table
    Soma.diam = 20#assumed'5d
    Soma.nseg = 7
    #
    pNeurit.L = 45#from table
    pNeurit.diam = 3#assumed
    pNeurit.nseg = 7
    #
    AxonInit.L = 50#from table
    AxonInit.diam = 3#assumed
    AxonInit.nseg = 9
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
    #
    aDend.L = 230#from table
    aDend.diam = 2
    aDend.nseg = 21
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
    #Electrode = h.IClamp(Soma(0.5))
    #Electrode.delay = 10
    #Electrode.amp =0.2
    #Electrode.dur = 50
    meaninterval=20
    anc = []
    asyn = []
    astim = []
    atv = []
    #if a_on:
    #    if b_on:
    #        Na = 10
    #    else:
    #        Na = 20
    #else:
    #    Na = 1
    if hasrandomness:
        mynoise = 1.0
        mydelay = 0.5
        mynumber = (tstop/meaninterval)+20
    else:
        mynoise = 0.0001
        mydelay = 0.5
        mynumber = 1
    for iasyn in range(inpn[0]):
        astim.append(h.NetStim())
        asyn.append( h.ExpSyn(aDend(0.75+(iasyn/100.0))) )
        atv.append(h.Vector())
        asyn[iasyn].tau = syntau
        asyn[iasyn].e = 0
        astim[iasyn].number = mynumber
        astim[iasyn].start = starta
        astim[iasyn].noise = mynoise
        astim[iasyn].interval = meaninterval
        astim[iasyn].seed(seed)
        anc.append(h.NetCon(astim[iasyn],asyn[iasyn]))
        anc[iasyn].record(atv[iasyn])
        anc[iasyn].delay=mydelay
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
        bstim[ibsyn].number = mynumber
        bstim[ibsyn].start = startb
        bstim[ibsyn].noise = mynoise
        bstim[ibsyn].interval = meaninterval
        bstim[ibsyn].seed(seed)
        bnc.append(h.NetCon(bstim[ibsyn],bsyn[ibsyn]))
        bnc[ibsyn].record(btv[ibsyn])
        bnc[ibsyn].delay=mydelay
        
        if b_on:
            bnc[ibsyn].weight[0]=synw
        else:
            bnc[ibsyn].weight[0]=0
    print 'Na=' + str(inpn[0]) + ' + Nb=' + str(inpn[1])#debug
        
    # INFRASTRUCTURE
    SCN_SVm = h.Vector()
    SCN_SVm.record(Soma(0.5)._ref_v)
    SCN_AVm = h.Vector()
    SCN_AVm.record(Node2(0.99)._ref_v)


    #----------SIMULATE--------------
    tkt.nrngo(tstop+pretime,-66.5)
    # PACK AND EXPORT DATA
    Result = {}
    tempres_avm = np.array(SCN_AVm)
    tempres_svm = np.array(SCN_SVm)
    Result['AVm'] = tempres_avm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    Result['SVm'] = tempres_svm[int(pretime/dt):int((tstop+pretime)/dt)] #remove "additional" samples
    Result['atv'] = atv#packaged as hoc objects... need to unpack later on
    Result['btv'] = btv#packaged as hoc objects... need to unpack later on
    return(Result)   

def autocorr(D,tstop=1000.0,dt=0.01):
    '''
    Calculate mean autocorrelation of repetitions of spiketrains
    Based on histogram of time-differences
    '''
    count=0
    NRep=len(D)
    R=[]
    for irep,thisD in enumerate(D):
        NSp=len(thisD['PeakT'])
        R.append(NSp/(tstop/1000.0))
        for isp,spt in enumerate(thisD['PeakT']):
            if count == 0:
                Dist = np.array(thisD['PeakT']-spt)
            else:
                Dist = np.concatenate((Dist,thisD['PeakT']-spt))
            count = count +1
    Dist=np.sort(Dist)
    MR=np.mean(R)
    H=np.histogram(Dist,tstop*2)
    Res=[]
    Res.append(H[1][1:])
    Res.append(H[0]/MR*NRep)
    return(Res)

def autocorr2(D,tstop=1000.0,dt=0.01):
    '''
    Calculate mean autocorrelation of repetitions of spiketrains
    Based on correlation of convolved time-domain signals
    '''
    count=0
    NRep=len(D)
    ac=[]
    for irep,thisD in enumerate(D):
        g=gauss_pdf(200)
        x=np.zeros((int(tstop/dt),1))
        sp=np.array(thisD['PeakT'])
        I=np.round(sp/dt)
        I=np.array(I,dtype=int)
        x[I]=np.ones((len(I),1))
        x=x.flatten()
        tdx=np.convolve(x,g,'valid')
        ac.append(np.correlate(tdx,tdx,'full')/(len(sp)*4))
    Res = []
    Res.append(np.linspace(-(tstop-dt),tstop,len(ac[0])))
    Res.append(np.mean(ac,axis=0))
    return(Res)

def xcorr(Out,In,tstop=1000.0,dt=0.01):
    '''
    Calculate mean crosscorrelation of repetitions of input and output spiketrains 
    '''
    count=0
    NRep=len(Out)
    R=[]
    for irep,thisO in enumerate(Out):
        NSp=len(thisO['PeakT'])
        R.append(NSp/(tstop/1000.0))
        for isp,spt in enumerate(thisO['PeakT']):
            if count == 0:
                Dist = np.array(In[irep]-spt)
            else:
                Dist = np.concatenate((Dist,In[irep]-spt))
            count = count +1
    Dist=np.sort(Dist)
    MR=np.mean(R)
    H=np.histogram(Dist,tstop*2)
    Res=[]
    Res.append(H[1][1:])
    Res.append(H[0]/MR*NRep)
    return(Res)

def xcorr2(Out,In,tstop=1000.0,dt=0.01):
    '''
    Calculate mean crosscorrelation of repetitions of spiketrains
    Based on correlation of convolved time-domain signals
    '''
    count=0
    NRep=len(Out)
    xc=[]
    for irep,thisO in enumerate(Out):
        g=gauss_pdf(200)
        xo=np.zeros((int(tstop/dt),1))
        xi=np.zeros((int(tstop/dt),1))
        spo=np.array(thisO['PeakT'])
        spi=np.array(In[irep])
        Io=np.round(spo/dt)
        Io=np.array(Io,dtype=int)
        Ii=np.round(spi/dt)
        Ii=np.array(Ii,dtype=int)
        xo[Io]=np.ones((len(Io),1))
        xo=xo.flatten()
        xi[Ii]=np.ones((len(Ii),1))
        xi=xi.flatten()
        tdxo=np.convolve(xo,g,'valid')
        tdxi=np.convolve(xi,g,'valid')
        xc.append(np.correlate(tdxo,tdxi,'full')/(len(spo)*4))
    Res = []
    Res.append(np.linspace(-(tstop-dt),tstop,len(xc[0])))
    Res.append(np.mean(xc,axis=0))
    return(Res)

def gauss_pdf(N=1000,sigma=0.1,mu=0.0):
    x = np.linspace(-sigma*5,sigma*5,N)
    return(1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2 / (2*sigma**2)))

if __name__ == '__main__':
    plt.rc('font',**{'size': 8})
    plt.rc('axes',**{'labelweight': 'bold', 'labelsize': 'medium'})
    plt.rc('xtick',**{'labelsize': 'medium'})
    plt.rc('ytick',**{'labelsize': 'medium'})
    plt.rcParams['font.family'] = 'sans' 
    plt.rcParams['pdf.fonttype'] = 42
    if int(sys.argv[2])==1:
        docalc = True
    else:
        docalc = False
    if int(sys.argv[1])==1:
        #1-----------SHOW NICE PLOTS-----------------------------------------------
        tstop=1000
        dt=0.01
        R=[]
        R.append(runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,inpn=(10,10),
                    syntau=2,synw=0.002,seed=32768))#syntau=2,synw=0.002
        R.append(runmodel(tstop=tstop,dt=dt,a_on=True,b_on=False,inpn=(20,0),
                    syntau=2,synw=0.002,seed=32769))#syntau=2,synw=0.002
        R.append(runmodel(tstop=tstop,dt=dt,a_on=False,b_on=True,inpn=(0,20),
                    syntau=2,synw=0.002,seed=32770))#syntau=2,synw=0.002
        taxis = np.linspace(0,tstop-dt,tstop/dt)
        titles = ('10 apical + 10 basal inputs --> ','20 apical inputs --> ','20 basal inputs -->')
        #
        for isub in range(3):
            plt.subplot(3,1,isub+1)
            plt.plot(taxis,R[isub]['AVm'],'r',label='Axon_Vm')
            ASp = tkt.SimpleDetectAP(R[isub]['AVm'],dt=dt,thr=-50,LM=-30)
            plt.plot(ASp['PeakT'],ASp['PeakV'],'m^',label='Axon_AP')
            plt.plot(taxis,R[isub]['SVm'],'k',label='Soma_Vm')
            SSp = tkt.SimpleDetectAP(R[isub]['SVm'],dt=dt,thr=-30,LM=-30)
            plt.plot(SSp['PeakT'],SSp['PeakV'],'gv',label='Soma_AP')
            plt.ylabel('Vm (mV)')
            plt.title(titles[isub] + str(len(ASp['PeakT'])) + 'Hz')
            if isub==0:
                plt.legend()
            if isub==2:
                plt.xlabel('Time (ms)')
        plt.show()
        np.save('/home/kuenzel/Documents/Python/SCN/Example_Traces.npy',R)
    
    #2-----------Quantifications-----------------------------------------------
    elif int(sys.argv[1])==2:
        tstop=1000
        dt=0.01
        reps = 100
        seeds = (np.arange(reps)*3)+32768
        if docalc:#calculate RES
            RM=[]
            SRM=[]
            CVM=[]
            RA=[]
            SRA=[]
            CVA=[]
            RB=[]
            SRB=[]
            CVB=[]
            #----CALC REPETITIONS----
            for irep in range(reps):
                VM=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,seed=seeds[irep])
                RM.append(tkt.SimpleDetectAP(VM['AVm'],dt=dt,thr=-50,LM=-30))
                SRM.append(len(RM[irep]['PeakT'])*(1000.0/tstop))
                ISI=[]
                ISI = np.diff(np.array(RM[irep]['PeakT']))
                CVM.append( np.std(ISI) / np.mean(ISI) )
                #
                VA=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=False,seed=seeds[irep])
                RA.append(tkt.SimpleDetectAP(VA['AVm'],dt=dt,thr=-50,LM=-30))
                SRA.append(len(RA[irep]['PeakT'])*(1000.0/tstop))
                ISI=[]
                ISI = np.diff(np.array(RA[irep]['PeakT']))
                CVA.append( np.std(ISI) / np.mean(ISI) )
                #
                VB=runmodel(tstop=tstop,dt=dt,a_on=False,b_on=True,seed=seeds[irep])
                RB.append(tkt.SimpleDetectAP(VB['AVm'],dt=dt,thr=-50,LM=-30))
                SRB.append(len(RB[irep]['PeakT'])*(1000.0/tstop))
                ISI=[]
                ISI = np.diff(np.array(RB[irep]['PeakT']))
                CVB.append( np.std(ISI) / np.mean(ISI) )
            #----SAVE DATA----
            RES={}
            RES['date'] = str(datetime.now())
            RES['tstop']= tstop
            RES['dt'] = dt
            RES['reps']= reps
            RES['RM']=RM
            RES['SRM']=SRM
            RES['CVM']=CVM
            RES['RA']=RA
            RES['SRA']=SRA
            RES['CVA']=CVA
            RES['RB']=RB
            RES['SRB']=SRB
            RES['CVB']=CVB
            np.save('/home/kuenzel/Documents/SCN/scnqres.npy',RES)
        else:#load RES
            try:
                RES = np.load('/home/kuenzel/Documents/SCN/scnqres.npy')
                RES = RES.tolist()
            except:
                print "Data not found yet, use 1 as second argument!"
        #now data in RES is either freshly calculated or just loaded from disk
        #-----CREATE FIGURES-----
        #1-rasterplots
        plt.figure(figsize=(30.0/2.54,15.0/2.54))
        plt.title('SCN output spiketrains')
        MISI=np.array(())
        MSP=np.array(())
        AISI=np.array(())
        ASP=np.array(())
        BISI=np.array(())
        BSP=np.array(())
        sp1=plt.subplot(3,5,(1,3))
        plt.xticks(())
        plt.ylabel('Rep. #')
        plt.locator_params(nbins=5,axis='xy')
        sp2=plt.subplot(3,5,(6,8))
        plt.xticks(())
        plt.ylabel('Rep. #')
        plt.locator_params(nbins=5,axis='xy')
        sp3=plt.subplot(3,5,(11,13))
        plt.xlabel('Time (ms)')
        plt.ylabel('Rep. #')
        plt.locator_params(nbins=5,axis='xy')
        for irep in range(RES['reps']):
            ##1
            plt.sca(sp1)
            thisspt=[]
            thisspt = RES['RM'][irep]['PeakT']
            MSP=np.concatenate((MSP,thisspt))
            MISI=np.concatenate((MISI,np.diff(thisspt)))
            ycoords = []
            ycoords = np.ones(len(thisspt))+irep
            plt.plot(thisspt,ycoords,'k|')
            ##2
            plt.sca(sp2)
            thisspt=[]
            thisspt = RES['RA'][irep]['PeakT']
            ASP=np.concatenate((ASP,thisspt))
            AISI=np.concatenate((AISI,np.diff(thisspt)))
            ycoords = []
            ycoords = np.ones(len(thisspt))+irep
            plt.plot(thisspt,ycoords,'k|')
            ##3
            plt.sca(sp3)
            thisspt=[]
            thisspt = RES['RB'][irep]['PeakT']
            BSP=np.concatenate((BSP,thisspt))
            BISI=np.concatenate((BISI,np.diff(thisspt)))
            ycoords = []
            ycoords = np.ones(len(thisspt))+irep
            plt.plot(thisspt,ycoords,'k|')
        MISI=np.array(MISI)
        AISI=np.array(AISI)
        BISI=np.array(BISI)
        plt.subplot(3,5,4)
        tbins=np.linspace(0,1000,100)
        MPSTH=np.histogram(MSP,tbins)
        plt.plot(MPSTH[1][0:-1],MPSTH[0],'k')
        plt.ylabel('AP/bin')
        plt.ylim((0,110))
        plt.gca().locator_params(nbins=3)
        #
        plt.subplot(3,5,5)
        bins=np.linspace(0.1,100.0,100)
        MH=np.histogram(MISI,bins=bins)
        plt.plot(MH[1][0:-1],np.array(MH[0],dtype=float)/np.max(MH[0]),'k')
        plt.ylabel('Rel. Occurrence')
        plt.gca().locator_params(nbins=3)
        #
        plt.subplot(3,5,9)
        APSTH=np.histogram(ASP,tbins)
        plt.plot(APSTH[1][0:-1],APSTH[0],'r')
        plt.ylabel('AP/bin')
        plt.ylim((0,110))
        plt.gca().locator_params(nbins=3)
        #
        plt.subplot(3,5,10)
        AH=np.histogram(AISI,bins=bins)
        plt.plot(AH[1][0:-1],np.array(AH[0],dtype=float)/np.max(AH[0]),'r')
        plt.ylabel('Rel. Occurrence')
        plt.gca().locator_params(nbins=3)
        #
        plt.subplot(3,5,14)
        BPSTH=np.histogram(BSP,tbins)
        plt.plot(BPSTH[1][0:-1],BPSTH[0],'g')
        plt.ylabel('AP/bin')
        plt.ylim((0,110))
        plt.xlabel('Time, binned (ms)')
        plt.gca().locator_params(nbins=3)
        #
        plt.subplot(3,5,15)
        BH=np.histogram(BISI,bins=bins)
        plt.plot(BH[1][0:-1],np.array(BH[0],dtype=float)/np.max(BH[0]),'g')
        plt.ylabel('Rel. Occurrence')
        plt.xlabel('ISI (ms)')
        plt.gca().locator_params(nbins=3)
        #
        plt.tight_layout()
        #
        plt.figure(figsize=(12.0/2.54,12.0/2.54))
        plt.plot(bins[0:-1],np.cumsum(MH[0],dtype=float)/np.sum(MH[0],dtype=float),'k',label='apical+basal')
        plt.plot(bins[0:-1],np.cumsum(AH[0],dtype=float)/np.sum(AH[0],dtype=float),'r',label='apical only')
        plt.plot(bins[0:-1],np.cumsum(BH[0],dtype=float)/np.sum(BH[0],dtype=float),'g',label='basal only')
        plt.title('Cumulative Histogram of ISI')
        plt.xlabel('ISI (ms)')
        plt.ylabel('Cumulative Occurence')
        plt.legend(loc='lower right')
        #
        #
        #2-rate comparison + statistics
        plt.figure(figsize=(12.0/2.54,12.0/2.54))
        for iii in range(len(RES['SRM'])):
            plt.plot(1+np.random.normal()/20,RES['SRM'][iii],'o',color=(.8,.8,.8))
        plt.errorbar(1,np.mean(RES['SRM']),yerr=np.std(RES['SRM']),fmt='ko',ecolor='k',ms=12,label='A+B')
        for iii in range(len(RES['SRA'])):
            plt.plot(2+np.random.normal()/20,RES['SRA'][iii],'o',color=(.8,.8,.8))
        plt.errorbar(2,np.mean(RES['SRA']),yerr=np.std(RES['SRA']),fmt='ro',ecolor='r',ms=12,label='A')
        for iii in range(len(RES['SRB'])):
            plt.plot(3+np.random.normal()/20,RES['SRB'][iii],'o',color=(.8,.8,.8))
        plt.errorbar(3,np.mean(RES['SRB']),yerr=np.std(RES['SRB']),fmt='go',ecolor='g',ms=12,label='B')
        plt.ylabel('AP Rate (Hz)')
        plt.xticks((1,2,3),('apical+basal','apical only','basal only'))
        #
        rstat=stats.kruskal(RES['SRM'],RES['SRA'],RES['SRB'])
        ab=stats.mannwhitneyu(RES['SRM'],RES['SRA'])
        ac=stats.mannwhitneyu(RES['SRM'],RES['SRB'])
        bc=stats.mannwhitneyu(RES['SRA'],RES['SRB'])
        print 'KW-Test: ' + str(rstat[0]) + '/' + str(rstat[1])
        print '1vs2: ' + str(ab[0]) + '/' + str(ab[1])
        print '1vs3: ' + str(ac[0]) + '/' + str(ac[1])
        print '2vs3: ' + str(bc[0]) + '/' + str(bc[1])
        plt.title('Mean Response Rate')
                
        #3-cv comparison + statistics
        plt.figure(figsize=(12.0/2.54,12.0/2.54))
        for iii in range(len(RES['CVM'])):
            plt.plot(1+np.random.normal()/20,RES['CVM'][iii],'o',color=(.8,.8,.8))
        plt.errorbar(1,np.mean(RES['CVM']),yerr=np.std(RES['CVM']),fmt='ko',ecolor='k',ms=12)
        for iii in range(len(RES['CVA'])):
            plt.plot(2+np.random.normal()/20,RES['CVA'][iii],'o',color=(.8,.8,.8))
        plt.errorbar(2,np.mean(RES['CVA']),yerr=np.std(RES['CVA']),fmt='ro',ecolor='r',ms=12)
        for iii in range(len(RES['CVB'])):
            plt.plot(3+np.random.normal()/20,RES['CVB'][iii],'o',color=(.8,.8,.8))
        plt.errorbar(3,np.mean(RES['CVB']),yerr=np.std(RES['CVB']),fmt='go',ecolor='g',ms=12)
        plt.ylabel('Coefficient of Variation')
        plt.xticks((1,2,3),('apical+basal','apical','basal'))
        #
        rstat=stats.kruskal(RES['CVM'],RES['CVA'],RES['CVB'])
        ab=stats.mannwhitneyu(RES['CVM'],RES['CVA'])
        ac=stats.mannwhitneyu(RES['CVM'],RES['CVB'])
        bc=stats.mannwhitneyu(RES['CVA'],RES['CVB'])
        print 'KW-Test: ' + str(rstat[0]) + '/' + str(rstat[1])
        print '1vs2: ' + str(ab[0]) + '/' + str(ab[1])
        print '1vs3: ' + str(ac[0]) + '/' + str(ac[1])
        print '2vs3: ' + str(bc[0]) + '/' + str(bc[1])
        plt.title('Mean ISI Coefficient of Variation')
        #
        plt.show()
    
    #3-----------Synchrony metric experiment 1-----------------------------------------------
    elif int(sys.argv[1])==3:
        tstop=1000
        dt=0.01
        a_on=True
        b_on=True
        rep = 10
        #R=[]
        #Out=[]
        R=runmodel(tstop=tstop,dt=dt,a_on=a_on,b_on=b_on,inpn=(10,10))
        Out=tkt.SimpleDetectAP(R['AVm'],dt=dt,thr=-50,LM=-30)
        InA = []
        InB = []
        for Ai,Aisp in enumerate(R['atv']):
            InA.append(np.array(Aisp))
        for Bi,Bisp in enumerate(R['btv']):
            InB.append(np.array(Bisp))
        #compute average correlation between InA[0..n] and Out? Ist that lower/higher than 
        #between InB and Out? Maybe even quantify "connectedness" through this?
        dim1=len(Out['PeakT'])
        if a_on:
            plt.figure()
            AllCA=[]
            HA=np.zeros((tstop*2,len(InA)))
            for iInA in range(len(InA)):
                dim2=len(InA[iInA])
                CC=np.zeros((dim2,dim1))
                for isp in range(dim1):
                    CC[:,isp]=InA[iInA]-Out['PeakT'][isp]
                AllCA.append(CC.flatten())
                HTA=np.histogram(AllCA[iInA],tstop*2)
                HA[:,iInA]=HTA[0]
                plt.plot(HTA[1][1:],HTA[0])
            plt.xlim((-10,10))
        
        if b_on:
            plt.figure()
            AllCB=[]
            HB=np.zeros((tstop*2,len(InB)))
            for iInB in range(len(InB)):
                dim2=len(InB[iInB])
                CC=np.zeros((dim2,dim1))
                for isp in range(dim1):
                    CC[:,isp]=InB[iInB]-Out['PeakT'][isp]
                AllCB.append(CC.flatten())
                HTB=np.histogram(AllCB[iInB],tstop*2)
                HB[:,iInB]=HTB[0]
                plt.plot(HTB[1][1:],HTB[0])
            plt.xlim((-10,10))
        
        plt.figure()
        if a_on:
            plt.plot(HTA[1][1:],np.mean(HA,axis=1),'k')
        if b_on:
            plt.plot(HTB[1][1:],np.mean(HB,axis=1),'r')
        plt.xlim((-10,10))
        plt.show()
    
    #4-----------Synchrony metric experiment 2-----------------------------------------------
    elif int(sys.argv[1])==4:
        if int(sys.argv[2])==1:
            tstop=1000.0
            dt=0.025
            a_on=True
            b_on=True
            rep = 25
            inpn=(10,10)
            R=[]
            Out=[]
            InA = []
            InB = []
            for irep in range(rep):
                R.append(runmodel(tstop=tstop,dt=dt,a_on=a_on,b_on=b_on,inpn=inpn))
                Out.append(tkt.SimpleDetectAP(R[irep]['AVm'],dt=dt,thr=-50,LM=-30))
                InA.append(np.array(R[irep]['atv'][2]))
                InB.append(np.array(R[irep]['btv'][2]))
            s={}
            s['tstop']=tstop
            s['dt']=dt
            s['rep']=rep
            s['a_on']=a_on
            s['b_on']=b_on
            s['inpn']=inpn
            s['Out']=Out
            s['InA']=InA
            s['InB']=InB
            np.save('/home/kuenzel/Documents/SCN/scnqres_Corr.npy',s)
        else:
            try:
                s = np.load('/home/kuenzel/Documents/SCN/scnqres_Corr.npy')
                s = s.tolist()
                tstop=s['tstop']
                dt=s['dt']
                rep=s['rep']
                a_on=s['a_on']
                b_on=s['b_on']
                inpn=s['inpn']
                Out=s['Out']
                InA=s['InA']
                InB=s['InB']
            except:
                sys.exit('Data not found yet, use 1 as second argument!')
        AC=autocorr2(Out,tstop,dt)
        XCA=xcorr2(Out,InA,tstop,dt)
        XCB=xcorr2(Out,InB,tstop,dt)
        plt.figure()
        plt.suptitle('Input to Output Cross-Correlation',fontsize=15)
        plt.subplot(1,2,1)
        plt.plot(AC[0],AC[1]/np.max(AC[1]),'k')
        plt.xlabel('Time (ms)')
        plt.ylabel('Autocorrelation')
        plt.xlim((-10,10))
        plt.subplot(1,2,2)
        plt.plot(XCA[0],XCA[1]/np.max(AC[1]),'g',label='Apical #2')
        plt.plot(XCB[0],XCB[1]/np.max(AC[1]),'r',label='Basal #2')
        plt.xlabel('Time (ms)')
        plt.ylabel('Crosscorrelation')
        plt.legend()
        plt.xlim((-1,10))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
    
    #5-----------onset differences-----------------------------------------------
    elif int(sys.argv[1])==5:
        stardiff=100.0
        if int(sys.argv[2])==1:
            R=[]
            tstop=400
            dt=0.01
            rep = 10#1000
            #
            EE=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,starta=200.0,startb=200.0)
            R.append(EE['AVm'])
            BE=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,starta=200.0,startb=200.0-stardiff)
            R.append(BE['AVm'])
            AE=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,starta=200.0-stardiff,startb=200.0)
            R.append(AE['AVm'])
            #
            spE=[]
            spA=[]
            spB=[]
            for irep in range(rep):
                tRE=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,starta=200.0,startb=200.0,seed=32768+irep)
                tspE=tkt.SimpleDetectAP(tRE['AVm'],dt=dt,thr=-50,LM=-30)
                spE.append(tspE['PeakT'])
                tRB=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,starta=200.0,startb=100.0,seed=32768+irep)
                tspB=tkt.SimpleDetectAP(tRB['AVm'],dt=dt,thr=-50,LM=-30)
                spB.append(tspB['PeakT'])
                tRA=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,starta=100.0,startb=200.0,seed=32768+irep)
                tspA=tkt.SimpleDetectAP(tRA['AVm'],dt=dt,thr=-50,LM=-30)
                spA.append(tspA['PeakT'])
            spE=np.array(spE).flatten()
            spA=np.array(spA).flatten()
            spB=np.array(spB).flatten()
            #
            s={}
            s['tstop']=tstop
            s['dt']=dt
            s['rep']=rep
            s['R']=R
            s['stardiff']=stardiff
            s['spE']=spE
            s['spA']=spA
            s['spB']=spB
            np.save('/home/kuenzel/Documents/SCN/scnqres2.npy',s)
        else:#load RES
            try:
                s = np.load('/home/kuenzel/Documents/SCN/scnqres2.npy')
                s = s.tolist()
                tstop=s['tstop']
                dt=s['dt']
                rep=s['rep']
                R=s['R']
                spE=s['spE']
                spA=s['spA']
                spB=s['spB']
            except:
                print sys.exit('Data not found yet, use 1 as second argument!')
        spE=np.hstack(spE)
        spA=np.hstack(spA)
        spB=np.hstack(spB)
        #now data is either freshly calculated or just loaded from disk
        taxis = np.linspace(0,tstop-dt,tstop/dt)
        titles = ('identical start= ','basal lead= ','apical lead= ')
        #
        plt.rc('font',**{'size': 8})
        plt.rc('axes',**{'labelweight': 'bold', 'labelsize': 'medium'})
        plt.rc('xtick',**{'labelsize': 'medium'})
        plt.rc('ytick',**{'labelsize': 'medium'})
        plt.rcParams['font.family'] = 'sans' 
        plt.rcParams['pdf.fonttype'] = 42
        #
        for isub in range(3):
            plt.subplot(3,1,isub+1)
            plt.plot(taxis,R[isub],'r',label='Axon_Vm')
            ASp = tkt.SimpleDetectAP(R[isub],dt=dt,thr=-50,LM=-30)
            plt.plot(ASp['PeakT'],ASp['PeakV'],'m^',label='Axon_AP')
            #plt.plot(taxis,R[isub]['SVm'],'k',label='Soma_Vm')
            #SSp = tkt.SimpleDetectAP(R[isub]['SVm'],dt=dt,thr=-30,LM=-30)
            #plt.plot(SSp['PeakT'],SSp['PeakV'],'gv',label='Soma_AP')
            plt.ylabel('Vm (mV)')
            plt.title(titles[isub] + str(len(ASp['PeakT'])) + 'Hz')
            if isub == 0:
                plt.plot((200.0,200.0),(-80.0,40.0),'g')
            else:
                plt.plot((100.0,100.0),(-80.0,40.0),'g:')
                plt.plot((200.0,200.0),(-80.0,40.0),'g')
            if isub==0:
                plt.legend()
            if isub==2:
                plt.xlabel('Time (ms)')
        plt.tight_layout()
        nbinspsth=400
        HE=np.histogram(spE,np.linspace(0,tstop,nbinspsth))
        HA=np.histogram(spA,np.linspace(0,tstop,nbinspsth))
        HB=np.histogram(spB,np.linspace(0,tstop,nbinspsth))
        plt.figure()
        plt.suptitle('PSTH delayed onsets, ' + str(stardiff) + 'ms differences',fontsize=16)
        plt.subplot(121)
        plt.plot(HE[1][1:],HE[0],'b',label='No Delay')
        #plt.plot(HE[1][1:],smoothy(HE[0],windowlen=3,windowtype='hamming'),'b',label='No Delay')
        plt.plot(HA[1][1:],HA[0],'g',label='Apical lead')
        #plt.plot(HA[1][1:],smoothy(HA[0],windowlen=3,windowtype='hamming'),'g',label='Apical lead')
        plt.plot(HB[1][1:],HB[0],'r',label='Basal lead')
        #plt.plot(HB[1][1:],smoothy(HB[0],windowlen=3,windowtype='hamming'),'r',label='Basal lead')
        plt.legend()
        plt.xlabel('Time (binned; ms)')
        plt.ylabel('AP (per bin, per repetition)')
        plt.subplot(122)
        plt.plot(HE[1][1:],HE[0],'b',label='No Delay')
        #plt.plot(HE[1][1:],smoothy(HE[0],windowlen=3,windowtype='hamming'),'b',label='No Delay')
        plt.plot(HA[1][1:],HA[0],'g',label='Apical lead')
        #plt.plot(HA[1][1:],smoothy(HA[0],windowlen=3,windowtype='hamming'),'g',label='Apical lead')
        plt.plot(HB[1][1:],HB[0],'r',label='Basal lead')
        #plt.plot(HB[1][1:],smoothy(HB[0],windowlen=3,windowtype='hamming'),'r',label='Basal lead')
        plt.legend()
        plt.xlabel('Time (binned; ms)')
        plt.ylabel('AP/bin, normalized')
        plt.xlim((180,230))
        #
        plt.show()
        
    #6-----------conductance threshold-----------------------------------------
    #------------------NEW 2020----------------------------------------
    elif int(sys.argv[1])==6:
        tstop=40
        dt=0.01
        if int(sys.argv[2])==1:
            RWM=[]
            RWA=[]
            RWB=[]
            nconds = 100
            weights = np.linspace(0,0.004,nconds)
            Vpeak_WM=np.zeros((nconds,))
            Vpeak_WA=np.zeros((nconds,))
            Vpeak_WB=np.zeros((nconds,))
            for icond in range(nconds):
                WM=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,
                                 starta=20.0,startb=20.0,inpn=(1,1),
                                 syntau=2,synw=weights[icond],hasrandomness=False)
                WA=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=False,
                                 starta=20.0,startb=20.0,inpn=(2,0),
                                 syntau=2,synw=weights[icond],hasrandomness=False)
                WB=runmodel(tstop=tstop,dt=dt,a_on=False,b_on=True,
                                 starta=20.0,startb=20.0,inpn=(0,2),
                                 syntau=2,synw=weights[icond],hasrandomness=False)
                RWM.append(WM['AVm'])
                RWA.append(WA['AVm'])
                RWB.append(WB['AVm'])
                Vpeak_WM[icond]=np.max(WM['AVm'])
                Vpeak_WA[icond]=np.max(WA['AVm'])
                Vpeak_WB[icond]=np.max(WB['AVm'])
            R={}
            R['nconds']=nconds
            R['weights']=weights
            R['WM']=RWM
            R['WA']=RWA
            R['WB']=RWB
            R['Vpeak_WM']=Vpeak_WM
            R['Vpeak_WA']=Vpeak_WA
            R['Vpeak_WB']=Vpeak_WB
            np.save('/home/kuenzel/Documents/Python/SCN/scnsingleinteractions.npy',R)
        else:
            try:
                R = np.load('/home/kuenzel/Documents/Python/SCN/scnsingleinteractions.npy')
                R = R.tolist()
                weights=R['weights']
                RWM=R['WM']
                RWA=R['WA']
                RWB=R['WB']
                Vpeak_WM=R['Vpeak_WM']
                Vpeak_WA=R['Vpeak_WA']
                Vpeak_WB=R['Vpeak_WB']
            except:
                sys.exit('Data not found yet, use 1 as second argument!')
        tax=np.linspace(0,10.0-dt,round(int(10.0/dt)))
        Istart = int(round(20.0/dt))
        Istop = int(round(30.0/dt))
        plt.figure(figsize=(11.6,3))
        sp1=plt.subplot(1,4,1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        sp2=plt.subplot(1,4,2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        sp3=plt.subplot(1,4,3)
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        for icond in range(R['nconds']/8):
            sp1.plot(tax,RWM[icond*8][Istart:Istop],'k-')
            sp2.plot(tax,RWA[icond*8][Istart:Istop],'b-')
            sp3.plot(tax,RWB[icond*8][Istart:Istop],'r-')
        thr=-10
        sp4=plt.subplot(1,4,4)
        plt.xlabel(r'g_syn (nS)')
        plt.ylabel('Peak Vm (mV)')
        wx=0.5*weights*1000.0
        sp4.plot(wx,Vpeak_WM,'k-',label='both')
        thrWM=np.argmax(R['Vpeak_WM'] > thr)
        sp4.plot(wx[thrWM],Vpeak_WM[thrWM],'ko',markerfacecolor='w')
        sp4.plot(wx,Vpeak_WA,'b-',label='apical')
        thrWA=np.argmax(R['Vpeak_WA'] > thr)
        sp4.plot(wx[thrWA],Vpeak_WA[thrWA],'bo',markerfacecolor='w')
        sp4.plot(wx,Vpeak_WB,'r-',label='basal')
        thrWB=np.argmax(R['Vpeak_WB'] > thr)
        sp4.plot(wx[thrWB],Vpeak_WB[thrWB],'ro',markerfacecolor='w')
        plt.legend()
        plt.tight_layout()
        plt.show()
    #6-----------temporal delay threshold-----------------------------------------
    #------------------NEW 2020----------------------------------------
    elif int(sys.argv[1])==7:
        tstop=40
        dt=0.01
        if int(sys.argv[2])==1:
            RWM=[]
            RWA=[]
            RWB=[]
            nconds = 100
            delays = np.linspace(-2.0,2.0,nconds)
            Vpeak_WM=np.zeros((nconds,))
            for icond in range(nconds):
                WM=runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,
                                 starta=20.0+delays[icond],startb=20.0,inpn=(1,1),
                                 syntau=2,synw=0.0028,hasrandomness=False)
                RWM.append(WM['AVm'])
                Vpeak_WM[icond]=np.max(WM['AVm'])
            R={}
            R['nconds']=nconds
            R['delays']=delays
            R['WM']=RWM
            R['Vpeak_WM']=Vpeak_WM
            np.save('/home/kuenzel/Documents/Python/SCN/scnsingleinteractions2.npy',R)
        else:
            try:
                R = np.load('/home/kuenzel/Documents/Python/SCN/scnsingleinteractions2.npy')
                R = R.tolist()
                delays=R['delays']
                RWM=R['WM']
                Vpeak_WM=R['Vpeak_WM']
            except:
                sys.exit('Data not found yet, use 1 as second argument!')
        tax=np.linspace(0,20.0-dt,round(int(20.0/dt)))
        Istart = int(round(15.0/dt))
        Istop = int(round(35.0/dt))
        plt.figure(figsize=(11.6,3))
        sp1=plt.subplot(1,4,1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        sp2=plt.subplot(1,4,2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        sp3=plt.subplot(1,4,3)
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        colors=np.linspace(0,0.8,R['nconds']/10)
        for icond in range(R['nconds']/10):
            sp3.plot(tax,RWM[icond*10][Istart:Istop],color=(colors[icond],colors[icond],colors[icond]))
            sp3.plot(delays[icond*10]+6,-70,'.',color=(colors[icond],colors[icond],colors[icond]))
            sp3.plot(6,-75,'r.')
        sp4=plt.subplot(1,4,4)
        plt.xlabel('Onset delay (ms)')
        plt.ylabel('Peak Vm (mV)')
        sp4.plot([0,0],[-80,50],color=(.8,.8,.8))
        sp4.plot(delays,Vpeak_WM,'k-',label='both')
        thr=-10
        negthr=np.argmax(R['Vpeak_WM']>-10)
        posthr=99-np.argmax(np.flip(R['Vpeak_WM']>-10,0))
        sp4.plot(delays[negthr],Vpeak_WM[negthr],'ko',markerfacecolor='w')
        sp4.plot((delays[negthr],0),(Vpeak_WM[negthr],Vpeak_WM[negthr]),'g--')
        sp4.annotate("{0:.3g}".format(delays[negthr]),[-0.7,-5])
        sp4.plot(delays[posthr],Vpeak_WM[posthr],'ko',markerfacecolor='w')
        sp4.plot((0,delays[posthr]),(Vpeak_WM[posthr],Vpeak_WM[posthr]),'g--')
        sp4.annotate("{0:.2g}".format(delays[posthr]),[0.2,-15])
        plt.tight_layout()
        plt.show()
        
    #7-----------re-evaluating overall temporal onset delay---------------------
    #------------------NEW 2020----------------------------------------
    if int(sys.argv[1])==8:    
        tstop=2000
        dt=0.025
        R=[]
        R.append(runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,inpn=(10,10),
                    syntau=2,synw=0.002,starta=200.0,startb=175.0,seed=32768))#syntau=2,synw=0.002
        R.append(runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,inpn=(10,10),
                    syntau=2,synw=0.002,starta=200.0,startb=200.0,seed=32768))#syntau=2,synw=0.002
        R.append(runmodel(tstop=tstop,dt=dt,a_on=True,b_on=True,inpn=(10,10),
                    syntau=2,synw=0.002,starta=175.0,startb=200.0,seed=32768))#syntau=2,synw=0.002
        taxis = np.linspace(0,tstop-dt,tstop/dt)
        titles = ('Basal leads 25ms -> ','Equal onset -> ','Apikal leads 25ms -> ')
        #
        for isub in range(3):
            plt.subplot(3,1,isub+1)
            plt.plot(taxis,R[isub]['AVm'],'r',label='Axon_Vm')
            ASp = tkt.SimpleDetectAP(R[isub]['AVm'],dt=dt,thr=-50,LM=-30)
            plt.plot(ASp['PeakT'],ASp['PeakV'],'m^',label='Axon_AP')
            plt.plot(taxis,R[isub]['SVm'],'k',label='Soma_Vm')
            SSp = tkt.SimpleDetectAP(R[isub]['SVm'],dt=dt,thr=-30,LM=-30)
            plt.plot(SSp['PeakT'],SSp['PeakV'],'gv',label='Soma_AP')
            plt.ylabel('Vm (mV)')
            print ASp['PeakT'][0:10]
            sp=np.array(ASp['PeakT'])
            sp=sp[sp>200.0]
            sp=sp[sp<1200.0]
            print sp[0:10]
            plt.title(titles[isub] + str(len(sp)) + 'Hz')
            if isub==0:
                plt.legend()
            if isub==2:
                plt.xlabel('Time (ms)')
        plt.show()
        np.save('/home/kuenzel/Documents/Python/SCN/Example_Traces_temporal.npy',R)
    else:
        print "say (1) for examples, (2) for quantification or (3/4) for similarity, (5) for relative onset, (6) for single interactiosn, conductance and (7) for single interactions / timing"
