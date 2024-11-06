# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/8b0658a863b16a7a7c1fd5bee7fc3188687e69a2/Coding/Qiskit/CoinedQuantumWalk/runWalk.py
import sys
#sys.path.append('../Tools')
#from IBMTools import( 
#        simul,
#        savefig,
#        saveMultipleHist,
#        printDict,
#        plotMultipleQiskit,
#        plotMultipleQiskitIbm,
#        plotMultipleQiskitIbmSim,
#        multResultsSim,
#        setProvider,
#        leastBusy,
#        listBackends,
#        getJob)
import tools as pyt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from qiskit import( ClassicalRegister,
        QuantumRegister,
        QuantumCircuit,
        execute,
        Aer,
        IBMQ,
        transpile)
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.providers.aer.noise import NoiseModel
from qiskit.visualization import( plot_histogram,
                        plot_state_city,
                        plot_gate_map, 
                        plot_circuit_layout)
from math import (log,ceil)
plt.rcParams['figure.figsize'] = 11,8
matplotlib.rcParams.update({'font.size' : 15})


#CNot decomposition
def cnx(qc,*qubits):
    if len(qubits) >= 3:
        last = qubits[-1]
        #A matrix: (made up of a  and Y rotation, lemma4.3)
        qc.crz(np.pi/2, qubits[-2], qubits[-1])
        #cry
        qc.cu(np.pi/2, 0, 0,0, qubits[-2],qubits[-1])
        #Control not gate
        cnx(qc,*qubits[:-2],qubits[-1])
        #B matrix (cry again, but opposite angle)
        qc.cu(-np.pi/2, 0, 0,0, qubits[-2], qubits[-1])
        #Control
        cnx(qc,*qubits[:-2],qubits[-1])
        #C matrix (final rotation)
        qc.crz(-np.pi/2,qubits[-2],qubits[-1])
   # elif len(qubits)==3:
   #     qc.ccx(*qubits)
    elif len(qubits)==2:
        qc.cx(*qubits)
    return qc

def incr(qwc,q,subnode,n):
    for j in range(-1,n-1):
        if(j==-1):
            cnx(qwc,subnode[0],*q[-1::-1])
            #qwc.barrier()
        else:
            cnx(qwc,subnode[0],*q[-1:j:-1])
           # qwc.barrier()
    return qwc

def decr(qwc,q,subnode,n):
    qwc.x(subnode[0])
    c=0
    qwc.x(q[-1:0:-1])
    for j in range(-1,n-1):
        if(j==-1):
            c+=1
            cnx(qwc,subnode[0],*q[-1::-1])
            qwc.x(q[c])
            #qwc.barrier()
        else:
            c+=1
            cnx(qwc,subnode[0],*q[-1:j:-1])
            if(c==n):
                break
            qwc.x(q[c])
            #qwc.barrier()
    qwc.x(subnode[0])
    return qwc

def incrCirc(qc,q,subnode,n,toGate):
    for j in range(-1,n-1):
        if(j==-1):
            cnx(qc,subnode[0],*q[-1::-1])
        else:
            cnx(qc,subnode[0],*q[-1:j:-1])
    if toGate:
        qc = qc.to_gate()
        qc.name = '      INC      '
    return qc

def decrCirc(qc,q,subnode,n,toGate):
    qc.x(subnode[0])
    c=0
    qc.x(q[-1:0:-1])
    for j in range(-1,n-1):
        if(j==-1):
            c+=1
            cnx(qc,subnode[0],*q[-1::-1])
            qc.x(q[c])
        else:
            c+=1
            cnx(qc,subnode[0],*q[-1:j:-1])
            if(c==n):
                break
            qc.x(q[c])
    qc.x(subnode[0])
    if toGate:
        qc = qc.to_gate()
        qc.name = '      DEC      '
    return qc

def runWalk(N,steps,stateVec):
    "Creates a single instance of the coined quantum walk cicuit."
    qreg = QuantumRegister(N)
    qsub = QuantumRegister(1)
    creg = ClassicalRegister(N)
    qwc = QuantumCircuit(qreg,qsub,creg)
    qwc.x(qreg[0])
    for i in range(0,steps):
        qwc.h(qsub[0])
        qwc.barrier()
        incr(qwc,qreg,qsub,N)
        qwc.barrier()
        decr(qwc,qreg,qsub,N)
        qwc.barrier()
    if not stateVec:
        qwc.measure(qreg,creg)
    #qwc.draw(output='mpl')
    #plt.show()
    return qwc


def runMultipleWalks(N,steps,stateVec):
    "Creates several instances of the coined quantum walk circuit."
    circList = []
    circListAux = []
    for n in N:
        qreg = QuantumRegister(n)
        qsub = QuantumRegister(1)
        creg = ClassicalRegister(n)
        for step in steps:
            circ = QuantumCircuit(qreg,qsub,creg)
            circ = runWalk(n,step,stateVec)
            circListAux.append(circ)
        circList.append(circListAux)
        circListAux = []
    return circList

def runMultipleWalksLite(N,steps,stateVec):
    "Creates several instances of the coined quantum walk circuit."
    circList = []
    circListAux = []
    qreg = QuantumRegister(N)
    qsub = QuantumRegister(1)
    creg = ClassicalRegister(N)
    for step in steps:
        circ = QuantumCircuit(qreg,qsub,creg)
        circ = runWalk(N,step,stateVec)
        circList.append(circ)
    return circList


def circRunWalk(N,steps,toGate):
    "Creates a single instance of the coined quantum walk cicuit."
    qreg = QuantumRegister(N,name='node')
    qsub = QuantumRegister(1, name='coin')
    creg = ClassicalRegister(N)
    qwc = QuantumCircuit(qreg,qsub,creg)
    incrCirc1 = QuantumCircuit(qreg,qsub)
    decrCirc1 = QuantumCircuit(qreg,qsub)
    incrCirc1 = incrCirc(incrCirc1,qreg,qsub,N,toGate)
    decrCirc1 = decrCirc(decrCirc1,qreg,qsub,N,toGate)
    qwc.x(qreg[0])
    qwc.h(qsub[0])
    qwc.barrier()
    for i in range(0,steps):
        qwc.append(incrCirc1,[N]+list(range(0,N)))
        qwc.append(decrCirc1,[N]+list(range(0,N)))
        qwc.barrier()
        if i!=steps-1:
            qwc.h(qsub[0])
    qwc.measure(qreg,creg)
    return qwc

def initialCond(string,N,toGate):
    qc = QuantumCircuit(N+1)
    for x in range(N):
        print(string[x])
        if string[x] == '1':
            qc.x(x)
    qc.h(N)
    if toGate:
        qc = qc.to_gate()
        qc.name = '      INIT      '

    return qc

def circRunWalk2(N,steps,string,toGate):
    "Creates a single instance of the coined quantum walk cicuit."
    qreg = QuantumRegister(N,name='node')
    qsub = QuantumRegister(1, name='coin')
    creg = ClassicalRegister(N)
    qwc = QuantumCircuit(qreg,qsub,creg)
    incrCirc1 = QuantumCircuit(qreg,qsub)
    decrCirc1 = QuantumCircuit(qreg,qsub)
    incrCirc1 = incrCirc(incrCirc1,qreg,qsub,N,toGate)
    decrCirc1 = decrCirc(decrCirc1,qreg,qsub,N,toGate)
    initCond = initialCond(string,N,True)
    qwc.append(initCond, [N]+list(range(0,N)))
    qwc.barrier()
    for i in range(0,steps):
        qwc.append(incrCirc1,[N]+list(range(0,N)))
        qwc.append(decrCirc1,[N]+list(range(0,N)))
        qwc.barrier()
        if i!=steps-1:
            qwc.h(qsub[0])
    qwc.measure(qreg,creg)
    return qwc

def circRunMultipleWalks(N,steps,toGate):
    "Creates several instances of the coined quantum walk circuit."
    circList = []
    circListAux = []
    for n in N:
        qreg = QuantumRegister(n)
        qsub = QuantumRegister(1)
        creg = ClassicalRegister(n)
        for step in steps:
            circ = QuantumCircuit(qreg,qsub,creg)
            circ = circRunWalk(n,step,toGate)
            circListAux.append(circ)
        circList.append(circListAux)
        circListAux = []
    return circList

def saveCoinedWalkFig(N,steps,fig, filePath, defaultFileName):
    specificFileName = ""
    i=0
    for n in N:
        specificFileName+= "N%s_S"%n
        for step in steps:
            specificFileName+="%s"%step
        i+=1
        if(len(N)-i==0):
            break
        specificFileName+="_"
    pyt.savefig(fig, filePath,defaultFileName+specificFileName)
    return specificFileName

def printIncr(N,steps,style):
    "Creates a single instance of the coined quantum walk cicuit."
    for n in N:
        qreg = QuantumRegister(n,name='node')
        qsub = QuantumRegister(1, name='coin')
        incrCirc1 = QuantumCircuit(qreg,qsub)
        incrCirc1 = incrCirc(incrCirc1,qreg,qsub,n,False)
        fig = incrCirc1.draw(output='mpl',style=style) 
    return fig 

def printDecr(N,steps,style):
    "Creates a single instance of the coined quantum walk cicuit."
    for n in N:
        qreg = QuantumRegister(n,name='node')
        qsub = QuantumRegister(1, name='coin')
        decrCirc1 = QuantumCircuit(qreg,qsub)
        decrCirc1 = decrCirc(decrCirc1,qreg,qsub,n,False)
        fig = decrCirc1.draw(output='mpl',style=style) 
    return fig 

def drawCirc(circMultWalk,style):
    for circList in circMultWalk:
        for circ in circList:
            fig = circ.draw(output='mpl',style=style)
    return fig

#IBMQ.load_account()
#provider = setProvider('ibm-q-minho','academicprojects','quantalab')
##leastBusyBackend =leastBusy(10,provider)
##print("Least busy backend:",leastBusyBackend)
##8
#melBackend = provider.get_backend('ibmq_16_melbourne')
##32QV
#bogBackend = provider.get_backend('ibmq_bogota')
#parisBackend = provider.get_backend('ibmq_paris')
#manhatBackend = provider.get_backend('ibmq_manhattan')
#torontoBackend = provider.get_backend('ibmq_toronto')
#casablancaBackend = provider.get_backend('ibmq_casablanca')
##Chosen
#backend = parisBackend
#simulator = provider.get_backend('ibmq_qasm_simulator')

filePath = 'CoinedQuantumWalk/'
circFilePath = 'CoinedQuantumWalk/Circuits/'
circIncrFilePath = 'CoinedQuantumWalk/Circuits/'
circDecrFilePath = 'CoinedQuantumWalk/Circuits/'
defaultFileName = "CoinedQW_"
circDefaultFileName = "circCoinedQW_"
circIncrDefaultFileName = "circIncr_"
circDecrDefaultFileName = "circDecr_"
style = {'figwidth':18,'fontsize':17,'subfontsize':14}
styleIncr = {'figwidth':15,'fontsize':17,'subfontsize':14, 'compress':True}
styleDecr = {'figwidth':15,'fontsize':17,'subfontsize':14 }

singleN = 3 
singleSteps = 1 

#circ = circRunWalk2(singleN,singleSteps, '100')
#circ = initialCond('110',singleN,False)
#circ.draw(output='mpl')
#plt.show()

#Coined quantum walk probability distribution.
N=[3]
steps=[0,1,2,3]
shots = 3000
circList = []
multipleWalks = runMultipleWalks(N,steps,False)
fig=pyt.plotMultipleQiskit(N,multipleWalks,steps,shots,True)
saveCoinedWalkFig(N,steps,fig,filePath,defaultFileName)
#print(multipleWalks)
#circsIbm = transpile(multipleWalks,backend=backend,optimization_level=3,layout_method='noise_adaptive')
#jobMult1 = execute(circsIbm, backend=backend, shots=shots)
#job_monitor(jobMult1)
#
#for circ in multipleWalks:
#    circAux =  transpile(circ,backend=backend,optimization_level=3,layout_method='noise_adaptive')
#    circList.append(circAux)
#
#jobMult2 = execute(circList, backend=backend, shots=shots)
#job_monitor(jobMult2)
#singleWalkCirc1 = transpile(singleWalk,backend=backend,optimization_level=3,layout_method='noise_adaptive')

#singleWalkCirc2 = transpile(singleWalk,backend=backend,optimization_level=3,layout_method='dense')
#singleWalkCirc3 = transpile(singleWalk,backend=backend,optimization_level=3)
#listJob1 = [singleWalkCirc1, singleWalkCirc1, singleWalkCirc1]
#listJob2 = [singleWalkCirc2, singleWalkCirc2, singleWalkCirc2]
#listJob3 = [singleWalkCirc3, singleWalkCirc3, singleWalkCirc3]

#job1= execute(listJob1, backend=backend, shots=shots)
#print("Starting Noise Adpative")
#job_monitor(job1,1)
#job2 = execute(listJob2, backend=backend, shots=shots)
#print("Starting Dense")
#job_monitor(job2,1)
#job3 = execute(listJob3, backend=backend, shots=shots)
#print("Starting Vanilla")
#job_monitor(job3,1)
#result = job.result()
#counts = result.get_counts()
#oldJob = getJob('604c9cb4a2ec4ec349013901',provider,backend) 
#oldJob = [{'000': 195, '001': 2737, '010': 1, '011': 26, '100': 3, '101': 37, '111': 1}, {'000': 317, '001': 247, '010': 302, '011': 268, '100': 505, '101': 460, '110': 528, '111': 373}, {'000': 399, '001': 343, '010': 379, '011': 361, '100': 423, '101': 348, '110': 397, '111': 350}, {'000': 388, '001': 284, '010': 434, '011': 312, '100': 509, '101': 333, '110': 398, '111': 342}]
#print(oldJob)
#plotMultipleQiskitIbmSim(N,multipleWalks,oldJob,steps,shots,True)
#noiseModel = NoiseModel.from_backend(backend)
#basisGates = noiseModel.basis_gates
#couplingMap = backend.configuration().coupling_map
#singleWalkIbm = transpile(singleWalk,backend=backend,layout_method='noise_adaptive',optimization_level=3)
#job = execute(singleWalkIbm,backend,shots=shots)
#job_monitor(job)
#jobResult = job.result()
#jobCount = jobResult.get_counts(singleWalkIbm)
#resultNoise = execute(singleWalkIbm,simulator,noise_model=noiseModel,coupling_map=couplingMap,basis_gates=basisGates,shots=shots).result()
#countsNoise = resultNoise.get_counts(singleWalkIbm)
#correctedResult = { str(int(k[::-1],2)) : v/shots for k, v in countsNoise.items()}

#plot_histogram(correctedResult)
#plt.show()
#oldResult = oldJob.result()
#oldCounts = oldResult.get_counts()
#print(oldCounts)

#multSimDict = multResultsSim(multipleWalks,shots,True)
#print(multSimDict) 
#fig = plotMultipleQiskit(N,multipleWalks,steps,shots,True)
#saveCoinedWalkFig(N,steps,fig,filePath,defaultFileName)

##Coined quantum walk probability distribution.
#N5=[5]
#steps5=[0,5,10,15,20]
#shots = 3000
#multipleWalks5 = runMultipleWalks(N5,steps5,False)
#fig5 = plotMultipleQiskit(N5,multipleWalks5,steps5,shots,True)
#saveCoinedWalkFig(N5,steps5,fig5,filePath,defaultFileName)
#
##Coined quantum walk circuit.
#circN = [3]
#circSteps = [3]
#circMultWalk = circRunMultipleWalks(circN,circSteps)
#circFig = drawCirc(circMultWalk,style)
#saveCoinedWalkFig(circN,circSteps,circFig,circFilePath,circDefaultFileName)
#
##Increment circuit.
#circIncrN = [3]
#circIncrSteps = [3]
#circIncrFig= printIncr(circIncrN,circIncrSteps,styleIncr)
#saveCoinedWalkFig(circIncrN,circIncrSteps,circIncrFig,circIncrFilePath,circIncrDefaultFileName)
#
##Decrement circuit.
#circDecrN = [3]
#circDecrSteps = [3]
#circDecrFig= printDecr(circDecrN,circDecrSteps,styleDecr)
#saveCoinedWalkFig(circDecrN,circDecrSteps,circDecrFig,circDecrFilePath,circDecrDefaultFileName)
