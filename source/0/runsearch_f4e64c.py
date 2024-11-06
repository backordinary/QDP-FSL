# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/09a901ac6c7087728a54aaceece34e273a2df84d/Coding/Qiskit/GroverQiskit/runSearch.py
import sys
sys.path.append('../Tools')
from IBMTools import( 
        simul,
        savefig,
        saveMultipleHist,
        printDict,
        plotMultipleQiskit,
        plotMultipleQiskitGrover)
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from qiskit import( ClassicalRegister,
        QuantumRegister,
        QuantumCircuit,
        execute,
        Aer,
        transpile
        )
from qiskit.visualization import( plot_histogram,
        plot_state_city)
mpl.rcParams['figure.figsize'] = 11,8
mpl.rcParams.update({'font.size' : 15})

def markedListGrover(markedList,N):
    oracleList = np.ones(2**N)
    for element in markedList:
        oracleList[element] = -1
    return oracleList.tolist()

def getOracle(markedList,N):
    oracleList = np.eye(2**N)
    for element in markedList:
        oracleList[element][element] = -1
    return oracleList

def oracleGrover(markedList,N):
    qreg = QuantumRegister(N)
    qc = QuantumCircuit(qreg,name='    Oracle    ')
    qc.diagonal(markedList,qreg)
    qc=transpile(qc,optimization_level=3)
    return qc

def diffusionGrover(N):
    qreg = QuantumRegister(N)
    difCirc = QuantumCircuit(qreg,name='     Diff    ')
    difCirc.h(qreg)
    aux = markedListGrover([0],N)
    qcAux = oracleGrover(aux,N)
    difCirc.append(qcAux,range(N))
    difCirc.h(qreg)
    difCirc=transpile(difCirc,optimization_level=3)
    return difCirc

def grover(N,steps,marked):
    qc = QuantumCircuit(N,N)
    qcOracle = oracleGrover(markedListGrover(marked,N),N)
    qcDiffusion = diffusionGrover(N)
    qc.h(range(N))
    for i in range(steps):
        qc.append(qcOracle,range(N))
        qc.barrier()
        qc.append(qcDiffusion,range(N))
        qc.barrier()
    qc.barrier()
    qc.measure(range(N),range(N))
    qc = transpile(qc,optimization_level=1)
    return qc

def drawCirc(N,steps,marked,style):
    qc = QuantumCircuit(N,N)
    qcOracle = oracleGrover(markedListGrover(marked,N),N)
    qcDiffusion = diffusionGrover(N)
    qc.h(range(N))
    qc.barrier()
    for i in range(steps):
        qc.append(qcOracle,range(N))
        qc.append(qcDiffusion,range(N))
        qc.barrier()
    qc.measure(range(N),range(N))
    qc = transpile(qc)
    fig = qc.draw(output='mpl',style=style)
    return qc

def drawOracle(N,marked,style):
    qreg = QuantumRegister(N)
    qc = QuantumCircuit(qreg,name='    Oracle    ')
    qc.diagonal(markedListGrover(marked,N),qreg)
    qc=transpile(qc,basis_gates=['cx','rz','ccx','x','h'])
    fig = qc.draw(output='mpl',style=style)
    return fig 

def drawDiffusion(N,style):
    qreg = QuantumRegister(N)
    difCirc = QuantumCircuit(qreg,name='     Diff    ')
    difCirc.h(qreg)
    aux = markedListGrover([0],N)
    qcAux = oracleGrover(aux,N)
    difCirc.append(qcAux,range(N))
    difCirc.h(qreg)
    difCirc=transpile(difCirc,basis_gates=['cx','rz','ccx','x','h'])
    fig = difCirc.draw(output='mpl',style=style)
    return fig


def saveGroverSearchFig(N,steps,markedVertex,fig, filePath, defaultFileName):
    specificFileName = ""
    i=0
    for n,m in zip(N,markedVertex):
        specificFileName+= "N%s_M%s_S"%(n,m)
        for step in steps:
            specificFileName+="%s"%step
        i+=1
        if(len(N)-i==0):
            break
        specificFileName+="_"
    savefig(fig, filePath,defaultFileName+specificFileName)
    plt.clf()
    return specificFileName

def runMultipleSearchComplete(N,steps,markedVertex):
     "Creates several instances of the grover search circuit."
     circList = []
     circListAux = []
     for n in N:
         qreg = QuantumRegister(n)
         qsub = QuantumRegister(1)
         creg = ClassicalRegister(n)
         for step in steps:
             circ = QuantumCircuit(qreg,qsub,creg)
             circ = grover(n,step,markedVertex)
             circListAux.append(circ)
         circList.append(circListAux)
         circListAux = []
     return circList

filePath = 'GroverQiskit/'
defaultFileName = "GroverQiskitSearch_"
circFilePath = 'GroverQiskit/Circuits/'
defaultCircFileName = "GroverQiskitCirc_"
defaultCircOracleFileName = "GroverQiskitCircOracle_"
defaultCircDiffFileName = "GroverQiskitCircDiff_"


style = {'figwidth':20,'fontsize':17,'subfontsize':14}#,'compress':True}

N = [3]
markedList = [4] 
steps = [0,1,2,3]
shots = 3000

singleN = 3
singleSteps = 3

circN = [3]
circSteps = [3]
circMarked = [4]

#drawCircFig = drawCirc(singleN,singleSteps,circMarked,style)
#saveGroverSearchFig(circN,circSteps,circMarked,drawCircFig, circFilePath, defaultCircFileName)
#
#drawCircOracleFig = drawOracle(singleN,circMarked,style)
#saveGroverSearchFig(circN,circSteps,circMarked,drawCircOracleFig, circFilePath, defaultCircOracleFileName)
#
#drawCircDiffFig = drawDiffusion(singleN,style)
#saveGroverSearchFig(circN,circSteps,circMarked,drawCircDiffFig, circFilePath, defaultCircDiffFileName)
#multipleGrover = runMultipleSearchComplete(N,steps,markedList)
#fig = plotMultipleQiskitGrover(N,multipleGrover,steps,shots,True)
#saveGroverSearchFig(N,steps,markedList,fig,filePath,defaultFileName)
