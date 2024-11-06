# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/8b0658a863b16a7a7c1fd5bee7fc3188687e69a2/Coding/Qiskit/CoinedQuantumWalk/tools.py
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit import *
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from qiskit.visualization import( plot_histogram,
                        plot_state_city)
import numpy as np

def savefig(fig,filePath,fileName):
    plt.savefig(r'/home/jaime/Programming/Jaime-Santos-Dissertation/Results/Qiskit/'+filePath+fileName)
    plt.clf()


def simul(qc,stateVec,shots):
    if stateVec:
        backend = Aer.get_backend('statevector_simulator')
        result = execute(qc,backend,shots=shots).result().get_statevector(qc,decimals=3)
    else:
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc,backend,shots=3000).result().get_counts()
    return result

def binResultDict(n):
    "Retuns a dictionary composed of a range of N keys converted to binary."
    baseDict = {}
    for decNumber in range(2**n):
        decToBin = bin(decNumber)[2:].zfill(ceil(log(2**n,2)))
        baseDict[str(decToBin)] = 0
    return baseDict

def multBinResultDict(N,steps):
    "Returns multiple binary dictionaries."
    baseResultDictList = []
    for n in N:
        for step in steps:
            baseDict = binResultDict(n)
            baseResultDictList.append(baseDict)
    return baseResultDictList

def multSubPlotSim(resultListSim,steps):
    Tot = len(steps)
    Cols = 1
    # Compute Rows required
    Rows = Tot // Cols
    Rows += Tot % Cols
    # Create a Position index
    Position = range(1,Tot + 1)
    fig = plt.figure(1)
    #mpl.rcParams.update(mpl.rcParamsDefault)
    #mpl.rcParams['figure.figsize'] = 11,8
    #mpl.rcParams.update({'font.size' : 15})
    i = 0
    for k,resultDictSim,step in zip(range(Tot),resultListSim,steps):
            countsSim = resultDictSim.values()
            ax = fig.add_subplot(Rows,Cols,Position[k])
            if i ==0:
                ax.set_title("Steps=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge',label='qasm_simulator')
                ax.legend()
            else:
                ax.set_title("Steps=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge')
            plt.ylim(0,1.2)
            plt.yticks([0,0.5,1])
            plt.xlim(0-1,len(countsSim))
            w = ax.get_xaxis()
            if(i==Tot-1):
                w.set_visible(True)
                plt.xticks(range(0,len(countsSim)))
            else:
                w.set_visible(False)
            i+=1
    plt.xlabel("Graph Node")
    plt.ylabel("Probability")
    fig.tight_layout(pad=1.0)

def multResultsSim(multipleCircs,shots,Decimal):
    resultList = []
    result = {}
    correctedResult = {}
    for circList in multipleCircs:
        for circ in circList:
            result = simul(circ,False,shots)
            print(result)
            if Decimal:
                correctedResult = { int(k[::-1],2) : v/shots for k, v in result.items()}
            else:
                correctedResult = { k: v/shots for k, v in result.items()}
            resultList.append(correctedResult)
            result = {}
    return resultList

def decResultDict(n):
    "Retuns a dictionary composed of a range of N keys converted to binary."
    baseDict = {}
    for decNumber in range(2**n):
        dec = decNumber
        baseDict[dec] = 0
    return baseDict

def multDecResultDict(N,steps):
    "Returns multiple binary dictionaries."
    baseResultDictList = []
    for n in N:
        for step in steps:
            baseDict = decResultDict(n)
            baseResultDictList.append(baseDict)
    return baseResultDictList

def multNormalizedResultDict(baseDictList,qiskitDictList):
    normalizedResultDictList = []
    for baseDict,qiskitDict in zip(baseDictList,qiskitDictList):
        #new_d1 = {int(key): int(value) for key, value in baseDict.items()}
        #new_d2 = {int(key): int(value) for key, value in qiskitDict.items()}
        #new_d1 = baseDict
        #new_d2=qiskitDict
        #normalizedResultDict = {**new_d2,**new_d1}
        baseDict.update(qiskitDict)
        normalizedResultDictList.append(baseDict)
        #print(baseDict)
        #print()
        #print(qiskitDict)
    return normalizedResultDictList

def plotMultipleQiskit(N,multipleCircs,steps,shots,Decimal):
    qiskitSimResultList = multResultsSim(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,qiskitSimResultList)
    fig = multSubPlotSim(normalizedResultDictList,steps)
    return fig
