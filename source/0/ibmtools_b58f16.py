# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/8b0658a863b16a7a7c1fd5bee7fc3188687e69a2/Coding/Qiskit/Tools/IBMTools.py
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
#IBMQ.load_account()

def run(circuit, backend, **kwargs):
    if type(backend) is str:
        backend = Aer.get_backend(backend)
    return execute(circuit, backend, **kwargs)

def textResults(results,collisions):
    for key in results:
        if(results[key]>collisions):
            text= str(key)+ '->'+  str(results[key])
    return text

def setProvider(hub,group,project):
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    return provider

def leastBusy(minQubits,provider):
    large_enough_devices = provider.backends(filters=lambda x: x.configuration().n_qubits > minQubits  and not x.configuration().simulator)
    leastBusybackend = least_busy(large_enough_devices)
    return leastBusybackend

def getJobCounts(result,backend):
    jobID = result.job_id()
    job = backend.retrieve_job(jobID)
    resultCount = job.result().get_counts()
    return resultCount

def listBackends(provider):
    for backend in provider.backends():
        print( backend.name())

def getJob(jobID,provider,backend):
    job = backend.retrieve_job(jobID)
    resultCount = job.result().get_counts()
    return resultCount

def printBestSeed(qc,basisGatesD,deviceBackend,startSeed,endSeed):
    dict = {}
    dict2 = {}
    for i in range(startSeed,endSeed):
        qCirc = transpile(qc,basis_gates=basisGatesD,backend=deviceBackend,optimization_level=3,layout_method='noise_adaptive',seed_transpiler=i)
        dict[i] = qCirc.count_ops()['cx']
        dict2[i] = qCirc.depth()
    print(min(dict.items(), key=lambda x: x[1])) 
    print(min(dict2.items(), key=lambda x: x[1]))

def simul(qc,stateVec,shots):
    if stateVec:
        backend = Aer.get_backend('statevector_simulator')
        result = execute(qc,backend,shots=shots).result().get_statevector(qc,decimals=3)
    else:
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc,backend,shots=3000).result().get_counts()
    return result

def savefig(fig,filePath,fileName):
    plt.savefig(r'/home/jaime/Programming/Jaime-Santos-Dissertation/Results/Qiskit/'+filePath+fileName)
    plt.clf()

def saveMultipleHist(N,steps,circListList,filePath,defaultFileName):
    "Saves len(steps) histogram plots of N qubit quantum walks as .png files. Default path is /Coding/Qiskit. Default file name should be \"name_N\" "
    fileName = ""
    fileNameAux = ""
    for n,circList in zip(N,circListList):
        fileName += defaultFileName+str(n)
        for circ,step in zip(circList,steps):
            fileNameAux = "_S"+str(step)  
            result = simul(circ,False)
            resultFig = plot_histogram(result)
            fileName += fileNameAux
            fileNameAux = ""
            savefig(resultFig,filePath,fileName)
            print("%s was saved to %s"%(fileName,filePath))
            fileName = defaultFileName+str(n)
        fileName = ""

def printDict(dictionary):
    for i,k in zip(dictionary.keys(),dictionary.values()):
        print("%s: %s"%(i,k))

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
    elif len(qubits)==3:
        qc.ccx(*qubits)
    elif len(qubits)==2:
        qc.cx(*qubits)
    return qc

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

def multDecResultDict2(N,steps):
    "Returns multiple binary dictionaries."
    baseResultDictList = []
    for step in steps:
    	 baseDict = decResultDict(N)
    	 baseResultDictList.append(baseDict)
    return baseResultDictList

def multDecResultDictIbm(N,steps):
    "Returns multiple binary dictionaries."
    baseResultDictList = []
    for step in steps:
        baseDict = decResultDict(N)
        baseResultDictList.append(baseDict)
    return baseResultDictList

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

def multNormalizedResultDict(baseDictList,qiskitDictList):
    "Returns the result of merging qiskit produced dictionaries with dictionaries produced from multBinResultDict for graph formatting reasons."
    normalizedResultDictList = []
    for baseDict,qiskitDict in zip(baseDictList,qiskitDictList):
        #new_d1 = {int(key): int(value) for key, value in baseDict.items()}
        #new_d2 = {int(key): int(value) for key, value in qiskitDict.items()}
        new_d1 = baseDict
        new_d2=  qiskitDict
        normalizedResultDict = {**new_d1,**new_d2}
        normalizedResultDictList.append(normalizedResultDict)
    return normalizedResultDictList

def multResultsSim(multipleCircs,shots,Decimal):
    "Returns the dictionary produced by QASM simulator with the MSB changed to convention, and values (previously frequencies) converted to probabilities."
    resultList = []
    result = {}
    correctedResult = {}
    for circList in multipleCircs:
        for circ in circList:
            result = simul(circ,False,shots)
            if Decimal:
                correctedResult = { int(k[::-1],2) : v/shots for k, v in result.items()}
            else:
                correctedResult = { k[::-1] : v/shots for k, v in result.items()}
            resultList.append(correctedResult)
            result = {}
    return resultList 


def multResultsSim2(multipleCircs,shots,Decimal):
    "Returns the dictionary produced by QASM simulator with the MSB changed to convention, and values (previously frequencies) converted to probabilities."
    resultList = []
    result = {}
    correctedResult = {}
    for circ in multipleCircs:
        result = simul(circ,False,shots)
        if Decimal:
            correctedResult = { int(k[::-1],2) : v/shots for k, v in result.items()}
        else:
            correctedResult = { k[::-1] : v/shots for k, v in result.items()}
        resultList.append(correctedResult)
        result = {}
    return resultList 

def multResultsSim3(multipleCircs,shots,Decimal):
    "Returns the dictionary produced by QASM simulator with the MSB changed to convention, and values (previously frequencies) converted to probabilities."
    resultList = []
    result = {}
    correctedResult = {}
    for circList in multipleCircs:
        for circ in circList:
            result = simul(circ,False,shots)
            if Decimal:
                correctedResult = { int(k,2) : v/shots for k, v in result.items()}
            else:
                correctedResult = { k : v/shots for k, v in result.items()}
            resultList.append(correctedResult)
            result = {}
    return resultList 

def formatResultIBM(ibmJobList,shots,Decimal):
    "Returns the dictionary produced by QASM simulator with the MSB changed to convention, and values (previously frequencies) converted to probabilities."
    resultList = []
    result = {}
    correctedResult = {}
    for resultDict in ibmJobList:
        result = resultDict 
        if Decimal:
            correctedResult = { int(k[::-1],2) : v/shots for k, v in result.items()}
        else:
            correctedResult = { k[::-1] : v/shots for k, v in result.items()}
        resultList.append(correctedResult)
    return resultList

#TODO: Delegar formatacao para uma funcao propria.
#TODO: Os labels dos eixos nao estao perfeitamente centrados. O do y fica no ultimo subplot, por alguma razao.
def multSubPlot(resultList,steps):
    "Produces a matplotlib figure composed of several subplots for different numbers of graph nodes and circuit iterations."
    nrows = len(resultList) 
    ncols = 1
    index = 1
    fig = plt.figure()
    axList = []
    auxList = []
    for resultAux,step in zip(resultList,steps):
        axList.append(fig.add_subplot(nrows,ncols,index))
        axList[-1].bar(resultAux.keys(),resultAux.values(),width=0.4,label = "Steps=%s"%step)
        axList[-1].legend()
        index+=1
    for ax in axList:
        axList[-1].get_shared_y_axes().join(axList[-1],ax)
    for ax in axList[:-1]:
        ax.set_xticklabels([])
    axList[-1].set_xticklabels(resultList[-1].keys(),rotation=45)
    plt.xlabel("Graph Node")
    plt.ylabel("Probability")
    fig.tight_layout(pad=1.0)
    return axList 

def multSubPlotIbmSim(resultListIbm,resultListSim,steps):
    "Produces a matplotlib figure composed of several subplots for different numbers of graph nodes and circuit iterations."
    nrows = len(resultListIbm) 
    ncols = 1
    index = 1
    fig = plt.figure()
    axList = []
    auxList = []
    i  = 0
    epsilon = 1e-7
    for resultAuxSim,resultAuxIbm,step in zip(resultListSim,resultListIbm,steps):
        axList.append(fig.add_subplot(nrows,ncols,index))
        if i==0:
            counts1 = [x+epsilon for x in resultAuxSim.values()]
            counts2 = [x+epsilon for x in resultAuxIbm.values()]
            print(*zip(*enumerate(counts1)))
            axList[-1].set_title("Steps=%s"%step)
            axList[-1].bar(*zip(*enumerate(counts1)),width=0.4,label = "ibmq_qasm",bottom=0,align='edge')
            axList[-1].bar(*zip(*enumerate(counts2)),width=-0.4,label = "ibmq_casablanca",bottom=0,align='edge')
            axList[-1].legend()
        else:
            counts1 = [x+epsilon for x in resultAuxSim.values()]
            counts2 = [x+epsilon for x in resultAuxIbm.values()]
            axList[-1].set_title("Steps=%s"%step)
            print(*zip(*enumerate(counts1)))
            axList[-1].bar(*zip(*enumerate(counts1)),width=0.4,bottom=0,align='edge')
            axList[-1].bar(*zip(*enumerate(counts2)),width=-0.4,bottom=0,align='edge')
        i+=1
        index+=1
        counts1= []
        counts2= []
    for ax in axList:
        axList[0].get_shared_y_axes().join(axList[0],ax)
    #for ax in axList:
    #    ax.set_xticklabels([])
    axList[-1].set_xticklabels(resultListIbm[-1].keys(),rotation=45)
    plt.xlabel("Graph Node")
    plt.ylabel("Probability")
    fig.tight_layout(pad=1.0)
    return axList 

def multSubPlotIbmSim2(resultListIbm,resultListSim,steps,backend):
    Tot = len(steps)
    Cols = 1 
    # Compute Rows required
    Rows = Tot // Cols 
    Rows += Tot % Cols
    # Create a Position index
    Position = range(1,Tot + 1)
    fig = plt.figure(1)
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['figure.figsize'] = 11,8
    mpl.rcParams.update({'font.size' : 15})
    i = 0
    for k,resultDictIbm,resultDictSim,step in zip(range(Tot),resultListIbm,resultListSim,steps):
            countsIbm = resultDictIbm.values()
            countsSim = resultDictSim.values()
            ax = fig.add_subplot(Rows,Cols,Position[k])
            if i ==0:
                ax.set_title("Time=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge',label='qasm_simulator')
                ax.bar(*zip(*enumerate(countsIbm)),width=-0.4,bottom=0,align='edge',label = str(backend))
                ax.legend()
            else:
                ax.set_title("Time=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge')
                ax.bar(*zip(*enumerate(countsIbm)),width=-0.4,bottom=0,align='edge' )
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
    return fig
    
def multSubPlotSimGrover(resultListSim,steps):
    Tot = len(steps)
    Cols = 1 
    # Compute Rows required
    Rows = Tot // Cols 
    Rows += Tot % Cols
    # Create a Position index
    Position = range(1,Tot + 1)
    fig = plt.figure(1)
    #mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['figure.figsize'] = 11,8
    mpl.rcParams.update({'font.size' : 15})
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

def multSubPlotSimContSearch(resultListSim,steps):
    Tot = len(steps)
    Cols = 1 
    # Compute Rows required
    Rows = Tot // Cols 
    Rows += Tot % Cols
    # Create a Position index
    Position = range(1,Tot + 1)
    fig = plt.figure(1)
    #mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['figure.figsize'] = 11,8
    mpl.rcParams.update({'font.size' : 15})
    i = 0
    for k,resultDictSim,step in zip(range(Tot),resultListSim,steps):
            countsSim = resultDictSim.values()
            ax = fig.add_subplot(Rows,Cols,Position[k])
            if i ==0:
                ax.set_title("Time=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge',label='qasm_simulator')
                ax.legend()
            else:
                ax.set_title("Time=%s"%step)
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

def plotMultipleQiskitIbm(N,ibmJobDictList,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    formatedDictList = formatResultIBM(ibmJobDictList,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDictIbm(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,formatedDictList)
    fig = multSubPlot(normalizedResultDictList,steps)
    return fig

def plotMultipleQiskitIbmSim2(N,multipleCircs,ibmJobDictList,steps,shots,Decimal,backend):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    formatedIbmList = formatResultIBM(ibmJobDictList,shots,Decimal)
    formatedSimList = multResultsSim2(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict2(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultListIbm = multNormalizedResultDict(baseDictList,formatedIbmList)
    normalizedResultListSim = multNormalizedResultDict(baseDictList,formatedSimList)
    fig = multSubPlotIbmSim2(normalizedResultListIbm,normalizedResultListSim,steps,backend)
    return fig

def plotMultipleQiskitIbmSim(N,multipleCircs,ibmJobDictList,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    formatedIbmList = formatResultIBM(ibmJobDictList,shots,Decimal)
    formatedSimList = multResultsSim(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultListIbm = multNormalizedResultDict(baseDictList,formatedIbmList)
    normalizedResultListSim = multNormalizedResultDict(baseDictList,formatedSimList)
    fig = multSubPlotIbmSim(normalizedResultListIbm,normalizedResultListSim,steps)
    return fig

def plotMultipleQiskit(N,multipleCircs,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    qiskitSimResultList = multResultsSim(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,qiskitSimResultList)
    fig = multSubPlot(normalizedResultDictList,steps)
    return fig

def plotMultipleQiskitGrover(N,multipleCircs,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    qiskitSimResultList = multResultsSim3(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,qiskitSimResultList)
    fig = multSubPlotSimGrover(normalizedResultDictList,steps)
    return fig

def plotMultipleQiskitGrover2(N,multipleCircs,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    qiskitSimResultList = multResultsSim2(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict2(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,qiskitSimResultList)
    fig = multSubPlotSimGrover(normalizedResultDictList,steps)
    return fig


def plotMultipleQiskitContSearch(N,multipleCircs,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    qiskitSimResultList = multResultsSim2(multipleCircs,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict2(N,steps)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,qiskitSimResultList)
    fig = multSubPlotSimContSearch(normalizedResultDictList,steps)
    return fig
    
def multSubPlotConTrot(resultListIbm,resultListSim,steps):
    Tot = len(steps)
    Cols = 1 
    # Compute Rows required
    Rows = Tot // Cols 
    Rows += Tot % Cols
    # Create a Position index
    Position = range(1,Tot + 1)
    fig = plt.figure(1)
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['figure.figsize'] = 11,8
    mpl.rcParams.update({'font.size' : 15})
    i = 0
    for k,resultDictIbm,resultDictSim,step in zip(range(Tot),resultListIbm,resultListSim,steps):
            countsIbm = resultDictIbm.values()
            countsSim = resultDictSim.values()
            ax = fig.add_subplot(Rows,Cols,Position[k])
            if i ==0:
                ax.set_title("Time=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge',label='Trotter=1')
                ax.bar(*zip(*enumerate(countsIbm)),width=-0.4,bottom=0,align='edge',label = 'Trotter=2')
                ax.legend()
            else:
                ax.set_title("Time=%s"%step)
                ax.bar(*zip(*enumerate(countsSim)),width=0.4,bottom=0,align='edge')
                ax.bar(*zip(*enumerate(countsIbm)),width=-0.4,bottom=0,align='edge' )
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
    return fig
    
def plotMultipleQiskitContSearchR(N,multipleCircsR1,multipleCircsR2,steps,shots,Decimal):
    "Brings every dictionar and plot building functions together to either show or save the matplotlib figure."
    qiskitSimResultList = multResultsSim2(multipleCircsR1,shots,Decimal)
    print(qiskitSimResultList)
    qiskitSimResultList2 = multResultsSim2(multipleCircsR2,shots,Decimal)
    if Decimal:
        baseDictList = multDecResultDict2(N,steps)
        print(N)
    else:
        baseDictList = multBinResultDict(N,steps)
    normalizedResultDictList = multNormalizedResultDict(baseDictList,qiskitSimResultList)
    normalizedResultDictList2 = multNormalizedResultDict(baseDictList,qiskitSimResultList2)
    fig = multSubPlotConTrot(normalizedResultDictList2,normalizedResultDictList,steps)
    return fig
