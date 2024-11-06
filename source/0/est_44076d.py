# https://github.com/hcoffey1/Quarry/blob/f21fd7cd4c2ce6f4ba14b840adbe8ebd26ff64bd/src/Est.py
from qiskit import QuantumCircuit, Aer, execute, IBMQ
import sys
import time
import os
from os.path import exists
from qiskit.providers.aer.noise import NoiseModel
from qiskit import transpile
import EvalMetrics as EM
import argparse

import QUtil

import PredictorV1
import PredictorV2
import SwapPredictor

def evalCircuitSim(resultDict, qc, backend):
    '''Run circuit on simulated backend and collect result metrics'''
    optimizationLevel = 0
    backendName = backend.configuration().backend_name

    if qc.name not in resultDict:
        resultDict[qc.name] = []

    print(backendName, backend.configuration().n_qubits)
    out = QUtil.simCircuit(qc, backend, optimizationLevel)
    if out != None:
        resultDict[qc.name].append([backendName, out])


def evalCircuitESP(resultDict, qc, backend):
    '''Estimate circuit fidelity via ESP'''
    optimizationLevel = 0
    backendName = backend.configuration().backend_name

    if qc.name not in resultDict:
        resultDict[qc.name] = []

    print(backendName, backend.configuration().n_qubits)
    unroll_qc = transpile(
        qc, optimization_level=optimizationLevel, backend=backend)

    esp = QUtil.getESP(unroll_qc, NoiseModel.from_backend(backend))
    resultDict[qc.name].append([backendName, esp])


def evalCircuitPredictV1(resultDict, qc, backend):
    backendName = backend.configuration().backend_name

    if qc.name not in resultDict:
        resultDict[qc.name] = []

    args = QUtil.getV1Input(qc, backend)

    outDict = {}
    for output in PredictorV1.out_columns:
        outDict[output] = float(PredictorV1.queryModel(args, output)[0][0])
    outDict["Fitness"] = EM.fitness(
        outDict["PST"], outDict["TVD"], outDict["Entropy"], outDict["Swaps"], 1, 1)
    resultDict[qc.name].append([backendName, outDict])


def evalCircuitPredictV2(resultDict, qc, backend):
    backendName = backend.configuration().backend_name

    if qc.name not in resultDict:
        resultDict[qc.name] = []

    args = QUtil.getV2Input(qc, backend)

    outDict = {}
    for output in PredictorV2.out_columns:
        outDict[output] = float(PredictorV2.queryModel(args, output)[0][0])
    outDict["Fitness"] = EM.fitness(
        outDict["PST"], outDict["TVD"], outDict["Entropy"], outDict["Swaps"], 1, 1)
    resultDict[qc.name].append([backendName, outDict])


def evalSwapPredictor(resultDict, qc, backend):
    backendName = backend.configuration().backend_name

    if qc.name not in resultDict:
        resultDict[qc.name] = {}

    if backendName not in resultDict[qc.name]:
        resultDict[qc.name][backendName] = {}

    args = QUtil.getSWAPInput(qc, backend)

    predSwaps = int(SwapPredictor.queryModel(args)[0][0])
    resultDict[qc.name][backendName]['PredSwaps'] = predSwaps


def evalSwapCompiler(resultDict, qc, backend):
    backendName = backend.configuration().backend_name

    optimizationLevel = 2

    if qc.name not in resultDict:
        resultDict[qc.name] = {}

    if backendName not in resultDict[qc.name]:
        resultDict[qc.name][backendName] = {}

    actSwaps = QUtil.getSwapCount(qc, backend, optimizationLevel)
    resultDict[qc.name][backendName]['ActSwaps'] = actSwaps


def simCircuitIBMQ(resultDict, qc, backend):
    '''Run circuit on simulated backend and collect result metrics, TODO: Update this to work with new framework'''

    backendName = backend.configuration().backend_name
    nm = NoiseModel.from_backend(backend)

    ideal_result = execute(
        qc, backend=Aer.get_backend('qasm_simulator'), max_parallel_threads=QUtil.MAX_JOBS).result()
    noisy_result = execute(qc, backend=Aer.get_backend(
        'qasm_simulator'), noise_model=nm, max_parallel_threads=QUtil.MAX_JOBS).result()

    PST = EM.computePST(correct_answer=ideal_result.get_counts(
    ).keys(), dict_in=noisy_result.get_counts())

    TVD = EM.computeTVD(dict_ideal=ideal_result.get_counts(
    ), dict_in=noisy_result.get_counts())

    IST = EM.computeIST(correct_answer=ideal_result.get_counts(
    ).keys(), dict_in=noisy_result.get_counts())

    Entropy = EM.computeEntropy(dict_in=noisy_result.get_counts())

    if qc.name not in resultDict:
        resultDict[qc.name] = {}

    if backendName not in resultDict[qc.name]:
        resultDict[qc.name][backendName] = {}

    resultDict[qc.name][backendName] = [
        ideal_result, noisy_result, PST, TVD, IST, Entropy]


def printResultsESP(resultDict, execTime):
    '''Prints metrics per backend on each circuit'''

    header = [
        ("Backend Name", 20),
        ("ESP", 10)
    ]

    def printHeader(header):
        for h in header:
            print("{h:{field_size}}".format(h=h[0], field_size=h[1]), end='')
        print('')

    for k in resultDict.keys():
        resultDict[k] = sorted(
            resultDict[k], key=lambda i: i[1], reverse=True)

    for file in resultDict.keys():
        print("{} {:.6f}(s) {}".format(
            file, execTime/(10**9), '++++++++++++++'))
        printHeader(header)

        for i in range(len(resultDict[file])):
            backend = resultDict[file][i][0]
            ESP = resultDict[file][i][1]
            print("{:20}{:<10.3f}".format(
                backend, ESP))


def printResultsSwapCompare(resultDict, execTimePred, execTimeAct):
    '''Prints metrics per backend on each circuit'''

    header = [
        ("Backend Name", 20),
        ("Predicted Swaps", 20),
        ("Actual Swaps", 20)
    ]

    def printHeader(header):
        for h in header:
            print("{h:{field_size}}".format(h=h[0], field_size=h[1]), end='')
        print('')

    #Create list of circuit/backend/swap counts
    sortedKeys = []
    for i in resultDict.keys():
        for j in resultDict[i].keys():
            sortedKeys.append(
                [i, j, resultDict[i][j]['PredSwaps'], resultDict[i][j]['ActSwaps']])

    #Sort list on predicted counts
    sortedKeys = sorted(
        sortedKeys, key=lambda i: i[2], reverse=False)

    #Organize by circuit
    sortedKeysDict = {}
    for file in resultDict.keys():
        sortedKeysDict[file] = list(filter(lambda i: i[0] == file, sortedKeys))
        sortedKeysDict[file] = [i[1] for i in sortedKeys]

    for file in resultDict.keys():
        print("{} PredTime: {:.6f}(s) ActTime: {:.6f}(s) {}".format(
            file, execTimePred/(10**9), execTimeAct/(10**9), '++++++++++++++'))
        printHeader(header)

        for i in sortedKeysDict[file]:
            backend = i
            predSwaps = resultDict[file][i]['PredSwaps']
            actSwaps = resultDict[file][i]['ActSwaps']
            print("{:20}{:<20}{:<20}".format(
                backend, predSwaps, actSwaps))


def printResultsSwap(resultDict, execTimePred):
    '''Prints metrics per backend on each circuit'''

    header = [
        ("Backend Name", 20),
        ("Swaps", 20),
    ]

    def printHeader(header):
        for h in header:
            print("{h:{field_size}}".format(h=h[0], field_size=h[1]), end='')
        print('')

    #TODO: This is really bad code, probably should fix this
    #Extract dict field into list, key could be Act or Pred swaps
    sortedKeys = []
    for i in resultDict.keys():
        for j in resultDict[i].keys():
            for k in resultDict[i][j].keys():
                sortedKeys.append([i, j, resultDict[i][j][k]])

    #Sort list on predicted counts
    sortedKeys = sorted(
        sortedKeys, key=lambda i: i[2], reverse=False)

    #Organize by circuit
    sortedKeysDict = {}
    for file in resultDict.keys():
        sortedKeysDict[file] = list(filter(lambda i: i[0] == file, sortedKeys))

    for file in resultDict.keys():
        print("{} Time: {:.6f}(s) {} ".format(
            file, execTimePred/(10**9), '++++++++++++++'))
        printHeader(header)

        for i in sortedKeysDict[file]:
            backend = i[1]
            predSwaps = str(i[2])
            print("{:20}{:<20}".format(backend, predSwaps))


def printResults(resultDict, execTime):
    '''Prints metrics per backend on each circuit'''

    header = [
        ("Backend Name", 20),
        ("PST", 10),
        ("TVD", 10),
        ("Entropy", 10),
        ("Swaps", 10),
        ("L2", 10),
        ("Hellinger", 10),
        ("Fitness", 10)
    ]

    def printHeader(header):
        for h in header:
            print("{h:{field_size}}".format(h=h[0], field_size=h[1]), end='')
        print('')

    for k in resultDict.keys():
        resultDict[k] = sorted(
            resultDict[k], key=lambda i: i[1]["Fitness"], reverse=True)

    for file in resultDict.keys():
        print("{} {:.6f}(s) {}".format(
            file, execTime/(10**9), '++++++++++++++'))
        printHeader(header)

        for i in range(len(resultDict[file])):
            backend = resultDict[file][i][0]
            PST = resultDict[file][i][1]["PST"]
            TVD = resultDict[file][i][1]["TVD"]
            Entropy = resultDict[file][i][1]["Entropy"]
            swapCount = int(resultDict[file][i][1]["Swaps"])
            Fitness = resultDict[file][i][1]["Fitness"]

            if "L2" not in resultDict[file][i][1]:
                print("{:20}{:<10.3f}{:<10.3f}{:<10.3f}{:<10}{:<10.3}{:<10}{:<10.3f}".format(
                    backend, PST, TVD, Entropy, swapCount, "N/A", "N/A", Fitness))
            else:
                L2 = resultDict[file][i][1]["L2"]
                Hellinger = resultDict[file][i][1]["Hellinger"]
                print("{:20}{:<10.3f}{:<10.3f}{:<10.3f}{:<10}{:<10.3f}{:<10.3f}{:<10.3f}".format(
                    backend, PST, TVD, Entropy, swapCount, L2, Hellinger, Fitness))


def query(qc, backends, queryFunc):
    #Simulate circuit on each backend
    timeBegin = time.time_ns()

    resultDictSim = {}
    for backend in backends:
        queryFunc(resultDictSim, qc, backend)
    timeEnd = time.time_ns()

    #Time taken to simulate
    execTime = timeEnd - timeBegin

    return resultDictSim, execTime


def QuarryInit(qasmFile, n=10):
    QUtil.GLOBAL_BASIS_GATES = QUtil.getGlobalBasisGates()

    #Read in given circuit
    qc = QuantumCircuit.from_qasm_file(qasmFile)
    qc.name = qasmFile

    backendsIBMQ = None
    TOKEN_FILE = os.environ.get('IBMQ_TOKEN')

    #TODO: Revisit support for IBMQ, this is all out of date at this point
    if TOKEN_FILE != None and exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            TOKEN = f.read()

        print("Token found, using pulling IBMQ data...", end='')
        IBMQ.save_account(TOKEN, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.providers()[0]
        backendsIBMQ = provider.backends()
        backendsIBMQ = list(filter(
            lambda backend: "simulator" not in backend.configuration().backend_name, backendsIBMQ))
        print("Done")

    else:
        backends = QUtil.getFakeBackends(qc, n)

    return qc, backends


def main():
    parser = argparse.ArgumentParser(
        description="Make a query to Quarry for the given QASM circuit file.")
    parser.add_argument(
        'file', type=str, help='QASM file to estimate fidelity on.')
    parser.add_argument(
        'mode', type=str, help='Method type to query with (simulation|P1|P2|ESP|SWAP_PRED|SWAP_COMPILE|SWAP_COMPARE)')
    parser.add_argument(
        '--n', type=int, help='Number of backend platforms to test on. (Default 10)', default=10)

    args = parser.parse_args()

    sys.stderr.write("QUARRY LOG HEADER ==========\n")

    backendCount = args.n
    inputFile = args.file

    #Transform file to Qiskit circuit and retrieve compatible platforms
    qc, backends = QuarryInit(inputFile, backendCount)

    sys.stderr.write("Mode: {}\n".format(args.mode.lower()))
    sys.stderr.write("CircuitSize: {}\n".format(qc.size()))
    sys.stderr.write("NumQubits: {}\n".format(qc.num_qubits))

    #ESP Estimate
    if args.mode.lower() == "esp":
        resultDict, execTime = query(qc, backends, evalCircuitESP)
        printResultsESP(resultDict, execTime)

    #Simulation
    elif args.mode.lower() == "simulation":
        resultDict, execTime = query(qc, backends, evalCircuitSim)
        printResults(resultDict, execTime)

    #ML Models
    elif args.mode.lower() == "p1":
        PredictorV1.load_models()
        resultDict, execTime = query(qc, backends, evalCircuitPredictV1)
        printResults(resultDict, execTime)

    elif args.mode.lower() == "p2":
        PredictorV2.load_models()
        resultDict, execTime = query(qc, backends, evalCircuitPredictV2)
        printResults(resultDict, execTime)

    elif args.mode.lower() == "swap_pred":
        SwapPredictor.load()
        resultDictSwapPred, execTimeSwapPred = query(
            qc, backends, evalSwapPredictor)

        printResultsSwap(resultDictSwapPred, execTimeSwapPred)

    elif args.mode.lower() == "swap_compile":
        resultDictSwapAct, execTimeSwapAct = query(
            qc, backends, evalSwapCompiler)

        printResultsSwap(resultDictSwapAct, execTimeSwapAct)

    elif args.mode.lower() == "swap_compare":
        SwapPredictor.load()
        resultDictSwapPred, execTimeSwapPred = query(
            qc, backends, evalSwapPredictor)
        resultDictSwapAct, execTimeSwapAct = query(
            qc, backends, evalSwapCompiler)

        #Merge predicted and actual dicts
        for i in resultDictSwapPred.keys():
            for j in resultDictSwapPred[i].keys():
                resultDictSwapPred[i][j] = {
                    **(resultDictSwapPred[i][j]), **(resultDictSwapAct[i][j])}

        printResultsSwapCompare(
            resultDictSwapPred, execTimeSwapPred, execTimeSwapAct)


if __name__ == "__main__":
    main()
