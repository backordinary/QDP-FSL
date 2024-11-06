# https://github.com/Seanaventure/HighErrorRateRouting/blob/dd4d9d3ea04076d278ccfc859a2baf94c3cc654f/HERRBvCompTime.py
import matplotlib.pyplot as plt
import networkx as nx
import qiskit
import HERR
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing import *
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from math import pi
from qiskit.compiler import transpile, assemble
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.tools.visualization import dag_drawer
import random
from qiskit.circuit.instruction import Instruction
import time


def countTwoQubitGates(transpiledCircuit):
    num = 0
    for gate in transpiledCircuit.data:
        # print(type(gate[0]))
        if issubclass(type(gate[0]), Instruction):
            if gate[0].name == "cx":
                num += 1
    return num

s = '1101'
n = len(s)

circuit = QuantumCircuit(8, n)
# Step 0

circuit.x(n)  # the n+1 qubits are indexed 0...n, so the last qubit is index n

circuit.barrier()  # just a visual aid for now

# Step 1

# range(n+1) returns [0,1,2,...,n] in Python. This covers all the qubits
circuit.h(range(n+1))

circuit.barrier()  # just a visual aid for now

# Step 2

for ii, yesno in enumerate(reversed(s)):
    if yesno == '1':
        circuit.cx(ii, n)

circuit.barrier()  # just a visual aid for now

# Step 3

# range(n+1) returns [0,1,2,...,n] in Python. This covers all the qubits
circuit.h(range(n+1))

circuit.barrier()  # just a visual aid for now

# measure the qubits indexed from 0 to n-1 and store them into the classical bits indexed 0 to n-
circuit.measure(range(n), range(n))

provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_lima')

basis_gates = backend.configuration().basis_gates


squareCouplingList = list()
if 0 is not 0:
    if abs(0-0) == 1:
        squareCouplingList.append([0, 0])
if 0 is not 1:
    if abs(0-1) == 1:
        squareCouplingList.append([0, 1])
if 0 is not 2:
    if abs(0-2) == 1:
        squareCouplingList.append([0, 2])
if 0 is not 3:
    if abs(0-3) == 1:
        squareCouplingList.append([0, 3])
if 1 is not 0:
    if abs(1-0) == 1:
        squareCouplingList.append([1, 0])
if 1 is not 1:
    if abs(1-1) == 1:
        squareCouplingList.append([1, 1])
if 1 is not 2:
    if abs(1-2) == 1:
        squareCouplingList.append([1, 2])
if 1 is not 3:
    if abs(1-3) == 1:
        squareCouplingList.append([1, 3])
if 2 is not 0:
    if abs(2-0) == 1:
        squareCouplingList.append([2, 0])
if 2 is not 1:
    if abs(2-1) == 1:
        squareCouplingList.append([2, 1])
if 2 is not 2:
    if abs(2-2) == 1:
        squareCouplingList.append([2, 2])
if 2 is not 3:
    if abs(2-3) == 1:
        squareCouplingList.append([2, 3])
if 3 is not 0:
    if abs(3-0) == 1:
        squareCouplingList.append([3, 0])
if 3 is not 1:
    if abs(3-1) == 1:
        squareCouplingList.append([3, 1])
if 3 is not 2:
    if abs(3-2) == 1:
        squareCouplingList.append([3, 2])
if 3 is not 3:
    if abs(3-3) == 1:
        squareCouplingList.append([3, 3])
squareCouplingList.append(([0, 3]))
squareCouplingList.append(([3, 0]))
squareCouplingMap = CouplingMap(squareCouplingList)


gridCouplingList = list()
if 0 is not 0:
    if abs(0-0) == 1:
        gridCouplingList.append([0, 0])
if 0 is not 1:
    if abs(0-1) == 1:
        gridCouplingList.append([0, 1])
if 0 is not 2:
    if abs(0-2) == 1:
        gridCouplingList.append([0, 2])
if 0 is not 3:
    if abs(0-3) == 1:
        gridCouplingList.append([0, 3])
if 1 is not 0:
    if abs(1-0) == 1:
        gridCouplingList.append([1, 0])
if 1 is not 1:
    if abs(1-1) == 1:
        gridCouplingList.append([1, 1])
if 1 is not 2:
    if abs(1-2) == 1:
        gridCouplingList.append([1, 2])
if 1 is not 3:
    if abs(1-3) == 1:
        gridCouplingList.append([1, 3])
if 2 is not 0:
    if abs(2-0) == 1:
        gridCouplingList.append([2, 0])
if 2 is not 1:
    if abs(2-1) == 1:
        gridCouplingList.append([2, 1])
if 2 is not 2:
    if abs(2-2) == 1:
        gridCouplingList.append([2, 2])
if 2 is not 3:
    if abs(2-3) == 1:
        gridCouplingList.append([2, 3])
if 3 is not 0:
    if abs(3-0) == 1:
        gridCouplingList.append([3, 0])
if 3 is not 1:
    if abs(3-1) == 1:
        gridCouplingList.append([3, 1])
if 3 is not 2:
    if abs(3-2) == 1:
        gridCouplingList.append([3, 2])
if 3 is not 3:
    if abs(3-3) == 1:
        gridCouplingList.append([3, 3])
if 4 is not 4:
    if abs(4-4) == 1:
        gridCouplingList.append([4, 4])
if 4 is not 5:
    if abs(4-5) == 1:
        gridCouplingList.append([4, 5])
if 4 is not 6:
    if abs(4-6) == 1:
        gridCouplingList.append([4, 6])
if 4 is not 7:
    if abs(4-7) == 1:
        gridCouplingList.append([4, 7])
if 5 is not 4:
    if abs(5-4) == 1:
        gridCouplingList.append([5, 4])
if 5 is not 5:
    if abs(5-5) == 1:
        gridCouplingList.append([5, 5])
if 5 is not 6:
    if abs(5-6) == 1:
        gridCouplingList.append([5, 6])
if 5 is not 7:
    if abs(5-7) == 1:
        gridCouplingList.append([5, 7])
if 6 is not 4:
    if abs(6-4) == 1:
        gridCouplingList.append([6, 4])
if 6 is not 5:
    if abs(6-5) == 1:
        gridCouplingList.append([6, 5])
if 6 is not 6:
    if abs(6-6) == 1:
        gridCouplingList.append([6, 6])
if 6 is not 7:
    if abs(6-7) == 1:
        gridCouplingList.append([6, 7])
if 7 is not 4:
    if abs(7-4) == 1:
        gridCouplingList.append([7, 4])
if 7 is not 5:
    if abs(7-5) == 1:
        gridCouplingList.append([7, 5])
if 7 is not 6:
    if abs(7-6) == 1:
        gridCouplingList.append([7, 6])
if 7 is not 7:
    if abs(7-7) == 1:
        gridCouplingList.append([7, 7])

gridCouplingList.append(([0, 4]))
gridCouplingList.append(([4, 0]))
gridCouplingList.append(([1, 5]))
gridCouplingList.append(([5, 1]))
gridCouplingList.append(([2, 6]))
gridCouplingList.append(([6, 2]))
gridCouplingList.append(([3, 7]))
gridCouplingList.append(([7, 3]))

gridCouplingMap = CouplingMap(gridCouplingList)

jakatraCouplingList = [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3,5], [5,3], [4,5], [5,4], [6,5], [5,6]]
jakatraCouplingMap = CouplingMap(jakatraCouplingList)
circDag = circuit_to_dag(circuit)

targetCouplingMap = squareCouplingMap

bSwap = BasicSwap(targetCouplingMap)
baseTime = time.perf_counter()
bSwap.run(circDag)
bSwapTime = time.perf_counter() - baseTime

sabreSwap = SabreSwap(targetCouplingMap)
baseTime = time.perf_counter()
sabreSwap.run(circDag)
sabreSwapTime = time.perf_counter() - baseTime

stochasticSwap = StochasticSwap(targetCouplingMap)
baseTime = time.perf_counter()
stochasticSwap.run(circDag)
stochasticSwapTime = time.perf_counter() - baseTime

lookAheadSwap = LookaheadSwap(targetCouplingMap)
baseTime = time.perf_counter()
lookAheadSwap.run(circDag)
lookAheadSwapTime = time.perf_counter() - baseTime


for i in range(200):

    # Create a noise model for the simulations
    noise_model = noise.NoiseModel()
    errorRates = list()
    qiskitErrors = list()

    for i in range(len(squareCouplingList)//2):
        errorRates.append(random.randrange(1, 10, 1)/100.0)
        qiskitErrors.append(noise.depolarizing_error(errorRates[i], 2))

    edges = targetCouplingMap.get_edges()
    uniqueEdges = set()
    for edge in edges:
        uniqueEdges.add(tuple(sorted(edge)))
    
    noiseGraph = nx.Graph()
    noiseGraph.add_nodes_from([0, 7])

    errorIdex = 0
    for edge in uniqueEdges:
        noise_model.add_quantum_error(qiskitErrors[errorIdex], ['cx'], edge)
        noiseGraph.add_edge(edge[0], edge[1], weight=1-errorRates[errorIdex])
        errorIdex += 1

    herr = HERR.HERR(targetCouplingMap, noiseGraph)
    basSwap = BasicSwap(targetCouplingMap)

    #print(gridCouplingMap)
    # Run HERR
    baseTime = time.perf_counter()
    HERRRes = herr.run(circDag)
    HERRSwapTime = time.perf_counter() - baseTime
    updatedCirc = dag_to_circuit(HERRRes)

    
    print(str(HERRSwapTime) + " " + str(bSwapTime) + " " + str(sabreSwapTime) + " " + str(stochasticSwapTime) + " " + str(lookAheadSwapTime))



