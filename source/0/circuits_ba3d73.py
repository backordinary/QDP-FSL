# https://github.com/anderseline/QuantumComputingBadge/blob/16703e117843cbcf0270d864b7858728829e16b5/quantum_algorithms/Circuits.py
from qiskit import visualization, QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import random

def getBloch(quantumC, backend):
    return visualization.plot_bloch_multivector(
        execute(quantumC, backend).result().get_statevector()
    )

def stateGenerator(state):
    state = state[::-1]
    current = int(state, 2)
    newState = [0 for i in range(0, 2**(len(state)))]
    newState[current] = 1
    return newState

def getMeasure(circ, n, backend):
    measures = []
    for i in range(0, n):
        outcome = execute(circ, backend).result()
        hist = outcome.get_counts()
        for i in hist.keys():
            measures.append(i)
        return measures
    
def makeCirc(n, m):
    initState = ''
    for _ in range(0, n-m):
        initState += str(random.randint(0, 1))
    for _ in range(0, m):
        initState += str(0)
    print(initState)
    qr = QuantumRegister(n)
    qc = ClassicalRegister(n)
    circ = QuantumCircuit(qr, qc)
    initializedState = stateGenerator(initState)
    circ.initialize(initializedState, qr)
    return circ

def makeCirc2(n, m, o, inp):
    initState = inp
    for _ in range(0, m):
        initState += str(0)
    for _ in range(0, o):
        initState += str(1)
    print(initState)
    qr = QuantumRegister(n)
    qc = ClassicalRegister(n)
    circ = QuantumCircuit(qr, qc)
    initializedState = stateGenerator(initState)
    circ.initialize(initializedState, qr)
    return circ

def makeCirc3(n, m, o, inp=None):
    initState = ''
    for i in range(0, o):
        initState += str(1)
    if inp != None:
        initState = inp
    else:
        for i in range(0, n-o-m):
            initState += str(random.randint(0, 1))
    for i in range(0, m):
        initState += str(0)
    print(initState)
    qr = QuantumRegister(n)
    qc = ClassicalRegister(n)
    circ = QuantumCircuit(qr, qc)
    initializedState = stateGenerator(initState)
    circ.initialize(initializedState, qr)
    return circ