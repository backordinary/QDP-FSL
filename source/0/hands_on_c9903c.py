# https://github.com/LeeFarrugia/Quantum-Summer-School/blob/fc5b0647f7da3b79cf2e73463b247c71db5e2134/Hands-On.py
import numpy as np
from pytket.extensions.qiskit import AerStateBackend, IBMQEmulatorBackend
from pytket import Circuit
from qiskit import IBMQ

# warm up question 1
def build_ghz_circuit(x):
    circ = Circuit(x)
    circ.H(0)
    for i in range(x-1):
        circ.CX(0, i+1)
    return circ
    
input = int(input('Please enter the amount of qubits to be used: '))
x = build_ghz_circuit(input)

backend = AerStateBackend()
compiled_circ = backend.get_compiled_circuit(x)
state = backend.run_circuit(compiled_circ).get_state()
print(state.round(5))

backend = IBMQEmulatorBackend('ibmq_belem', hub='ibm-q', group='open', project='main')
print(state.round(5))
