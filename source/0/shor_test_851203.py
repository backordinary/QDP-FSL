# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/TestScripts/shor_test.py
'''
This is Shor algorithm test script used to measure difference between ideal
algorithm execution and noisy environment execution.
'''

import math
import numpy as np
import time

import qiskit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor

def shor_ideal(N):
    shor = Shor(N)
    backend = Aer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(backend, shots=1024)

    start = time.time()
    result = shor.run(quantum_instance)
    end = time.time()
    print(result)

    print(f"Elapsed: {end - start} seconds")

    circ = shor.construct_circuit(False)

    ops = circ.count_ops() 

    print(ops)

    print(result['factors'][0])


def shor_noisy(N):
    provider = IBMQ.load_account()

    device = provider.get_backend('ibmq_16_melbourne')
    noise_model = NoiseModel.from_backend(device)
    shor = Shor(N)
    backend = Aer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(backend, shots=1024, noise_model=noise_model)

    properties = device.properties()
    coupling_map = device.configuration().coupling_map

    start = time.time()
    result = shor.run(quantum_instance)
    end = time.time()
    print(result)

    print(f"Elapsed: {end - start} seconds")
    circ = shor.construct_circuit(False)
    ops = circ.count_ops()

    print(ops)

    print(result['factors'][0])

IBMQ.save_account("") # Add your IBMQ API key here

print("running ideal tests")
#print("factoring 9")
#shor_ideal(9)
print("factoring 15")
shor_ideal(15)
print("factoring 21")
shor_ideal(21)
#print("factoring 27")
#shor_ideal(27)


print("running noisy tests")
#print("factoring 9")
#shor_noisy(9)
print("factoring 15")
shor_noisy(15)
print("factoring 21")
shor_noisy(21)
#print("factoring 27")
#shor_noisy(27)

