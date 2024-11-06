# https://github.com/esfinkel/quantum_artist/blob/cd40106bc7fb555667d21c6beed3be40176d213c/four_bit_quantum_rng.py
# with thanks to Russell Huffman of https://medium.com/qiskit/rothko-inspired-generative-quantum-art-6f34ca9d17cb
# for providing the inspiration for the project

# Importing standard Qiskit libraries and configuring account
from collections import defaultdict
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.test.mock import FakeVigo
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

##################################################

#paste qiskit code from website here
qreg_q4 = QuantumRegister(4, 'q')
creg_c4 = ClassicalRegister(4, 'c')
circuit4 = QuantumCircuit(qreg_q4, creg_c4)

circuit4.h(qreg_q4[0])
circuit4.h(qreg_q4[1])
circuit4.h(qreg_q4[2])
circuit4.ccx(qreg_q4[0], qreg_q4[1], qreg_q4[3])
circuit4.measure(qreg_q4[0], creg_c4[0])
circuit4.measure(qreg_q4[1], creg_c4[1])
circuit4.measure(qreg_q4[2], creg_c4[2])
circuit4.measure(qreg_q4[3], creg_c4[3])

#circuit4.draw()
################################################

def generate_data(circuit):
    # Loading your IBM Q account(s)
    # provider = IBMQ.load_account() # uncomment if using a real backend

    # pick a device to run on
    # backend = provider.get_backend('ibmq_santiago')
    backend = FakeVigo()

    # Execute the circuit on the backend
    job = execute(circuit4, backend, shots=1000, memory=True)

    # Grab results from the job
    result = job.result()

    #get individual shots
    outputArray = result.get_memory()

    res = defaultdict(int)
    for x in outputArray:
        res[x] += 1

    return outputArray

    # print(outputArray)
    # print(res)
