# https://github.com/shantanu-misra/Quantum_coumputing_with_Qiskit/blob/baebc362c436de5e5a3d026625f486e44d17203f/Classical%20Gates%20with%20Qubits/OR_with_qubits.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, execute


# Define quantum circuit
qc = QuantumCircuit(3,1)

for input in ['00', '01', '10', '11']:

    # Initialise all qubits to ket 0 to make life easy
    if input[0] == '1':
        qc.x(0)
    if input[1] == '1':
        qc.x(1)

    qc.cx(0,2)
    qc.cx(1,2)
    qc.ccx(0,1,2)

    qc.measure(2,0)

    job = execute(qc,Aer.get_backend('qasm_simulator'),shots=1000)
    counts = job.result().get_counts(qc)
    print("Input:", input, "Output:", counts)
