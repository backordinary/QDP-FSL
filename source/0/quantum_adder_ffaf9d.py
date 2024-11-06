# https://github.com/sami-b95/Qiskit-Toolbox/blob/14a378f4bb3594a441ae0a2f7e0af59dd4c07adf/QuantumAddition/Quantum_adder.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import os
os.chdir('../QFT')
from QFT import quantum_fourier_transform
os.chdir('../QuantumAddition')
from math import pi

### TO BE FINISHED

def quantum_adder(circuit, to_add_qubits, target_qubits):
    n_qubits = len(target_qubits)
    if n_qubits != len(to_add_qubits):
        raise NameError('The two quantum registers have not the same size.')
    else:
        circuit = quantum_fourier_transform(circuit, target_qubits)
        for i in range(n_qubits):
            for distance in range(i + 1): 
                circuit.cu1(pi / 2**distance, to_add_qubits[i - distance], target_qubits[n_qubits - i - 1])
        circuit = quantum_fourier_transform(circuit, target_qubits, inverse=True)
        return circuit


to_add_qubits = QuantumRegister(2)
target_qubits = QuantumRegister(2)
clbits = ClassicalRegister(4)

circuit = QuantumCircuit(target_qubits, clbits)
circuit.add_register(to_add_qubits)


circuit.x(target_qubits[0])  #NOT WORKING !! Have to handle the problem (probably algorithm problem)
#circuit = quantum_adder(circuit, to_add_qubits, target_qubits)
circuit.measure(circuit.qubits, clbits)

backend = Aer.get_backend('qasm_simulator')

job_sim = execute(circuit, backend, shots=10000)

sim_result = job_sim.result()
print(sim_result.get_counts(circuit))