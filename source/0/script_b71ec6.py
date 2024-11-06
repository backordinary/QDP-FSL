# https://github.com/QiyangGeng/QHackOH/blob/adbf7d0351110171f39d686a3bdd31d1fb63835a/script.py
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info.operators import Operator
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import qiskit.quantum_info as qi

# bus = ClassicalRegister(1)
quantumReg = QuantumRegister(3)
classicReg = ClassicalRegister(3)
circuit = QuantumCircuit(quantumReg, classicReg)

# Input
circuit.u(np.pi/2, np.pi/2, np.pi/2)

# Bell state
circuit.x(2)
Matrix = Operator([
    [1, 0, 0, 0],
    [0, 1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [0, 1j/np.sqrt(2), 1/np.sqrt(2), 0],
    [0, 0, 0, 1]])
circuit.unitary(Matrix, [0, 1])
QuantumCircuit.decompose(circuit)
circuit.rz(np.pi/2, 2)

# Cloning


# Measurement
circuit.h(quantumReg[0])
circuit.cx(quantumReg[0], quantumReg[0 + 1])
circuit.cx(quantumReg[1], quantumReg[1 + 1])

# Get ideal output state
target_state_bell5 = qi.Statevector.from_instruction(circuit)

# Generate circuits and run on simulator
qst_bell5 = state_tomography_circuits(circuit, quantumReg)
job = qiskit.execute(qst_bell5, Aer.get_backend('qasm_simulator'), shots=5000)

# Extract tomography data so that counts are indexed by measurement configuration
tomo_bell5 = StateTomographyFitter(job.result(), qst_bell5)

circuit.measure(0, 0)
circuit.measure(1, 1)
circuit.measure(2, 2)
