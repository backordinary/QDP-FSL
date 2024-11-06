# https://github.com/ronf6co/ClassiqChallenge/blob/25e571f8262ce0883be2e1b4b5467511d7abbea8/toffoli/toffoli.py
# ----------------------------------------------------------------------------------------------------------------------
# Created By  : Ron Cohen
# Created Date: 11/05/2022
# ----------------------------------------------------------------------------------------------------------------------
# BACKGROUND
# Many quantum operations include multi-controlled Toffoli (MCX) gates.
# Among the most notable are: Grover Operator, logical AND operator, various state preparation algorithms,
# and arithmetic comparators.
#
# This task focuses on the implementation of the MCX gate with a limited qubit count and circuit depth.
#
# THE PROBLEM
# Decompose an MCX gate with 14 control qubits into single-qubit and double-qubit CX gates.
# You may use up to 5 clean auxiliary qubits and should release (uncompute) them at the end of the circuit.
# Thus, the circuit can use no more than 20 total qubits:
#   14 control qubits,
#   one target qubit,
#   and up to five auxiliary qubits.
#
# Important: Make sure that when you submit a circuit, the order of the qubits is indeed as follows:
#   first the 14 control qubits,
#   then the target qubit,
#   then any auxiliary.
# This will help us validate the circuit.
#
# METRIC
# The winning solution will achieve this using a circuit of minimal depth.
# ----------------------------------------------------------------------------------------------------------------------

from qiskit import transpile
from qiskit import QuantumCircuit, QuantumRegister

# Init Circuit
qc = QuantumCircuit()
q_con = QuantumRegister(14, 'q_c')
q_t = QuantumRegister(1, 'q_t')
q_ancilla = QuantumRegister(5, 'q_a')
qc.add_register(q_con)
qc.add_register(q_t)
qc.add_register(q_ancilla)

# MCX c0,c1,c2 -> a0
# MCX c3,c4,c5 -> a1
# MCX c6,c7,c8 -> a2
qc.mcx([q_con[0],q_con[1],q_con[2]],q_ancilla[0])
qc.mcx([q_con[3],q_con[4],q_con[5]],q_ancilla[1])
qc.mcx([q_con[6],q_con[7],q_con[8]],q_ancilla[2])

qc.barrier()

# MCX a0,c9 ,c10 -> a3
# MCX a1,c11,c12 -> a4
qc.mcx([q_ancilla[0],q_con[9],q_con[10]],q_ancilla[3])
qc.mcx([q_ancilla[1],q_con[11],q_con[12]],q_ancilla[4])

qc.barrier()

# CCX a3,a4 -> t
qc.ccx(q_ancilla[3],q_ancilla[4],q_t)

print("total depth is: " , qc.decompose().depth())

qc.decompose().draw()


