# https://github.com/Salvo1108/Quantum_Programming_ASService/blob/b80462df050218490e521ea16b4ac88008b242b1/gateway/simons.py
# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, transpile, assemble

# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle

def s_algorithm(b, n):
    n = len(b)
    simon_circuit = QuantumCircuit(n * 2, n)

    # Apply Hadamard gates before querying the oracle
    simon_circuit.h(range(n))

    # Apply barrier for visual separation
    simon_circuit.barrier()

    simon_circuit += simon_oracle(b)

    # Apply barrier for visual separation
    simon_circuit.barrier()

    # Apply Hadamard gates to the input register
    simon_circuit.h(range(n))

    # Measure qubits
    simon_circuit.measure(range(n), range(n))

    return simon_circuit
