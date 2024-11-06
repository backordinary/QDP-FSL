# https://github.com/mcrl/quantum-benchmark/blob/37eb9e4256b08baabc97ef86992de9166211ba66/abstract/swap.py
import numpy as np
from qiskit import QuantumCircuit


def swap(circ, x, y): 
	circ.cx(x, y)
	circ.cx(y, x)
	circ.cx(x, y)
	

if __name__ == "__main__":
	num_qubits = 2
	circ = QuantumCircuit(num_qubits)

	circ.x(0)
	circ.h(0)

	circ.swap(circ, 0, 1)
	print(circ)
