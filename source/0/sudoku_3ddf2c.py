# https://github.com/erenaykrcn/qiskit-projects/blob/dcf0be7a104762bc53e73e5443fe491ef2a0c2ce/sudoku/sudoku.py
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.visualization import plot_histogram
from oracle_sudoku import sudoku_three_by_three_oracle

import sys
sys.path.append("../grover")
from oracle import diffuser

from math import floor
import numpy as np

def sudoku_grover_three_by_three(positions_list):
	qc = QuantumCircuit(13, 6)
	qc.h(12)
	qc.z(12)

	marked = 2
	n = 6
	m = 13

	oracle = sudoku_three_by_three_oracle(positions_list)

	r = range(int(floor(np.pi/(4*np.arcsin(np.sqrt(marked/2**(n))))-0.5)))

	qc.h(range(n))

	for _ in r:
		qc.append(oracle, range(m))
		qc.append(diffuser(n), range(n))

	qc.measure(range(n), range(n))

	qasm_sim = Aer.get_backend('qasm_simulator')
	count = execute(qc, qasm_sim, shots=1000).result().get_counts()

	fig = plot_histogram(count)
	fig.savefig(str(positions_list)+".png")

	return count


all_solutions = []
l = list(range(3))
l.remove(0)
for j in l:
	position = [0, j+3, 9-0-j]
	counts = sudoku_grover_three_by_three(position)
	solutions = [k for k, v in counts.items() if v > 250]

	for solution in solutions:
		bitstring = ""
		added_twos = 0
		for bit_index in range(9):
			if bit_index in position:
				bitstring += "2"
				added_twos = added_twos + 1
			else:
				bitstring += solution[bit_index - added_twos]
		all_solutions.append(bitstring)
l = list(range(3))
l.remove(1)
for j in l:
	position = [1, j+3, 9-1-j]
	counts = sudoku_grover_three_by_three(position)
	solutions = [k for k, v in counts.items() if v > 250]

	for solution in solutions:
		bitstring = ""
		added_twos = 0
		for bit_index in range(9):
			if bit_index in position:
				bitstring += "2"
				added_twos = added_twos + 1
			else:
				bitstring += solution[bit_index - added_twos]
		all_solutions.append(bitstring)
l = list(range(3))
l.remove(2)
for j in l:
	position = [2, j+3, 9-2-j]
	counts = sudoku_grover_three_by_three(position)
	solutions = [k for k, v in counts.items() if v > 250]

	for solution in solutions:
		bitstring = ""
		added_twos = 0
		for bit_index in range(9):
			if bit_index in position:
				bitstring += "2"
				added_twos = added_twos + 1
			else:
				bitstring += solution[bit_index - added_twos]
		all_solutions.append(bitstring)

print(all_solutions)
