# https://github.com/jikaufman/CS239-Final-Project/blob/8765859fc63b32648ab9d934f09c4ddf2ceca752/qaoa_qiskit.py
# Jacob Kaufman 204 929 264
# Nikhil Arora 204 965 841
# QAOA (MaxSat):

import cirq
from qiskit import IBMQ, assemble, transpile
from qiskit.circuit import QuantumCircuit as qiskitqc
import requests
import math
import networkx as nx


'''
QAOA Implementation for the MaxCut problem

'''

#########################################################################################
#                                  Main Code                                            #
#########################################################################################

# QAOA implementation, n=3
def qaoa(error_correct=False):

	API_TOKEN = '7cf33cd2f0d7044af8518c33321fa74d7380d3ba58d08b271404e8226aa1f5490818ba86e1635f89e6c3cfbf10aa091ff04c1f5ff61f0f391ed780b4484e0b18'

	# initialize qiskit
	IBMQ.save_account(API_TOKEN)
	provider = IBMQ.load_account()
	print(provider.backends())
	backend = provider.backends.ibmq_16_melbourne
	
	# initialize qubits with architecture in mind
	qubits = [cirq.GridQubit(1, 4), cirq.GridQubit(2, 4),\
	          cirq.GridQubit(3, 4)]

	if error_correct:
		error_qubits = [cirq.GridQubit(3, 4), cirq.GridQubit(3, 3),\
	                    cirq.GridQubit(3, 2), cirq.GridQubit(4, 3)]

	# initialize variables
	beta = math.pi/2
	gamma = math.pi/2

	graph = nx.Graph()
	graph.add_edge(0, 1)
	graph.add_edge(1, 2)
	graph.add_edge(2, 0)

	# construct circuit
	circuit = cirq.Circuit()

	# error correction setup. error correct qubit (2,3)
	if error_correct:
		circuit.append([cirq.CNOT(qubits[2], error_qubits[1])])
		circuit.append([cirq.SWAP(error_qubits[0], error_qubits[1])])
		circuit.append([cirq.CNOT(qubits[2], error_qubits[1])])
		circuit.append([cirq.SWAP(error_qubits[0], error_qubits[1])])

	circuit.append([cirq.H(q) for q in qubits])
	
	circuit.append([cirq.ZZPowGate(exponent=2 * gamma / math.pi, global_shift=-0.5).on(qubits[0], qubits[1])])
	circuit.append([cirq.ZZPowGate(exponent=2 * gamma / math.pi, global_shift=-0.5).on(qubits[1], qubits[2])])
	circuit.append([cirq.SWAP(qubits[1], qubits[2])])
	circuit.append([cirq.ZZPowGate(exponent=2 * gamma / math.pi, global_shift=-0.5).on(qubits[0], qubits[1])])
	circuit.append([cirq.SWAP(qubits[1], qubits[2])])

	circuit.append([cirq.rx(2*beta).on_each(*qubits)])

	# error detection and correction
	if error_correct:
		circuit.append([cirq.SWAP(error_qubits[2], error_qubits[1])])
		circuit.append([cirq.CNOT(qubits[2], error_qubits[1])])
		circuit.append([cirq.SWAP(error_qubits[2], error_qubits[1])])
		circuit.append([cirq.CNOT(error_qubits[1], error_qubits[2])])
		circuit.append([cirq.SWAP(error_qubits[3], error_qubits[1])])
		circuit.append([cirq.CNOT(qubits[2], error_qubits[1])])
		circuit.append([cirq.CNOT(error_qubits[0], error_qubits[1])])
		circuit.append([cirq.SWAP(error_qubits[1], error_qubits[3])])
		circuit.append([cirq.measure(error_qubits[2]), cirq.measure(error_qubits[3])])
		circuit.append([cirq.CCNOT(qubits[2], error_qubits[1], error_qubits[0])])

	circuit.append([cirq.measure(*qubits)])

	# export to qasm
	qasm_str = circuit.to_qasm()

	# import qiskit from qasm
	qiskit_circuit = qiskitqc.from_qasm_str(qasm_str)

	# run qiskit
	transpiled = transpile(qiskit_circuit, backend)
	qobj = assemble(transpiled, backend, shots=100)
	job = backend.run(qobj)
	print(job.job_id())
	result = job.result()
	counts = result.get_counts()
	delayed_result = backend.retrieve_job(job.job_id()).result()
	delayed_counts = delayed_result.get_counts()
	print(counts)
	print(delayed_counts)

# Main loop
if __name__ == "__main__":

	response = qaoa()

