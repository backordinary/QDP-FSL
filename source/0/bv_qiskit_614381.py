# https://github.com/jikaufman/CS239-Final-Project/blob/8765859fc63b32648ab9d934f09c4ddf2ceca752/bv_qiskit.py
# Jacob Kaufman 204 929 264
# Nikhil Arora 204 965 841
# Bernstein-Vazirani Problem:

import cirq
from qiskit import IBMQ, assemble, transpile
from qiskit.circuit import QuantumCircuit as qiskitqc
import requests

'''
Input: a function f: {0,1}^n → {0,1}.

Assumption: f(x) = a*x+b

Output: a,b

Notation: {0,1}^n is the set of bit strings of length n, a is an unknown bit string
of length n, * is inner product mod 2, + is addition mod 2, and b is an unknown single bit
'''

#########################################################################################
#                                  Main Code                                            #
#########################################################################################

# constructs a bernstein-vazirani circuit
# example circuit: a = 0101101, b = 0
def bernstein(error_correct=False):

	API_TOKEN = '7cf33cd2f0d7044af8518c33321fa74d7380d3ba58d08b271404e8226aa1f5490818ba86e1635f89e6c3cfbf10aa091ff04c1f5ff61f0f391ed780b4484e0b18'

	# initialize qiskit
	IBMQ.save_account(API_TOKEN)
	provider = IBMQ.load_account()
	print(provider.backends())
	backend = provider.backends.ibmq_16_melbourne

	# initialize qubits with architecture in mind
	qubits = [cirq.GridQubit(0, 5), cirq.GridQubit(1, 4),\
	          cirq.GridQubit(0, 6), cirq.GridQubit(2, 5),\
	          cirq.GridQubit(2, 3), cirq.GridQubit(1, 5),\
	          cirq.GridQubit(3, 4), cirq.GridQubit(2, 4)]

	if error_correct:
		error_qubits = [cirq.GridQubit(3, 4), cirq.GridQubit(3, 3),\
	                    cirq.GridQubit(3, 2), cirq.GridQubit(4, 3)]

	# construct circuit
	circuit = cirq.Circuit()

	# error correction setup. error correct qubit (2,3)
	if error_correct:
		circuit.append([cirq.CNOT(qubits[2], error_qubits[1])])
		circuit.append([cirq.SWAP(error_qubits[0], error_qubits[1])])
		circuit.append([cirq.CNOT(qubits[2], error_qubits[1])])
		circuit.append([cirq.SWAP(error_qubits[0], error_qubits[1])])

	# hadamards
	circuit.append([cirq.H(q) for q in qubits])

	# turn helper qubit to 1
	circuit.append([cirq.Z(qubits[7])])

	# oracle
	circuit.append([cirq.CNOT(qubits[1], qubits[7])])
	circuit.append([cirq.CNOT(qubits[3], qubits[7])])
	circuit.append([cirq.CNOT(qubits[4], qubits[7])])
	circuit.append([cirq.CNOT(qubits[6], qubits[7])])

	# hadamards
	circuit.append([cirq.H(q) for q in qubits[:-1]])

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

	# measure
	circuit.append([cirq.measure(q) for q in qubits[:-1]])

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

	return requests.post(url, json=job_payload)


if __name__ == '__main__':
	response = bernstein()
	print(response.text)
	
