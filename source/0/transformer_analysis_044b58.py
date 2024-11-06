# https://github.com/edyounis/qiskit-hackathon/blob/4d61cfa30b8fe92d27ae077eda896df384deab8b/transformer_analysis.py
#!/bin/python
# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.tools.qi.qi import random_unitary_matrix
from qiskit import BasicAer
from qiskit import execute
from qiskit import IBMQ
from qiskit.providers.aer.noise.device.models import basic_device_noise_model 
from qiskit.quantum_info.operators._measures import process_fidelity
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.extensions.standard import U3Gate
from qiskit.mapper import two_qubit_kak

import random
from math import pi
import numpy as np
from copy import deepcopy
from itertools import permutations

# Goal: Evaluate circuit transformers (compilers, etc) using a model
#	   that measures correctness of final output when run on potential,
#	   noisy intermediate-scale quantum devices.
def evaluate_transformer ( A, circuit_set = None ):

	# Set Random seed
	random.seed(0)
	np.random.seed(0)

	# Circuit set is a n-vector of circuits
	circuit_set = circuit_set or generate_circuit_set()

	# Overall average calculation
	overall_fidelity_sum = 0.

	for circuit in circuit_set:

		# Calculate ideal unitary
		ideal_result = simulate( circuit )

		# Transform circuit using input transformer
		transformed_circuit = A( circuit )

		# Injecting noise returns a set of circuits
		noisy_circuit_set = inject_noise( transformed_circuit )

		# Average the fidelity of all noise injected circuits
		fidelity_sum = 0.

		for noisy_circuit in noisy_circuit_set:
			noisy_results = simulate( noisy_circuit )
			fidelity_sum += compare_unitaries( ideal_result, noisy_results )

		fidelity_avg = fidelity_sum / float( len( noisy_circuit_set ) )

		overall_fidelity_sum += fidelity_avg

	overall_fidelity_avg = overall_fidelity_sum / float( len( circuit_set ) )
	return overall_fidelity_avg


# Generate a random set of circuits
def generate_circuit_set ( ):
	return [ build_model_circuit( 3, 1 ) for i in range( 10 ) ]

unitary_backend = BasicAer.get_backend( 'unitary_simulator' )

# Unitary Simulator that returns the unitary matrix representation for a quantum program
def simulate ( circuit ):
	unitary_job	 = execute( circuit, unitary_backend )
	unitary_results = unitary_job.result()
	unitary_matrix  = unitary_results.get_unitary( circuit )
	return unitary_matrix

# Injects Noise
def inject_noise ( circuit ):
	circuit_set = []

	for i in range( circuit.depth() + 1 ):
		circuit_prime = deepcopy( circuit )

		for qubit in circuit_prime.qregs[0]:
			error = sample_error_opt()
			circuit_prime.data.insert( i, U3Gate( error[0], error[1], error[2], qubit, circuit_prime ) )
		circuit_set.append( circuit_prime )
	return circuit_set

# Calculates process fidelity
def compare_unitaries ( U1, U2 ):
	fid_max = float('-inf')
	for i in permutations( U1 ):
		fid = process_fidelity( i, U2 )
		if fid >= fid_max:
			fid_max = fid
	return fid_max
	#overlap = np.trace(np.dot(U1.conj().transpose(), U2))
	#return abs(overlap) / (len(U1)**2)

# Generate Random Circuit
def build_model_circuit(width=3, depth=None):
	"""Create quantum volume model circuit on quantum register qreg of given
	depth (default depth is equal to width) and random seed.
	The model circuits consist of layers of Haar random
	elements of U(4) applied between corresponding pairs
	of qubits in a random bipartition.
	"""
	qreg = QuantumRegister( width, "q" )
	depth = depth or width

	circuit = QuantumCircuit( qreg )

	for _ in range(depth):
		# Generate uniformly random permutation Pj of [0...n-1]
		perm = np.random.permutation(width)

		# For each pair p in Pj, generate Haar random U(4)
		# Decompose each U(4) into CNOT + SU(2)
		for k in range(width // 2):
			U = random_unitary_matrix(4)
			for gate in two_qubit_kak(U):
				qs = [qreg[int(perm[2 * k + i])] for i in gate["args"]]
				pars = gate["params"]
				name = gate["name"]
				if name == "cx":
					circuit.cx(qs[0], qs[1])
				elif name == "u1":
					circuit.u1(pars[0], qs[0])
				elif name == "u2":
					circuit.u2(*pars[:2], qs[0])
				elif name == "u3":
					circuit.u3(*pars[:3], qs[0])
				elif name == "id":
					pass  # do nothing
				else:
					raise Exception("Unexpected gate name: %s" % name)
	return circuit

def sample_error_opt ( ):
	x_rotation = random.gauss( 0, 0.05 )
	y_rotation = random.gauss( 0, 0.05 )
	return Optimize1qGates.compose_u3( x_rotation, -pi/2, pi/2, y_rotation, 0, 0 )
