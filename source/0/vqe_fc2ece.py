# https://github.com/stiandb/Thesis/blob/c3f3243fdeed6a099a16006d2bf529e2ceedc2b4/VQE/VQE.py
import qiskit as qk 
import numpy as np
from scipy.optimize import minimize
from utils import *
from inspect import getfullargspec


class VQE:
	def __init__(self,hamiltonian_list,ansatz,n_qubits,ancilla=0,shots=1000,seed_simulator=None,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=None,basis_gates=None,max_energy=False,transpile=False,seed_transpiler=None,optimization_level=1,coupling_map=None,error_mitigator=None,print_energies=False,get_variance=False):
		"""
		Inputs:
			hamiltonian_list (list) - List containing each term of the hamiltionian
			ansatz (function) - A function which accepts theta,circuit and registers as its arguments.
								theta should be the parameters to solve for, circuit is the quantum circuit
								registers is a list of qiskit registers where the register to perform ansarz on is the first entry
								and the classical register is the last.
			n_qubits (int)    - The number of qubits in the ansatz register
			ancilla (int)     - Some ansatzes requires ancilla register. This int specifies the number of ancilla qubits
			shots (int) 	  - Specifies the number of times a circuit is run to evaluate expectation value
			seed_simulator  (NoneType or int)       - Seed for circuit measurement.
			backend - 			backend for qiskit execute function
			noise_model - 		Noise model when simulating noise
			basis_gates 		- Qiskit basis gates
			coupling_map 		-Coupling map for qiskit explaining connectivity between qubits
			max_energy (boolean) 		  - If False (default), the minimum energy is approximated. The maximum
											energy is approximated if True.
			transpile (boolean) - If True, transpiler is used
			seed_transpiler (int) - The seed to use for the transoiler
			optimization_level (int) - The optimization level for the transpiler. 0 is no optimization,
										3 is the heaviest optimization
			error_mitigator (functional) - Returns filter for error correction
			get_variance (boolean) - If True, the variance of the energy expectation value is measured
			print_energies (boolean) - If True, the energy expectation is printed each evaluation

		"""
		self.hamiltonian_list = hamiltonian_list
		self.seed_simulator = seed_simulator
		self.backend=backend
		self.ansatz = ansatz
		self.n_qubits = n_qubits
		self.shots = shots
		self.max_energy = max_energy
		self.ancilla=ancilla
		self.noise_model=noise_model
		self.basis_gates = basis_gates
		self.transpile=transpile
		self.seed_transpiler=seed_transpiler
		self.optimization_level=optimization_level
		self.coupling_map=coupling_map
		spec = getfullargspec(self.ansatz)
		if 'classical_bits' in spec[0]:
			self.set_classical_bits = True
		else:
			self.set_classical_bits = False
		self.energies = []
		self.energies_regularized = []
		self.error_mitigator = error_mitigator
		self.print_energies=print_energies
		self.theta = None
		self.h2 = None
		self.get_variance = get_variance


	def measure_variance(self,theta,E):
		hamiltonian_list=self.hamiltonian_list
		if self.h2 is None:
			h2 = []
			for i in range(len(hamiltonian_list)):
				for j in range(len(hamiltonian_list)):
					temp = operator_product(hamiltonian_list[i],hamiltonian_list[j])
					h2.append(temp)
			self.h2 = h2
		self.print_energies=False
		self.get_variance = False
		E2 = self.expectation_value(theta,self.h2)
		self.get_variance = True
		self.print_energies=True
		var = E2 - E**2
		return(var)



	def expectation_value(self,theta,hamiltonian_list = None):
		"""
		Calculates expectation values and adds them together
		"""
		E = 0
		circuit,registers = None, None
		if hamiltonian_list is None:
			h_l = self.hamiltonian_list
		else: 
			h_l = hamiltonian_list
		for pauli_string in h_l:
			factor = pauli_string[0]
			if factor == 0:
				continue
			classical_bits = len(pauli_string[1:])
			if classical_bits == 0:
				E += factor
				continue
			if self.set_classical_bits:
				circuit,registers = self.ansatz(theta,circuit,registers,classical_bits)
			else:
				circuit,registers = initialize_circuit(self.n_qubits,self.ancilla,classical_bits)
				circuit,registers = self.ansatz(theta,circuit,registers)
			qubit_list = []
			for qubit,gate in pauli_string[1:]:
				circuit,registers = pauli_expectation_transformation(qubit,gate,circuit,registers)
				qubit_list.append(qubit)
			E += measure_expectation_value(qubit_list,factor,circuit,registers,seed_simulator=self.seed_simulator,backend=self.backend,shots=self.shots,noise_model=self.noise_model,basis_gates=self.basis_gates,transpile=self.transpile,optimization_level=self.optimization_level,seed_transpiler=self.seed_transpiler,coupling_map=self.coupling_map,error_mitigator=self.error_mitigator)
			if not self.seed_simulator is None:
				self.seed_simulator += 1
		if self.max_energy:
			E = -E
		if self.print_energies:
			if hamiltonian_list is None:
				print('<E> = ', E)
		if self.get_variance:
			print(r'<E2> = ', np.real(self.measure_variance(theta,E)))
		return(E)
	


	def classical_optimization(self,theta,method='Powell',options=None):
		"""
		Performs a classical optimization method to find the optimal parameters.
		This function is used directly after initialization of class.
		Input:
			theta (numpy array) - 1D array with the parameters to optimize
			method (str)		- String specifying the optimization method to use.
								  See scipy.optimize minimize documentation for several options
			max_iters (int) 	- Maximum number of iterations for optimization method
		Output:
			theta (numpy array) - The optimal parameters  
		"""
		self.optimization = True
		result = minimize(self.expectation_value,theta,method=method,options=options)
		theta = result.x
		self.theta = np.array([theta]) if len(theta.shape) == 0 else theta 
		self.energies =np.array(self.energies)
		self.energies_regularized = np.array(self.energies_regularized)
		self.optimization=False
