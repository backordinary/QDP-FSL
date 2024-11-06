# https://github.com/vaap96/sym_breaking/blob/6a52d51b3d2263fe9dc3f744e26d06238e95284a/qaoa_opt.py
import numpy as np
import math
import sys
import ham_rep
import openfermion as of
import scipy.linalg as la
import warnings
import proj 
import group_rep
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

qubits = 4
p = 4
jx = -0.5
jy = -1
jz = -1
cyclic = group_rep.cyclic(qubits)[0]
ham_pb = of.get_sparse_operator(ham_rep.heisenberg_hamiltonian(qubits, jx, jy, jz, periodic = True)).toarray()
proj_c = proj.projectors(cyclic, qubits)

def invert_counts(counts):
	
	'''
	Invert the bitsrting order because the order that qiskit returns is reversed
	and returns the inversed bitstring.
	
	Args:
		counts (str): the bitstring that we want to change its order.
	'''
	
	return {k[::-1]:v for k, v in counts.items()}

def converter(bitstring, qubits):
	
	'''
	Converts the returned bitstring to a full statevector that we have to use 
	for the computation of the expectation value.
	
	Args:
		bitstring (str): bitstring that we want to transform.
		qubits (int): the number of qubits that the Hamiltonian acts on.
	'''
	
	to_int = int(bitstring, 2)
	res = np.zeros(2**qubits)
	res[to_int] = 1

	return res
	
def calc_obj(psi, ham, qubits):
	
	'''
	Function that calculates the objective value that we want to optimise. 
	Works both for bitsrting results and for full statevectors.

	Args: 
		psi (str or np.array): the value we get after the implementation of the 
			quantum circuit.
		ham (np.array): the Hamiltonian that we want to approximate its ground 
			state.
		qubits (int): the number of qubits that the Hamiltonian acts on.
	'''
	
	if type(psi) == 'str':
		psi = converter(psi, qubits)
		obj = psi.transpose().conj() @ ham @ psi
	else:
		obj = psi.transpose().conj() @ ham @ psi
	
	return obj

def expectation(circ, ham):
	
	'''
	Function that calculate the expectation of value when the quantum circuit 
	returns specific bitstring after measurement. Calculates the average 
	expectation value based on the probabilities to get a specific bitstring 
	after the measurement. 
	
	Args: 
		circ (dict): the result after the measurement of the quantum circuit.
		ham (np.array): the Hamiltonian that we want to approximate its ground 
			state.
	'''	

	total_exp = 0
	total_counts = 0
	
	for state, count in circ.items():
		obj = calc_obj(state, ham, qubits)
		total_exp += obj * count
		total_counts += count	
	
	return total_exp / total_counts 

def expectation_statevector(state, ham):
	
	'''
	If the quantum circuit returns a statevector then this function is used for 
	the calculation of the expectation value. Qiskit function is used.
	
	Args: 
		state (statevector): result that we get from the quantum circuit without
			the measurement.
		ham (np.array): the Hamiltonian that we want to approximate its ground
			state.
	'''
	expect = state.data.transpose().conj() @ ham_pb @ state.data	
	
	return expect

def cr_qaoa_circ(qubits, beta, gamma, layers, sb = False, sbt = False):
	
	'''
	Function that creates and implements the quantum circuit that is necessary
	for the QAOA procedure.
	
	Args:
		qubits (int): number of qubits that Hamiltonian acts on.
		beta (np.array): parameters of the mixing Hamiltonian.
		gamma (np.array): parameters of the problem Hamiltonian.
		layers (int): number of layers for the quantum circuit.
		sb (binary): if we break the symmetries for the evolution Hamiltonian. 
		sbt (binary): denotes if we will only use the symmetry breaking term for 
			the evolution.
	'''
	
	qc_qaoa = QuantumCircuit(qubits)
	
	# Initial state
	init = np.zeros(2**qubits)
	init[2**qubits - 1] = 1	
	qc_qaoa.initialize(init, qc_qaoa.qubits)

	for irep in range(0, layers):
		if sbt == False:
		
			# Problem Hamiltonian
			for i in range(qubits - 1):
				qc_qaoa.rzz(2 * jz * gamma[irep], i, i+1)
				qc_qaoa.rxx(2 * jx * gamma[irep], i, i+1)
				qc_qaoa.ryy(2 * jy * gamma[irep], i, i+1)
		
			if sb == False:		
				qc_qaoa.rzz(2 * jz * gamma[irep], 0, -1)
				qc_qaoa.rxx(2 * jx * gamma[irep], 0, -1)
				qc_qaoa.ryy(2 * jy * gamma[irep], 0, -1)
		else:
			# Problem Hamiltonian	
			qc_qaoa.rzz(2 * jz * gamma[irep], 0, -1)
			qc_qaoa.rxx(2 * jx * gamma[irep], 0, -1)
			qc_qaoa.ryy(2 * jy * gamma[irep], 0, -1)	
		
		# Mixing Hamiltonian
		for i in range(0, qubits):
			qc_qaoa.rz(2 * beta[irep], i)

	#qc_qaoa.measure_all()
	
	return qc_qaoa	

def opt_exp(theta, ham, qubits, p):
	
	'''
	Function that is used for the creation of the quantum circuit in order to be 
	used for the classical optimisation procedure. We also define the simulator 
	that we are going to use for the quantum circuit. Second function inside the 
	main function that implements a black box function that returns the 
	expectation value from the circuit and then is used for the classical 
	optimisation procedure.

	Args: 
		qubits (int): the number of qubits that the Hamiltonian acts on.
		ham (np.array): the Hamiltonian that we want to approximate its ground
			state.
		p (int): the number of layers for the quantum circuit.
		theta (np.array): the parameters that we want to optimise. 
	'''
	
	global proj_fidel

	backend = Aer.get_backend('statevector_simulator')
	
	beta = theta[:p]
	gamma = theta[p:]	
		
	qc = cr_qaoa_circ(qubits, beta, gamma, p, sb = True, sbt = True)
	res_circ = execute(qc, backend, seed_simulator=10).result().get_statevector()
	
	for pr in proj_c.values():
		proj_fidel.append(res_circ.data.transpose().conj() @ pr @ res_circ.data)
		
	return res_circ.data.transpose().conj() @ ham @ res_circ.data 

init_params = np.random.uniform(-np.pi,np.pi,size=2*p)
time = [i for i in list(np.arange(0, 8, 0.2))]

#proj_fidel = []

#opt_exp(init_params, ham_pb, qubits, p)

break_fidel = []

qc_qaoa = QuantumCircuit(qubits)
	
# Initial state
init = np.zeros(2**qubits)
init[2**qubits - 1] = 1	
qc_qaoa.initialize(init, qc_qaoa.qubits)



backend = Aer.get_backend('statevector_simulator')
		
qc = cr_qaoa_circ(qubits, beta, gamma, p, sb = True, sbt = True)
res_circ = execute(qc, backend, seed_simulator=10).result().get_statevector()
	
for pr in proj_c.values():
	proj_fidel.append(res_circ.data.transpose().conj() @ pr @ res_circ.data)


#resf = minimize(
#			opt_exp,
#            x0 = init_params,
#			args = (ham_pb, qubits, p),
#            method='COBYLA', 
#            options={'maxiter': 5000})
#
#print(len(proj_fidel))
#print(resf)
