# https://github.com/rohansk1/rohan_qcbm/blob/7ceb78dfa463431c911a06378aa249c94631c42a/qcbm.py
import qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, transpile
import numpy as np
from qiskit.algorithms.optimizers import NELDER_MEAD, ADAM, GradientDescent, L_BFGS_B


# angles is a 1d list, layers is a number
# angles is of size 3dn+3n = 3n(d+1)

def create_sim_circuit(n, angles, layers):

	circuit = QuantumCircuit(n,n)

	for layer in range(layers):

		# Mixing Layer
		for qubit in range(n):
			circuit.rz(angles[n*3*layer + 3*qubit], qubit)
			circuit.rx(angles[n*3*layer + 3*qubit+1], qubit)
			circuit.rz(angles[n*3*layer + 3*qubit+2], qubit)

		# Connections Layer
		for qubit in range(n):
			circuit.cx(qubit, (qubit+1) % n)

	# Final mixing layer
	for qubit in range(n):
		circuit.rz(angles[n*3*layers+3*qubit], qubit)
		circuit.rx(angles[n*3*layers+3*qubit+1], qubit)
		circuit.rz(angles[n*3*layers+3*qubit+2], qubit)

	for i in range(n):
		circuit.measure(i,i)

	return circuit


def norm_dict(dct):
	total_val = 0
	total_val = sum(dct.values())

	for elem in dct.keys():
		dct[elem] = dct[elem]/total_val

	return dct

def get_KL(model_dist_original, training_dist_original):
	epsilon = 1e-16
	KL = 0
	model_dist = {}
	training_dist = {}

	for elem in sorted(training_dist_original.keys()):
		model_dist[elem] = model_dist_original[elem]
		training_dist[elem] = training_dist_original[elem]

	for elem in training_dist.keys():
		if model_dist[elem] == 0:
			model_dist[elem] += epsilon
		if training_dist[elem] == 0:
			training_dist[elem] += epsilon

	model_dist = norm_dict(model_dist)
	training_dist = norm_dict(training_dist)

	#technically not normalized at this point but by a very small amount (~1e-10)

	for elem in training_dist.keys():
		KL += training_dist[elem] * np.log(training_dist[elem]/model_dist[elem])

	return KL

def get_dict_from_counts(counts):
	ret_dict = {}

	for elem in counts:
		if elem in ret_dict.keys():
			ret_dict[elem] += 1
		else:
			ret_dict[elem] = 1

	total_val = 0
	for val in ret_dict.values():
		total_val += val

	for elem in ret_dict.keys():
		ret_dict[elem] = ret_dict[elem]/total_val

	return ret_dict

# genbin function stolen from stackoverflow.com/questions/64890117 because its nice
def binary_strings(n):
	ret = []

	def genbin(n, bs=''):
		if len(bs) == n:
		    ret.append(bs)
		else:
		    genbin(n, bs + '0')
		    genbin(n, bs + '1')

	genbin(n)

	return ret

def main(n, training_dist, layers, initial_weights, maxiter, eps, lr, seed):

	backend = Aer.get_backend('qasm_simulator')
	bin_strings = binary_strings(n)
	iters = []

	for elem in bin_strings:
		if not (elem in training_dist.keys()):
			training_dist[elem] = 0

	print(bin_strings)
	#print(len(training_dist.keys()))
	
	def run(angles):
		circuit = create_sim_circuit(n, angles, layers)
		job = execute(circuit, backend, shots = 100, seed_transpiler = seed, seed_simulator = seed)
		counts = job.result().get_counts()
		model_dist = get_dict_from_counts(counts)
		for elem in bin_strings:
			if not (elem in model_dist.keys()):
				model_dist[elem] = eps
		divergence = get_KL(model_dist, training_dist)
		iters.append(1)
		if len(iters) % len(angles) == 0:
			print(divergence)
		return divergence

	np.random.seed(seed)
	optimizer = ADAM(maxiter = maxiter, eps=eps, lr = lr) 
	result = optimizer.optimize(num_vars = (3*layers+3)*n,
                    objective_function = run,
                    initial_point = initial_weights)

	return result


def uniform_dist(n):
	keys = binary_strings(n)
	val = 1/len(keys)
	dct = {}

	for key in keys:
		dct[key] = val

	return dct

n=8
training_dist = {'10101010':2/23, '10101000': 1/23, '10001010':1/23, '11111111':4/23, '11111101':1/23, '11110111':1/23, '11111110':1/23, '11011111':1/23, '10111010': 2/23, '11101010':1/23, '10101011':1/23, '11011101':2/23, '01110111': 1/23, '10110101':2/23, '11010110': 1/23, '01011011': 1/23}
# training_dist = {'1010101010101010':0.3, '1111111111111111':0.2, '1011101010111010': 0.2, '1101110111011101':0.3}
layers = 3
initial_weights = [0.05]*(3*layers+3)*n
maxiter = 1000
eps = 0.75
lr = 0.15
seed = 11

result = main(n, training_dist, layers, initial_weights, maxiter, eps, lr, 11)
print(result)