# https://github.com/erenaykrcn/qml-demos/blob/21d3bf14c4ca9e8522940ab20600d3cb3d51de69/step3_variational_classification/variational_classification.py
"""
	Implements a Quantum Variational Classifier. Data encoder uses a ZZFeatureMap
	and Variational Circuit uses a TwoLocal Circuit with Y/Z Rotations and a controlled
	phase gate (CZ).

	Training Data maps the (x1,x2) point to a label y=0 or 1 (arbitrarily). To generate the
	prediction 0 or 1, our algorithm uses the parity of the measured qubits. This
	parity function is chosen arbitrarily. From M amount of measurements, we get in total
	M amount of bits, all belonging to the the same label. Therefore the bit ratio of the 
	measured bitstring gives us the success/failure ratio directly (and with that the success
	probability p). 

	Cost function generates a cumulative cost of all the training data, whereas each data point
	costs -log(p + 1e-10), where p is the successful classification probability (acquired from the process above). 
	
	Encoder has to have 2 params (due to the data set having two features/dimensions) and since
	ZZFeatureMap works with num_params = num_qubits, we use 2 qubits. As a result of using TwoLocal
	with two qubits and 2 reps (with rots = 2 Pauli rotations(ry and rz)) => (reps + 1) * 
	(rots * numb_qubits) = 12 theta params to be optimized.
"""
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from data_points import TRAIN_DATA, TRAIN_LABELS
from qiskit import Aer, execute
from qiskit.algorithms.optimizers import SPSA

backend = Aer.get_backend('qasm_simulator')

encoder = ZZFeatureMap(feature_dimension=2, reps=2)
ansatz = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

AD_HOC_CIRCUIT = encoder.compose(ansatz)
AD_HOC_CIRCUIT.measure_all()


def assign_parameters(x_data, theta):
	"""Assigns parameter values to `AD_HOC_CIRCUIT`.
	Args:
		x_data (list): Data values to be encoded to the feature map
		theta (list): Parameter values for `ansatz`, to be optimized by training
	Returns:
		QuantumCircuit: `AD_HOC_CIRCUIT` with parameters assigned
	"""
	parameters = {}
	for i, p in enumerate(encoder.ordered_parameters):
		parameters[p] = x_data[i]
	for i, p in enumerate(ansatz.ordered_parameters):
		parameters[p] = theta[i]

	return AD_HOC_CIRCUIT.assign_parameters(parameters)


def parity(bitstring):
	"""Returns 1 if parity of `bitstring` is even, otherwise 0."""
	hamming_weight = sum(int(k) for k in list(bitstring))
	return (hamming_weight+1) % 2


def sum_to_probability(result):
	"""Converts a dict of bitstrings and their counts,
	to parities and their counts"""
	shots = sum(result.values())
	probabilities = {0: 0, 1: 0}
	for bitstring, counts in result.items():
		label = parity(bitstring)
		probabilities[label] += counts / shots
	return probabilities


def get_classification_probabilities(x_data, theta):
	circuits = [assign_parameters(x, theta) for x in x_data]
	results = execute(circuits, backend).result()
	return [
		sum_to_probability(results.get_counts(c)) for c in circuits]


def cost_function(theta):
	cost = 0
	classifications = get_classification_probabilities(TRAIN_DATA,theta) 
	for i, classification in enumerate(classifications):
		p = classification.get(TRAIN_LABELS[i])
		cost += -np.log(p + 1e-10)
	cost /= len(TRAIN_DATA)
	return cost


# Classical optimization, uses Simultaneous Perturbation 
# Stochastic Approximation (SPSA) method to compute the gradient.


class OptimizerLog():
	def __init__(self):
		self.evaluations = []
		self.theta_values = []
		self.costs = []
	def update(self, evaluation, theta, cost, _stepsize, _accept):
		self.evaluations.append(evaluation)
		self.theta_values.append(theta)
		self.costs.append(cost)

		print("Evaluations: " + str(evaluation) + "|| Cost: " + str(cost))


log = OptimizerLog()
optimizer = SPSA(maxiter=35, callback=log.update)

#initial_point = np.random.random(ansatz.num_parameters)
initial_point = np.array([3.28559355, 5.48514978, 5.13099949,
						0.88372228, 4.08885928, 2.45568528,
						4.92364593, 5.59032015, 3.66837805,
						4.84632313, 3.60713748, 2.43546])

result = optimizer.minimize(cost_function, initial_point)

opt_theta = result.x # Result of the optimization.
min_cost = result.fun


# Code for the visualization of the optimization.

import matplotlib.pyplot as plt
plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.savefig("optimizer.png")
