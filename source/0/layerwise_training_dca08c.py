# https://github.com/erenaykrcn/qml-demos/blob/21d3bf14c4ca9e8522940ab20600d3cb3d51de69/step2_training_methods/layerwise-training.py
"""
	To avoid the Barren-Plateau, we work with the local operator and 
	keep the depth (a.k.a number of layers) constant each time (namely: to 1)
	and optimize the params one layer at a time.
"""

import numpy as np
from qiskit.algorithms.optimizers import GradientDescent, SPSA, QNSPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer
from qiskit.opflow import Z, I, StateFn, PauliExpectation, Gradient, NaturalGradient
from qiskit.utils import QuantumInstance 
from qiskit.opflow import PauliExpectation, CircuitSampler


quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   # we set a seed for reproducibility
                                   shots = 1800, seed_simulator = 2718,
                                   seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance, caching="all")


num_qubits = 4
hamiltonian = Z ^ Z ^ (I ^ (num_qubits - 2)) # Local Operator

def minimize(ansatz, optimizer):
	"""
		Args:
			ansatz (QuantumCircuit): Ansatz circuit to train
			optimizer (Optimizer): Algorithm to be used for the optimization
		Returns:
			OptimizerResult: Result of the optimization
	"""
	initial_point = np.random.random(ansatz.num_parameters)

	expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
	gradient = Gradient().convert(expectation)

	expectation = PauliExpectation().convert(expectation)
	gradient = PauliExpectation().convert(gradient)

	def cost_function(theta):
		values_dict = dict(zip(ansatz.parameters, theta))
		return np.real(sampler.convert(expectation, values_dict).eval())

	def grad(theta):
		values_dict = dict(zip(ansatz.parameters, theta))
		return np.real(sampler.convert(gradient, values_dict).eval())

	return optimizer.minimize(cost_function,initial_point,grad)


def layerwise_training(ansatz, max_num_layers, optimizer):
	"""
		Args:
			ansatz (QuantumCircuit): Single circuit layer to train & repeat
        	max_num_layers (int): Maximum number of layers
        	optimizer (Optimizer): Algorithm to use to minimize exp. value
        Returns:
        	minimized_cost (float): Lowest value acheived
        	optimal_params (list[float]): Best parameters found
	"""
	optimal_parameters = []
	for reps in range(max_num_layers):
		ansatz.reps = reps

		# fix the already optimized parameters on the current layer
		# hence: "layerwise training"!
		values_dict = dict(zip(ansatz.parameters, optimal_parameters))
		partially_bound = ansatz.bind_parameters(values_dict)

		# This step works to optimize the same amount of params at each step
		# as we fix the already found params of the previous layers in the
		# previous step!
		result = minimize(partially_bound, optimizer)
		optimal_parameters += list(result.x)
		print('Layer:', reps, ' Best Value:', result.fun)

	return result.fun, optimal_parameters


ansatz = RealAmplitudes(num_qubits, entanglement='linear')
optimizer = GradientDescent(maxiter=50)

np.random.seed(12)  # for reproducibility
f_opt, optimal_parameters = layerwise_training(ansatz, 4, optimizer) # Layerwise linear depth
