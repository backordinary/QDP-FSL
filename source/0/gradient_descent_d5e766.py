# https://github.com/erenaykrcn/qml-demos/blob/21d3bf14c4ca9e8522940ab20600d3cb3d51de69/step2_training_methods/gradient_descent.py
"""
	Project that implements qiskits gradient descent learning algorithms for
	'Real Amplitudes' Ansatz and Hamiltonian H = Z^Z cost function. 

	Four different approaches are implemented: Gradient Descent (Parameter Shift rule),
	Natural Gradient, Simultaneous Perturbation Stochastic Approximation (SPSA) and
	Quantum Natural - SPSA (QNSPSA).  
"""

import numpy as np
from qiskit.algorithms.optimizers import GradientDescent, SPSA, QNSPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer
from qiskit.opflow import Z, StateFn, PauliExpectation, Gradient, NaturalGradient
from qiskit.utils import QuantumInstance 
from qiskit.opflow import PauliExpectation, CircuitSampler


quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   # we set a seed for reproducibility
                                   shots = 800, seed_simulator = 2718,
                                   seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance, caching="all")


""" 
	Ansatz that we use for this model: RealAmplitudes. 
	Implements the cost function as <phi|H|phi>, whereas the
	hamiltonian is the Z-Pauli applied on both qubits.
"""
ansatz = RealAmplitudes(num_qubits=2, reps=1, entanglement="linear").decompose()
hamiltonian = Z ^ Z
expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)

def cost_function(theta):
	"""
		No external training data is used to implement 
		the hamiltonian/cost function. 
	"""
	value_dict = dict(zip(ansatz.parameters, theta))
	pauli_basis = PauliExpectation().convert(expectation)

	# Measurement/Sampling
	result = sampler.convert(pauli_basis, params=value_dict).eval()
	return np.real(result)

def evaluate_gradient(theta):
	"""
		We use analytic gradient approach with parameter shift rule.
	"""
	value_dict = dict(zip(ansatz.parameters, theta))
	gradient = Gradient().convert(expectation)
	gradient_in_pauli_basis = PauliExpectation().convert(gradient)

	# Measurement/Sampling
	result = sampler.convert(gradient_in_pauli_basis,params=value_dict).eval()
	return np.real(result)


def evaluate_natural_gradient(theta):
	"""
		As an alternative to gradient, we can use natural gradient.
		This requires more measurements/evaluations so it is costlier 
		but is more efficient to reach the minimum by fewer iterations.
	"""
	value_dict = dict(zip(ansatz.parameters, theta))
	gradient = NaturalGradient(regularization='ridge').convert(expectation)
	gradient_in_pauli_basis = PauliExpectation().convert(gradient)

	# Measurement/Sampling
	result = sampler.convert(gradient_in_pauli_basis,params=value_dict).eval()
	return np.real(result)


class OptimizerLog:
	def __init__(self):
		self.loss = []
	def update(self, _nfevs, _theta, ftheta_, *_):
		self.loss.append(ftheta_)  # Saves the f(theta) values of every step


initial_point = np.array([0.43253681, 0.09507794, 0.42805949, 0.34210341])

gd_log_vanilla = OptimizerLog()
gd_vanilla = GradientDescent(maxiter=100,learning_rate=0.01,callback=gd_log_vanilla.update)
result = gd_vanilla.minimize(
		fun=cost_function,
		x0=initial_point,
		jac=evaluate_gradient
	)

gd_log_natural = OptimizerLog()
gd_natural = GradientDescent(maxiter=100,learning_rate=0.01,callback=gd_log_natural.update)
result = gd_natural.minimize(
		fun=cost_function,
		x0=initial_point,
		jac=evaluate_natural_gradient
	)

spsa_log = OptimizerLog()
spsa = SPSA(maxiter=100, learning_rate=0.01,
            perturbation=0.01, callback=spsa_log.update)
result = spsa.minimize(cost_function, initial_point)

qnspsa_log = OptimizerLog()
fidelity = QNSPSA.get_fidelity(ansatz,
                               quantum_instance,
                               expectation=PauliExpectation())
qnspsa = QNSPSA(fidelity, maxiter=100, learning_rate=0.01,
                                       perturbation=0.01,
                                       callback=qnspsa_log.update)
result = qnspsa.minimize(cost_function, initial_point)

"""
	Visualizing our results.
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(28, 12))
plt.plot(gd_log_vanilla.loss, label='vanilla gradient descent')
plt.plot(gd_log_natural.loss, label='natural gradient descent')
plt.plot(spsa_log.loss, 'C0', ls='--', label='SPSA')
plt.plot(qnspsa_log.loss, 'C1', ls='--', label='QN-SPSA')
plt.axhline(-1, ls='--', c='C3', label='target')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.legend()
plt.savefig("optimizers.png")