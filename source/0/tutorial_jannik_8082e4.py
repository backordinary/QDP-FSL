# https://github.com/jannikobenhoff/quantum_computation/blob/5326dbc1c3d064858291b30489862e8facaf2693/tutorial/tutorial_jannik.py
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, NLocal, EfficientSU2, RealAmplitudes
from qiskit.opflow import Z, I, StateFn, PauliExpectation, CircuitSampler, Gradient, NaturalGradient
from qiskit import Aer
from qiskit.utils import QuantumInstance

import sklearn
import math

from helpers import open_image

if __name__ == "__main__":

    """Parameterized quantum circuits"""

    theta = Parameter('θ')
    qc = QuantumCircuit(2)
    qc.rz(theta, 0)
    qc.crz(theta, 0, 1)
    print(qc, "\ngleiche Theta Winkel\n---")

    theta_list = ParameterVector('θ', length=2)
    qc = QuantumCircuit(2)
    qc.rz(theta_list[0], 0)
    qc.crz(theta_list[1], 0, 1)
    print(qc, "\nunterschiedliche Theta\n---")

    qc_zz = ZZFeatureMap(3, reps=1, insert_barriers=True)
    print(qc_zz.decompose(), "\nZZFeatureMap\n---")

    qc_twolocal = TwoLocal(num_qubits=3, reps=2, rotation_blocks=['ry', 'rz'],
                           entanglement_blocks='cz', skip_final_rotation_layer=True,
                           insert_barriers=True)
    print(qc_twolocal.decompose())

    # rotation block:
    rot = QuantumCircuit(2)
    params = ParameterVector('r', 2)
    rot.ry(params[0], 0)
    rot.rz(params[1], 1)

    # entanglement block:
    ent = QuantumCircuit(4)
    params = ParameterVector('e', 3)
    ent.crx(params[0], 0, 1)
    ent.crx(params[1], 1, 2)
    ent.crx(params[2], 2, 3)

    qc_nlocal = NLocal(num_qubits=6, rotation_blocks=rot,
                       entanglement_blocks=ent, entanglement='linear',
                       skip_final_rotation_layer=True, insert_barriers=True)

    print(qc_nlocal.decompose())

    """Data encoding"""
    """Basis encoding"""
    """Amplitude encoding"""
    # x1 = 1.5, 0
    # x2 = -2, 3

    desired_state = [
        1 / math.sqrt(15.25) * 1.5,
        0,
        1 / math.sqrt(15.25) * -2,
        1 / math.sqrt(15.25) * 3]
    qc = QuantumCircuit(2)
    qc.initialize(desired_state, [0, 1])

    print(qc.decompose().decompose().decompose().decompose().decompose())

    """Angle encoding"""
    # x = 0, pi/4, pi/2
    qc = QuantumCircuit(3)
    qc.ry(0, 0)
    qc.ry(math.pi/4, 1)
    qc.ry(math.pi/2, 2)
    print(qc)

    """Arbitrary encoding"""
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    circuit = EfficientSU2(num_qubits=3, reps=1, insert_barriers=True)
    encode = circuit.bind_parameters(x)
    print(encode.decompose())

    '''Training circuits'''

    '''Gradients'''
    ansatz = RealAmplitudes(num_qubits=2, reps=1,
                            entanglement='linear').decompose()
    print(ansatz)
    hamiltonian = Z ^ Z
    expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
    pauli_basis = PauliExpectation().convert(expectation)
    quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                       # we'll set a seed for reproducibility
                                       shots=8192, seed_simulator=2718,
                                       seed_transpiler=2718)
    sampler = CircuitSampler(quantum_instance)

    def evaluate_expectation(theta):
        value_dict = dict(zip(ansatz.parameters, theta))
        result = sampler.convert(pauli_basis, params=value_dict).eval()
        return np.real(result)

    point = np.random.random(ansatz.num_parameters)
    INDEX = 2
    EPS = 0.2  # Fehler
    # make identity vector with a 1 at index ``INDEX``, otherwise 0
    e_i = np.identity(point.size)[:, INDEX]

    plus = point + EPS * e_i
    minus = point - EPS * e_i

    finite_difference = (evaluate_expectation(plus) - evaluate_expectation(minus)) / (2 * EPS)
    print(finite_difference)

    # statt manuell, lieber mit Qiskit's Gradient class

    shifter = Gradient('fin_diff', analytic=False, epsilon=EPS)
    grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])
    print(grad)

    value_dict = dict(zip(ansatz.parameters, point))
    sampler.convert(grad, value_dict).eval().real

    '''Analytic gradients'''
    # analytisch berechnen mit f' = f(x+h) - f(x-h) / 2

    EPS = np.pi / 2
    e_i = np.identity(point.size)[:, INDEX]

    plus = point + EPS * e_i
    minus = point - EPS * e_i

    finite_difference = (evaluate_expectation(plus) - evaluate_expectation(minus)) / 2
    print(finite_difference)

    # mit Qiskit Gradient class
    shifter = Gradient(analytic=False, epsilon=EPS)  # parameter-shift rule is the default
    grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])
    sampler.convert(grad, value_dict).eval().real

    gradient = Gradient().convert(expectation)
    gradient_in_pauli_basis = PauliExpectation().convert(gradient)
    sampler = CircuitSampler(quantum_instance)


    def evaluate_gradient(theta):
        value_dict = dict(zip(ansatz.parameters, theta))
        result = sampler.convert(gradient_in_pauli_basis,
                                 params=value_dict).eval()
        return np.real(result)


    from qiskit.algorithms.optimizers import GradientDescent

    # initial_point = np.random.random(ansatz.num_parameters)
    initial_point = np.array([0.43253681, 0.09507794, 0.42805949, 0.34210341])
    class OptimizerLog:
        """Log to store optimizer's intermediate results"""

        def __init__(self):
            self.loss = []

        def update(self, _nfevs, _theta, ftheta, *_):
            """Save intermediate results. Optimizers pass many values
            but we only store the third ."""
            self.loss.append(ftheta)

    gd_log = OptimizerLog()
    gd = GradientDescent(maxiter=300,
                         learning_rate=0.01,
                         callback=gd_log.update)
    result = gd.minimize(
        fun=evaluate_expectation,  # function to minimize
        x0=initial_point,  # initial point
        jac=evaluate_gradient  # function to evaluate gradient
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.plot(gd_log.loss, label='vanilla gradient descent')
    plt.axhline(-1, ls='--', c='C3', label='target')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()
    # plt.show()

    '''Natural Gradients'''


    natural_gradient = (NaturalGradient(regularization='ridge').convert(expectation))
    natural_gradient_in_pauli_basis = PauliExpectation().convert(natural_gradient)
    sampler = CircuitSampler(quantum_instance, caching="all")

    def evaluate_natural_gradient(theta):
        value_dict = dict(zip(ansatz.parameters, theta))
        result = sampler.convert(natural_gradient, params=value_dict).eval()
        return np.real(result)

    print('Vanilla gradient:', evaluate_gradient(initial_point))
    print('Natural gradient:', evaluate_natural_gradient(initial_point))

    qng_log = OptimizerLog()
    qng = GradientDescent(maxiter=300,
                          learning_rate=0.01,
                          callback=qng_log.update)

    result = qng.minimize(evaluate_expectation,
                          initial_point,
                          evaluate_natural_gradient)

    # Plot loss
    plt.figure(figsize=(7, 3))
    plt.plot(gd_log.loss, 'C0', label='vanilla gradient descent')
    plt.plot(qng_log.loss, 'C1', label='quantum natural gradient')
    plt.axhline(-1, c='C3', ls='--', label='target')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()

    

