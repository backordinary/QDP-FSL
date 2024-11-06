# https://github.com/Arkonaire/QAOA-Combinatorics/blob/84ea91ef71076b1ff22cf6d0ec1484619c2b5ac6/qaoa_engine.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua.components.optimizers import COBYLA


class QAOA:

    """
    QAOA Engine class. Abstract Class for implementing the optimizations required in the QAOA algorithm.
    For inheriting into subclasses for specific combinatorial problems, simply override the following functions:

    - build_cost_ckt(self)
    - cost_function(self)
    - build_mixer_ckt(self) (optional)

    Refer docstrings for individual functionalities of these functions.
    """

    def __init__(self, n, p=4, backend=None):

        """
        Initialize the QAOA engine and set off the necessary sequence of actions for optimization.
        Args:
            n: No. of qubits in QAOA circuit.
            p: No. of cost-mixer layers in QAOA circuit.
            backend: Custom backend. Can be used for noisy simulations
        """

        # Assign size parameters
        self.n = n
        self.p = p

        # Assign control parameters
        self.num_shots = 8192
        self.error = -1

        # Set backend options
        self.backend = QasmSimulator() if backend is None else backend

        # Create variational parameters
        self.gamma = ParameterVector('gamma', length=self.p)
        self.beta = ParameterVector('beta', length=self.p)
        self.gamma_val = list(2*np.pi*np.random.rand(self.p))
        self.beta_val = list(2*np.pi*np.random.rand(self.p))

        # Create circuits
        self.cost_ckt = self.build_cost_ckt()
        self.mixer_ckt = self.build_mixer_ckt()
        self.variational_ckt = self.build_variational_ckt()

        # Dust off
        self.optimize()

    def cost_function(self, z):

        """
        Dummy cost function. Override in child class.
        Args:
            z: An integer or bitstring whose cost is to be determined.
        Return:
            Cost function as integer.
        """

        # Convert to bitstr
        if not isinstance(z, str):
            z = format(z, '0' + str(self.n) + 'b')
        z: str

        # Evaluate dummy C(z)
        cost = len(z)
        return cost

    def build_cost_ckt(self):

        """
        Dummy cost circuit. Override in child class.
        Return:
            QuantumCircuit. Parameterized cost circuit layer.
        """

        # Build dummy circuit
        circ = QuantumCircuit(self.n, name='$U(H_C,\\gamma)$')
        circ.rz(Parameter('param_c'), range(self.n))
        return circ

    def build_mixer_ckt(self):

        """
        Default QAOA mixer circuit. Override in child class if necessary.
        Return:
            QuantumCircuit. Parameterized mixer circuit layer.
        """

        # Build default mixer circuit.
        circ = QuantumCircuit(self.n, name='$U(H_B,\\beta)$')
        circ.rx(2*Parameter('param_b'), range(self.n))
        return circ

    def build_variational_ckt(self):

        """
        Combine p layers of cost-mixer layers to build the full variational circuit.
        Return:
            QuantumCircuit. Parameterized p-layer variational circuit.
        """

        # Acquire parameter handles
        param_c = list(self.cost_ckt.parameters)[0]
        param_b = list(self.mixer_ckt.parameters)[0]

        # Build variational circuit
        circ = QuantumCircuit(self.n)
        circ.h(range(self.n))
        for i in range(self.p):
            circ.append(self.cost_ckt.to_gate({param_c: self.gamma[i]}), range(self.n))
            circ.append(self.mixer_ckt.to_gate({param_b: self.beta[i]}), range(self.n))
        circ.measure_all()
        return circ

    def expectation(self, beta=None, gamma=None):

        """
        Runs a set of QASM simulations for a specified set of parameters to determine the expectation value.
        Args:
            beta: Mixer parameter set. List of size n. Defaults to current state of optimizer.
            gamma: Cost parameter set. List of size n. Defaults to current state of optimizer.
        Return:
            Measured expectation value as integer.
        """

        # Resolve default values
        if beta is None:
            beta = self.beta_val
        if gamma is None:
            gamma = self.gamma_val

        # Evaluate expectation value
        circ = self.variational_ckt.bind_parameters({self.beta: beta, self.gamma: gamma})
        result = execute(circ, self.backend, shots=self.num_shots).result()
        counts = result.get_counts()
        expval = sum([self.cost_function(z) * counts[z] / self.num_shots for z in counts.keys()])
        return expval

    def optimize(self):

        """
        Run COBYLA optimizer for QAOA circuit optimization.
        """

        # Define objective function
        def objfunc(params):
            return self.expectation(beta=params[0:self.p], gamma=params[self.p:2 * self.p])

        # Optimize parameters
        optimizer = COBYLA(maxiter=5000, tol=0.0001)
        params = self.beta_val + self.gamma_val
        ret = optimizer.optimize(num_vars=2 * self.p, objective_function=objfunc, initial_point=params)
        self.beta_val = ret[0][0:self.p]
        self.gamma_val = ret[0][self.p:2 * self.p]
        self.error = ret[1]
        return

    def sample(self, shots=None, vis=False):

        """
        Sample final optimized circuit for output.
        Args:
            shots: No. of samples to take.
            vis: Boolean value. Displays histogram if set to True.
        Return:
            Bitstring for output with max measurement counts and the average expectation value.
        """

        # Resolve defaults
        if shots is None:
            shots = self.num_shots

        # Sample maximum cost value
        circ = self.variational_ckt.bind_parameters({self.beta: self.beta_val, self.gamma: self.gamma_val})
        result = execute(circ, self.backend, shots=shots).result()
        counts = result.get_counts()

        # Display data if asked
        if vis:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.25)
            plot_histogram(counts, title='Sample Output', bar_labels=False, ax=ax)

        # Return optimized selection
        avg_cost = sum([self.cost_function(z)*count for z, count in counts.items()]) / self.num_shots
        return max(counts, key=counts.get), avg_cost
