# https://github.com/ronlv4/QuantumCircuit/blob/6bd50fb2cbb12e2ae5ff883a75844efb1f18ae42/A1Context.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from CircuitContext import CircuitContext


class A1Context(CircuitContext):
    def __init__(self, n, t, shots, samples, theta_range):
        super().__init__(t, shots, samples, theta_range)
        self.n = n
        self.theta = Parameter('θ')
        self.build_parametrized_circuits = self.set_parametrized_circuits()
        self.analytic_label = self.set_analytic_label()
        self.analytic_expression = self.set_analytic_expression()

    def get_circuit_specifier(self):
        return 'A1'

    def get_analytic_label(self):
        return self.analytic_label

    def get_analytic_expression(self):
        return self.analytic_expression

    def get_parametrized_circuits(self):
        return self.build_parametrized_circuits

    def set_analytic_expression(self):
        return lambda theta: np.cos(theta / 2) ** 2

    def set_analytic_label(self):
        return r'cos^2 (\frac{\theta}{2})'

    def set_parametrized_circuits(self):
        circuit: QuantumCircuit = QuantumCircuit(1, 1)
        circuit.rx(self.theta, 0)
        circuit.measure(0, 0)
        return [circuit.bind_parameters({self.theta: theta_val}) for theta_val in self.theta_range]
