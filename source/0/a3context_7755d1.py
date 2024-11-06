# https://github.com/ronlv4/QuantumCircuit/blob/6bd50fb2cbb12e2ae5ff883a75844efb1f18ae42/A3Context.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from CircuitContext import CircuitContext


class A3Context(CircuitContext):
    def __init__(self, n, t, shots, samples, theta_range):
        super().__init__(t, shots, samples, theta_range)
        self.n = n
        self.theta = Parameter('θ')
        self.build_parametrized_circuits = self.set_parametrized_circuits()
        self.analytic_label = self.set_analytic_label()
        self.analytic_expression = self.set_analytic_expression()

    def get_circuit_specifier(self):
        return 'A3'

    def get_analytic_label(self):
        return self.analytic_label

    def get_analytic_expression(self):
        return self.analytic_expression

    def get_parametrized_circuits(self):
        return self.build_parametrized_circuits

    def set_analytic_expression(self):
        return lambda theta: 1 - (np.sin(theta / np.sqrt(2)) ** 2) / 2

    def set_analytic_label(self):
        return r'$1 - \frac{1}{2} \sin^2 (\frac{\theta}{\sqrt{2}})$'

    def set_parametrized_circuits(self):
        circuit: QuantumCircuit = QuantumCircuit(1, 1)
        for _ in range(self.n):
            circuit.rx(self.theta, 0)
            circuit.rz(self.theta, 0)
            circuit.barrier()
        circuit.measure(0, 0)
        return [circuit.bind_parameters({self.theta: theta_val}) for theta_val in self.theta_range / self.n]
