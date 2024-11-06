# https://github.com/magn5452/QiskitQaoa/blob/8517ac7992eaeb4be090cb12dd53f349e5af5ff3/VehicleRouting/runable/testing_environment_4.py
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.extensions import HamiltonianGate
from qiskit.providers import aer
from qiskit.providers.aer import StatevectorSimulator
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram

from VehicleRouting.standard.concretization.CircuitPlotter import MPLCircuitPlotter

circuit = QuantumCircuit(4)
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)

circuit.measure_all()

circuit_plotter = MPLCircuitPlotter()
circuit_plotter.plot(circuit)

backend = StatevectorSimulator(precision='single')
backend.shots = 2 ** 8

counts = backend.run(circuit).result().get_counts()

plt.show()
