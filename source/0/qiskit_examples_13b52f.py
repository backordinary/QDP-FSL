# https://github.com/PhilWun/MAProjekt/blob/3b115e2a4c1c9f0419bbc515f2318bd50794d268/src/pl/qiskit_examples.py
from math import pi
from typing import List

import pennylane as qml
import torch
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from src.qk import QNN1


def generate_circuit_func(params: List[Parameter], qc: QuantumCircuit):
	qr = QuantumRegister(3, "q0")
	input_circ = QuantumCircuit(qr)
	input_params = [Parameter("input_1"), Parameter("input_2"), Parameter("input_3")]
	input_circ.rx(input_params[0], qr[0])
	input_circ.rx(input_params[1], qr[1])
	input_circ.rx(input_params[2], qr[2])

	combined_circuit = input_circ + qc

	def func(input_values: torch.Tensor, weights: torch.Tensor):
		value_dict = {}

		for param, value in zip(input_params, input_values):
			value_dict[param] = value

		for param, value in zip(params, weights):
			value_dict[param] = value

		qml.from_qiskit(combined_circuit)(value_dict)

		return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

	return func


def cost(input_values: torch.Tensor, weights: torch.Tensor, target: torch.Tensor, qnode: qml.QNode):
	output = (qnode(input_values, weights) - 1) / -2.0  # normalizes the expected values
	print(output)

	return torch.mean((output - target) ** 2)


def train_loop(
		iterations: int, input_values: torch.Tensor, weights: torch.Tensor, target: torch.Tensor, qnode: qml.QNode):
	opt = torch.optim.Adam([weights], lr=0.1)

	for i in range(iterations):
		print("iteration", i)

		for j in range(input_values.size()[0]):
			opt.zero_grad()
			loss = cost(input_values[j], weights, target[j], qnode)
			loss.backward()
			print(loss.item())

			opt.step()

		print()


def test_training():
	dev = qml.device("default.qubit", wires=3, shots=1000, analytic=False)
	qc = QNN1.create_qiskit_circuit("", 3)
	params: List[Parameter] = list(qc.parameters)

	input_values = torch.tensor([[0, 0, 0], [pi, pi, pi]], requires_grad=False)
	weights = torch.tensor(np.random.rand(len(params)) * 2 * pi, requires_grad=True)
	target = torch.tensor([[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]], requires_grad=False)

	qnode = qml.QNode(generate_circuit_func(params, qc), dev, interface="torch")

	train_loop(100, input_values, weights, target, qnode)


if __name__ == "__main__":
	test_training()
