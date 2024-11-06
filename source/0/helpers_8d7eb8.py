# https://github.com/jaschn/QuantumError/blob/93bb734f67d381637bfb1679b9dc893c484951e1/Noise_model/Custom/helpers.py
from collections import defaultdict
import math
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import logging

logger = logging.getLogger("main")


def get_error_rate(result: dict[str, int], perfect_result: dict[str, int]) -> float:
	sum_results = sum(result.values())
	sum_perfect = sum(perfect_result.values())
	result = {k: v/sum_results for k, v in result.items()}
	perfect_result = {k: v/sum_perfect for k, v in perfect_result.items()}
	diffs = []
	values = set(list(result.keys()) + list(perfect_result.keys()))
	for k in values:
		if k in result:
			if k in perfect_result:
				diff = abs(result[k] - perfect_result[k])
			else:
				diff = result[k]
		else:
			diff = perfect_result[k]
		diffs.append(diff)
	error_rate = 1 - sum(diffs)/2
	return error_rate

#https://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
def default_to_regular(d):
	if isinstance(d, defaultdict):
		d = {k: default_to_regular(v) for k, v in d.items()}
	return d

def measure_y_basis(qc: QuantumCircuit):
	n_qubits = qc.num_qubits
	qc.sdg(range(n_qubits))
	qc.h(range(n_qubits))
	qc.measure_all(add_bits=False)

def get_error_rate_per_qubit(result: dict[str, int], perfect_result: dict[str, int]) -> list[float]:
	sum_results = sum(result.values())
	sum_perfect = sum(perfect_result.values())
	result = {k: v/sum_results for k, v in result.items()}
	perfect_result = {k: v/sum_perfect for k, v in perfect_result.items()}
	diffs = []
	qubits = len(next(iter(result.keys())))
	correct_values: list[float] = []
	for i in range(qubits):
		correct_0 = 0
		for k, v in perfect_result.items():
			if k[-(i+1)] == "0":
				correct_0 += v
		correct_values.append(correct_0)

	for i in range(qubits):
		found_0 = 0
		for k, v in result.items():
			if k[-(i+1)] == "0":
				found_0 += v
		diff = abs(correct_values[i] - found_0)
		diffs.append(1 - diff)
	return diffs

def print_noise_model_params(noise: NoiseModel, readout_error=True, x_error=True, id_error=True):
	if readout_error:
		readout = noise._local_readout_errors
		for k, v in readout.items():
			print(f"Readout error on qubit {k[0]}:\n {v}")

	if x_error:
		x_err = noise._local_quantum_errors["x"]
		for k, v in x_err.items():
			print(f"X error on qubit {k[0]}: {v.probabilities}")

	if id_error:
		x_err = noise._local_quantum_errors["cx"]
		for k, v in x_err.items():
			print(f"CX error on qubit {k}: {v.probabilities}")


def number_of_combinations(true, count):
	return math.factorial(count) / (math.factorial(true) * math.factorial(count - true))

def f_universal(real, x_cnt):
	measured = 0
	if x_cnt % 2 == 0:
		for i in range(0, x_cnt+1, 2):
			measured += number_of_combinations(i, x_cnt) * real**i * (1-real)**(x_cnt-i)
	else:
		measured = real**x_cnt
		for i in range(1, x_cnt-1, 2):
			measured += number_of_combinations(i, x_cnt) * real**i * (1-real)**(x_cnt-i)

	return measured

def fit_reverse_gates(x_cnt):
	x = np.arange(0.9, 1.00000001, 0.000001)
	return Polynomial.fit(f_universal(x, x_cnt), x, x_cnt**2)