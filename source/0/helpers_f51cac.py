# https://github.com/jaschn/QuantumError/blob/93bb734f67d381637bfb1679b9dc893c484951e1/experiments/error_comparison/helpers.py
from collections import defaultdict
import math
import random
from qiskit.visualization.utils import _get_layered_instructions
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.providers import Backend
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate, Reset
from qiskit.circuit import Gate, Instruction
from typing import List, Optional, Union
from datetime import datetime, timedelta

from qiskit.providers import BaseBackend, Backend  # type: ignore[attr-defined]
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
import heapq


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