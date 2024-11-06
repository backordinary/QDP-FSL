# https://github.com/jaschn/QuantumError/blob/93bb734f67d381637bfb1679b9dc893c484951e1/Noise_model/structured_search/helpers.py
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

from qiskit.providers import Backend  # type: ignore[attr-defined]
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

#assumes, the cnot should result in 0s
def error_rate_cnot(result: dict[str, int], control: int, target: int):
	sum_results = sum(result.values())
	result = {k: v/sum_results for k, v in result.items()}
	error_rate = 0
	for k, r in result.items():
		if k[control] == "0" and k[target] == "0":
			error_rate += r
	return error_rate

def get_error_rate_per_qubit_without_measure_error(result: dict[str, int], perfect_result: dict[str, int], error_measure: list[float]) -> list[float]:
	sum_results = sum(result.values())
	sum_perfect = sum(perfect_result.values())
	result = {k: v/sum_results for k, v in result.items()}
	perfect_result = {k: v/sum_perfect for k, v in perfect_result.items()}
	diffs = []
	qubits = len(next(iter(result.keys())))
	correct_values: list[tuple] = []
	for i in range(qubits):
		correct_0 = 0
		correct_1 = 0
		for k, v in perfect_result.items():
			if k[-(i+1)] == "0":
				correct_0 += v
			else: 
				correct_1 += v
		correct_values.append((correct_0, correct_1))

	for i in range(qubits):
		found_0 = 0
		found_1 = 0
		for k, v in result.items():
			if k[-(i+1)] == "0":
				found_0 += v
			else: 
				found_1 += v
		diff = abs(correct_values[i][0] - found_0)
		diffs.append((1 - diff)/error_measure[i])
	return diffs


def get_error_rate_old(result, perfect_result) -> float:
	correct_g = (result[r] for r in perfect_result)
	correct = sum(correct_g)
	error_rate = correct / sum(result.values())
	return error_rate


def get_error_rate_per_layer(transpiled_circuits: list[QuantumCircuit], backend: Backend, backend_perfect: Backend):
	res = []
	for circ in transpiled_circuits:
		n_qubits = circ.num_qubits
		_, _, input_circuit_layers = _get_layered_instructions(circ)
		test_circuits = []
		for i in range(len(input_circuit_layers)):
			qc = QuantumCircuit(n_qubits, n_qubits)
			last_barrier = False
			for layer in input_circuit_layers[:i]:
				for node in layer:
					if node.op.name == "barrier":
						last_barrier = True
						break
					qc.append(node.op, node.qargs, node.cargs)
					last_barrier = False
			if not last_barrier:
				qc.measure_all(add_bits=False)
				test_circuits.append(qc)
		results = backend.run(test_circuits).result().get_counts()
		results_perfect = backend_perfect.run(test_circuits).result().get_counts()


		test_circuits_measure = []
		for i in range(n_qubits):
			qc = QuantumCircuit(n_qubits, n_qubits)
			qc.measure(i, i)
			qc.name = f"measure test circuit qubit {i}"
			test_circuits_measure.append(qc)
		results_measure = backend.run(test_circuits_measure).result().get_counts()
		error_measure = []
		for i, r in enumerate(results_measure):
			correct = 0
			for k, v in r.items():
				if k[-(i+1)] == "0":
					correct += v
			error = correct / sum(r.values())
			error_measure.append(min(error, 1))

		res.append([get_error_rate_per_qubit_without_measure_error(r, rp, error_measure) for r, rp in zip(results, results_perfect)])

	return res

def get_gate_by_str(gate_name: str) -> Gate:
	gate_mapping = {
		"cx": CXGate(),
		"id": IGate(),
		"rz": RZGate(random.random() * math.pi),
		"sx": SXGate(),
		"x": XGate()
	}
	if gate_name in gate_mapping:
		return gate_mapping[gate_name]
	else:
		raise Exception(f"No gate configured for {gate_name}")

def construct_test_circuits(gate_name: str, n_qubits: int, coupling_map: list[list[int]]) -> Gate:
	gate = get_gate_by_str(gate_name)
	qcs: list[Gate] = []
	ops: list[tuple] = []
	if gate_name == "cx":
		prev_neighbors = defaultdict(list)
		assert coupling_map, "No coupling map provided for cx gate"
		for target_qubit in range(n_qubits):
			neighbors = [coupling[1] for coupling in coupling_map if coupling[0] == target_qubit]
			for q in neighbors:
				if target_qubit in prev_neighbors and q in prev_neighbors[target_qubit]: continue
				qc = QuantumCircuit(n_qubits, n_qubits)
				qc.append(gate, [target_qubit, q])
				qc.barrier()
				qc.append(gate.inverse(), [target_qubit, q])
				qc.measure_all(add_bits=False)
				qcs.append(qc)
				ops.append((gate_name, (target_qubit, q)))
				prev_neighbors[q].append(target_qubit)
	else:
		for target_qubit in range(n_qubits):
			qc = QuantumCircuit(n_qubits, n_qubits)
			qc.append(gate, [target_qubit])
			qc.barrier()
			qc.append(gate.inverse(), [target_qubit])
			qc.measure_all(add_bits=False)
			qcs.append(qc)
			ops.append((gate_name, (target_qubit, )))


	return qcs, ops

#https://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
def default_to_regular(d):
	if isinstance(d, defaultdict):
		d = {k: default_to_regular(v) for k, v in d.items()}
	return d