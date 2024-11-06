# https://github.com/jaschn/QuantumError/blob/93bb734f67d381637bfb1679b9dc893c484951e1/Noise_model/Custom/check_calibration_to_noise_model.py
import os
import sys
import traceback
from turtle import back
from unittest import result
from venv import create
import matplotlib.pyplot as plt
import numpy as np
from helpers import get_error_rate, print_noise_model_params
from matplotlib.pyplot import figure
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import copy
from modify_probs_functions import modify_relaxation, number_of_noise_params, modify_noise_model_local, modify_noise_model_readout
import pickle
from qiskit.providers.ibmq import IBMQ, least_busy
from custom_noise_model_creator import CustomNoiseModelCreator
import math
import random
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate, ZGate, Reset, Measure
from NoiseModelImprove import NoiseModelImprove
import logging

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(filename='check_calibration_to_noise_model.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

def add_random_gate(qc: QuantumCircuit):
	param = random.uniform(0,2) * math.pi
	gates = [XGate(), CXGate()]
	gate = random.choice(gates)
	n_qubits = qc.num_qubits
	if isinstance(gate, CXGate):
		target = random.sample(range(n_qubits), k=2)
		qc.append(gate, target)
	else:
		target = random.choice(list(range(n_qubits)))
		qc.append(gate, [target])

	return qc

def get_circuit_with_length(hw, length=2):
	samples = np.linspace(np.ceil(length/2), length*3, length*5, dtype=int)
	n_qubits = hw.configuration().n_qubits
	for _ in range(100):
		circuits = []
		for sample in samples:
			qc = QuantumCircuit(n_qubits, n_qubits)
			qc_tmp = QuantumCircuit(n_qubits, n_qubits)
			for j in range(sample):
				qc_tmp = add_random_gate(qc_tmp)
			qc.barrier()
			qc.append(qc_tmp, range(n_qubits), range(n_qubits))
			qc.barrier(range(n_qubits))
			qc.append(qc_tmp.inverse(), range(n_qubits), range(n_qubits))
			qc.barrier()
			qc.measure_all(add_bits=False)
			circuits.append(qc)
		c_t = transpile(circuits, hw, optimization_level=0)
		depths = np.array([c.depth() - 1 for c in c_t])
		if length in depths:
			idx = np.nonzero(depths == length)[0]
			choosen_idx = random.choice(idx)
			assert c_t[choosen_idx].depth() - 1 == length
			return c_t[choosen_idx]
		else:
			pass
		
	raise Exception("no circuit found")

def create_test_circuits(hw, start, step, end, samples_per_step):
	samples = np.arange(start, end, step, dtype=int)
	circs_samples = []
	for sample in samples:
		for _ in range(samples_per_step):
			qc = get_circuit_with_length(hw, sample)
			circs_samples.append(qc)
	return circs_samples

provider = IBMQ.load_account()
backends = provider.backends(n_qubits=5, operational=True, simulator=False)
#backends = [AerSimulator.from_backend(b) for b in backends]

if False:
	test_jobs = {}
	test_creators = {}
	test_from_hardware_noise = {}
	for hw in backends:
		logger.info(hw.name())
		test_from_hardware_noise[hw.name()] = NoiseModel.from_backend(hw)
		if os.path.isfile(f"test_circs_cnot_{hw.name()}.pickle"):
			logger.debug("loading existing test circs")
			with open(f"test_circs_cnot_{hw.name()}.pickle", "rb") as file:
				test_circs = pickle.load(file)
		else:
			logger.debug(f"create new test circs for {hw.name()}")
			test_circs = create_test_circuits(hw, 6, 2, 25, 10)
			with open(f"test_circs_cnot_{hw.name()}.pickle", "wb") as file:
				pickle.dump(test_circs, file)

		creator = CustomNoiseModelCreator(hw)		
		creator.create_test_circuits()

		creator.run_circs()
		job = hw.run(test_circs)

		test_jobs[hw.name()] = job
		test_creators[hw.name()] = creator


	with open(f"tests_cnot.pickle", "wb") as file:
		pickle.dump((test_jobs, test_creators, test_from_hardware_noise), file)


	exit()

else:
	with open(f"tests_cnot.pickle", "rb") as file:
		test_jobs, test_creators, test_from_hardware_noise = pickle.load(file)

	all_errors = {}
	all_noise = {}

	for name, job in test_jobs.items():
		try:
			if not job.done():
				logger.info(f"skipped {name}")
				continue
			logger.info(f"\n\n\nTest with backend {name}")
			with open(f"test_circs_cnot_{name}.pickle", "rb") as file:
				test_circs = pickle.load(file)

			if os.path.isfile(f"test_results_cnot_{name}.pickle"):
				with open(f"test_results_cnot_{name}.pickle", "rb") as file:
					results = pickle.load(file)
			else:
				results = job.result().get_counts()
				with open(f"test_results_cnot_{name}.pickle", "wb") as file:
					pickle.dump(results, file)

			creator = CustomNoiseModelCreator(provider.get_backend(name))	
			creator.create_test_circuits()
			
			creator.job = test_creators[name].job
			#creator = test_creators[name]

			proposed_noise = {}
			proposed_noise["creator"] = creator.get_custom_noise_model()		

			proposed_noise["from_backend"] = test_from_hardware_noise[name]

			hw_error = np.array([get_error_rate(r, {"00000" : 1}) for r in results])

			proposed_sims = {k: AerSimulator(noise_model=v) for k, v in proposed_noise.items()}
			proposed_errors = {k: np.array([get_error_rate(r, {"00000" : 1}) for r in v.run(test_circs, shots=10000).result().get_counts()]) for k, v in proposed_sims.items()}

			for k, v in proposed_errors.items():
				diff = np.abs(hw_error-v).mean()
				logger.info(f"error with noise version \"{k}\": {diff}")

			all_errors[name] = proposed_errors
			all_noise[name] = proposed_noise

		except KeyboardInterrupt as e:
			logger.exception(f"\nException: {e}")

	logger.info("save to file")

	with open(f"noise_and_errors.pickle", "wb") as file:
		pickle.dump((all_noise, all_errors), file)

	logger.info("done")