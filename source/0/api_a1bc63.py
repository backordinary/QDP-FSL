# https://github.com/korgan00/dr-qubit/blob/d1afa2475187f676d577f4d54db696f88cdada04/FlaskServer/api.py
#!/usr/bin/env python3

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute, IBMQ


def run_qasm(qasm, backend_to_run="qasm_simulator", api_token=None, shots=1024, memory=False):
	active_account = IBMQ.active_account()
	if api_token:
		if active_account is None:
			IBMQ.enable_account(api_token)
		elif active_account['token'] != api_token:
			IBMQ.enable_account(api_token)

	qc = QuantumCircuit.from_qasm_str(qasm)
	backend = Aer.get_backend(backend_to_run)
	job_sim = execute(qc, backend, shots=shots, memory=memory)
	sim_result = job_sim.result()
	if memory:
		return sim_result.get_memory(qc)
	else:
		return sim_result.get_counts(qc)

def backend_configuration(backend_to_run="qasm_simulator"):
	backend = Aer.get_backend(backend_to_run)
	return backend.configuration().as_dict();