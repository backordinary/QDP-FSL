# https://github.com/hrahman12/EduQuantX/blob/32f8418913b886a6ef60d0a406f3bb824328cddd/IBMQ%20API/FlaskServer/api.py
#!/usr/bin/env python3

from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute, IBMQ


def run_qasm(qasm, backend_to_run="qasm_simulator", api_token=None, shots=1024, memory=False):
    if api_token:
        IBMQ.enable_account(api_token, '4aca908be9baaaa198b1e27c95b5d14364784d2d8a1fdb1b316fa2b560cf69285ab00cd5284263cd2b5254de134d57904c8eb2793941983c0bd55b28fae91974')
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