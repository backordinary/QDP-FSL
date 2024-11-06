# https://github.com/Aemerse/Quanage-Server/blob/4ad3ef66cb4a34dff5cd0921503009232ac3ec41/api.py
"""Import Qiskit"""
from qiskit import QuantumCircuit, BasicAer, execute


def run_qasm(qasm, backend_to_run='qasm_simulator', num_shots_str='1'):
    circuit = QuantumCircuit.from_qasm_str(qasm)
    backend = BasicAer.get_backend(backend_to_run)
    job_sim = execute(circuit, backend, shots=int(num_shots_str))
    result_sim = job_sim.result()
    return result_sim.get_counts(circuit)
