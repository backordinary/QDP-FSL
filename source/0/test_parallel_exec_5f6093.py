# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/test_parallel_exec.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer
from qiskit.tools.visualization import circuit_drawer
from qiskit.backends.jobstatus import JOB_FINAL_STATES

qc_list = []
n_qubits = 5
for i in range(n_qubits):
    q = QuantumRegister(n_qubits)
    c = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(q, c)
    qc.h(q[i])
    qc.measure(q, c)
    qc_list.append(qc)

backend_sim = Aer.get_backend('qasm_simulator')
# qobj = compile(qc, backend_sim, shots=2000)
# job_sim = backend.run(qobj)
# Also note that the call to execute (or run) is non blocking
job_sim_list = [execute(qc, backend_sim) for qc in qc_list]

while job_sim_list:
    for job_sim in job_sim_list:
        if job_sim.status() in JOB_FINAL_STATES:
            job_sim_list.remove(job_sim)
            print(job_sim.result().get_counts())
