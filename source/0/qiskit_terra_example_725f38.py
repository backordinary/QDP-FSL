# https://github.com/zfazal10/CPE322/blob/6d3808c85bda93e7280ef8a8efd1dc2085ec651f/Lab%209/qiskit_terra_example.py
# https://en.wikipedia.org/wiki/Qiskit
# https://qiskit.org/terra

from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(2, 2)

qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])

backend = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend)
sim_result = job_sim.result()

print(sim_result.get_counts(qc))
