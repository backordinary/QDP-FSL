# https://github.com/1chooo/Programming-Evolution/blob/ab8c8e388ab098163eebd736d47fe9a559ad1090/NCU/sophomore/CE3005/alg/quantum/CH01/prog_04.py
from qiskit import QuantumCircuit, transpile, execute 
from qiskit.providers.aer import AerSimulator

sim = AerSimulator()
qc = QuantumCircuit(1, 1)
qc.measure([0], [0])
print(qc)
cqc = transpile(qc, sim)
job=execute(cqc, backend = sim, shots = 1000) 
result = job.result()
counts = result.get_counts(qc)
print("Total counts for qubit states are:", counts)