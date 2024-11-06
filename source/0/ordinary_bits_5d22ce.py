# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/classical_comp/ordinary_bits.py
from qiskit import QuantumCircuit, execute, Aer
n = 8
n_q = n
n_b = n
qc_output = QuantumCircuit(n_q, n_b)

for j in range(n):
    qc_output.measure(j, j)

counts = execute(qc_output, Aer.get_backend('qasm_simulator')).result().get_counts()

print(qc_output)
print(counts)