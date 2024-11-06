# https://github.com/anott03/Quantum/blob/957b62d15e2143ea7f37151cac2c2f7c9f7fdd23/adder.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

# adder circuit
# input encoded in qubits 0 and 1
qc = QuantumCircuit(4, 2)

# define test input
qc.x(0)
qc.x(1)

# applying CNOT twice has the effect of encoding the
# result of an  XOR on qubits 0 and 1 in qubit 2
qc.cx(0, 2)
qc.cx(1, 2)
# the toffoli (ccx) gate will perform NOT on qubit 3
# when qubits 0 and 1 are both 1. aka if (0 and 1)
# then qc.x(3)
qc.ccx(0, 1, 3)

qc.measure([2, 3], [0, 1])

sim = AerSimulator()
job = sim.run(qc)
result = job.result()
print(result.get_counts())
