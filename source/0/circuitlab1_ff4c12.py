# https://github.com/jesse318s/labs-with-qiskit-python/blob/1f077559af27137ac866420bf1cc3acb710dc431/CircuitLab1.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit.quantum_info import Statevector
from qiskit import transpile 
from qiskit.providers.aer import QasmSimulator

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])

state = Statevector.from_int(0, 2**3)
state = state.evolve(circuit)

meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
meas.measure(range(3), range(3))
qc = meas.compose(circuit, range(3), front=True)

backend = QasmSimulator()
qc_compiled = transpile(qc, backend)
job_sim = backend.run(qc_compiled, shots=1024)
result_sim = job_sim.result()

counts = result_sim.get_counts(qc_compiled)
print(counts)