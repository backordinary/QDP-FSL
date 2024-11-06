# https://github.com/zillerium/shoro/blob/d1d4d7e0959921b800aa48107503cddbf91348b2/test1.py
import qiskit as qk
from qiskit import QuantumProgram

qp = QuantumProgram()
qr = qp.create_quantum_register('qr', 2)
cr = qp.create_classical_register('cr',2)
qc = qp.create_circuit('qc', [qr], [cr])
circuit = qp.get_circuit('qc')
quantum_r = qp.get_quantum_register('qr')
classical_r = qp.get_classical_register('cr')

circuit.x(quantum_r[0])
circuit.y(quantum_r[1])

circuit.cx(quantum_r[0], quantum_r[1])

circuit.measure(quantum_r[0], classical_r[0])
circuit.measure(quantum_r[1], classical_r[1])

backend = 'local_qsam_simulator'
circuits = ['qc']

result = qp.execute(circuit, backend, wait=2, timeout = 240)
print(result)
result.get_counts('qc')
out = result.get_ran_qasm('qc')
print(out)
