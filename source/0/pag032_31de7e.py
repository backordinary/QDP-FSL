# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/pag032.py
from qiskit import (IBMQ, Aer, ClassicalRegister, QuantumCircuit,
                    QuantumRegister, execute)
from qiskit.backends.ibmq import least_busy
from qiskit.tools.visualization import circuit_drawer

qr = QuantumRegister(2)
cr = ClassicalRegister(2)
qc = QuantumCircuit(qr, cr)
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.h(qr[0])
qc.h(qr[1])
qc.measure(qr, cr)
circuit_drawer(qc, filename="../imgs/pag032.png")

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend)
print(job.result().get_counts(qc))

IBMQ.load_accounts()
large_enough_devices = IBMQ.backends(
    filters=
    lambda x: x.configuration()['n_qubits'] > 2 and not x.configuration()['simulator']
)
backend = least_busy(large_enough_devices)
job = execute(qc, backend, max_credits=3)
print(job.result().get_counts(qc))
