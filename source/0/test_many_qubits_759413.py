# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/test_many_qubits.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer, IBMQ
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.visualization import plot_histogram
from qiskit.backends.ibmq import least_busy
import sys

qr = QuantumRegister(12)
cr = ClassicalRegister(12)
qc = QuantumCircuit(qr, cr)

qc.h(qr)
qc.measure(qr, cr)
# circuit_drawer(qc, filename="imgs/test_14_qubis.png")

if (len(sys.argv) > 1 and sys.argv[1] == 'c'):  # c stands for cloud
    IBMQ.load_accounts()
    print("Account loaded")
    if (sys.argv[2] == 'r'):  # r stands for real
        large_enough_devices = IBMQ.backends(
            filters=
            lambda x: x.configuration()['n_qubits'] >= 14 and not x.configuration()['simulator']
        )
        backend = least_busy(large_enough_devices)
        shots = 1024
        mx = 3
    elif (sys.argv[2] == 's'):  # shoud be a s for simulator
        large_enough_devices = IBMQ.backends(
            filters=
            lambda x: x.configuration()['n_qubits'] >= 14 and x.configuration()['simulator']
        )
        backend = least_busy(large_enough_devices)
        shots = 8192
        mx = 10
    else:
        exit("s/r")
else:
    print("Local simulator started")
    print(Aer.backends())
    backend = Aer.get_backend('qasm_simulator')
    shots = 4096
    mx = 3
print("backend = {0}".format(backend))
print("shots = {0}".format(shots))
print("max credits = {0}".format(mx))
job = execute(qc, backend=backend, shots=shots, max_credits=mx)
print("job id = {0}".format(job.job_id()))
result = job.result()
counts = result.get_counts()
print(counts)
#plot_histogram(counts)
