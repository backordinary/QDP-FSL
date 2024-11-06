# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/lec05.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.tools.visualization import circuit_drawer
import sys


def create_circuit():
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)

    qc = QuantumCircuit(qr, cr)

    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.x(qr[1])
    qc.h(qr[0])
    qc.measure(qr, cr)
    return qc


def run_circuit(qc, backend, shots, max_credits=10):
    job = execute(qc, backend=backend, shots=shots, max_credits=max_credits)
    result = job.result()
    print(result.get_counts(qc))


def local_simulate(qc):
    backend = Aer.get_backend('qasm_simulator')
    return backend


def online_real(qc):
    from qiskit import IBMQ
    from qiskit.backends.ibmq import least_busy
    IBMQ.load_accounts()
    large_enough_devices = IBMQ.backends(
        filters=
        lambda x: x.configuration()['n_qubits'] > 2 and not x.configuration()['simulator']
    )

    backend = least_busy(large_enough_devices)
    return backend


l = len(sys.argv[1:])
if (l < 1):
    print("error, missing argument (backend r/s, [draw 0/1])")
    exit()

qc = create_circuit()

if (l > 1 and sys.argv[2] == '1'):
    circuit_drawer(qc, filename="imgs/lec05.png")
if (sys.argv[1] == 's'):
    backend = local_simulate(qc)
    credits = 10
elif (sys.argv[1] == 'r'):
    backend = online_real(qc)
    credits = 3
else:
    exit
run_circuit(qc, backend, 4096, credits)
