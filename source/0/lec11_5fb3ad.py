# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/lec11.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.tools.visualization import circuit_drawer
import sys

img_path = "imgs/lec11.png"


def create_circuit():
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)

    qc = QuantumCircuit(qr, cr)

    qc.h(qr[0])
    qc.h(qr[1])
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


def letsgo():
    l = len(sys.argv[1:])
    if (l < 1):
        sys.exit("error, missing argument (backend r/s/b, [draw 0/1])")

    qc = create_circuit()

    if (l > 1 and sys.argv[2] == '1'):
        circuit_drawer(qc, filename=img_path)
    if (sys.argv[1] == 's'):
        print("Running simulation")
        backend = local_simulate(qc)
        credits = 10
        run_circuit(qc, backend, 4096, credits)
    elif (sys.argv[1] == 'r'):
        print("Running for real")
        backend = online_real(qc)
        credits = 3
        run_circuit(qc, backend, 4096, credits)
    elif (sys.argv[1] == 'b'):
        print("Running simulation")
        backend = local_simulate(qc)
        credits = 10
        run_circuit(qc, backend, 4096, credits)
        print("Running both")
        backend = online_real(qc)
        credits = 3
        run_circuit(qc, backend, 4096, credits)

    else:
        sys.exit("error, available parameters are 'r' 's' or 'b'")


if __name__ == "__main__":
    letsgo()
