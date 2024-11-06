# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/grover_thru.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer
from qiskit.tools.visualization import circuit_drawer


# Testing z gate
# If we apply a uniform input superposition, after z we should get 1 100%
def test0():
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)
    qc.z(qr)
    qc.h(qr)
    qc.measure(qr, cr)
    circuit_drawer(qc, filename="imgs/grover_thru.png")

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend)
    result = job.result()
    print(result.get_counts(qc))


def main():
    test0()


if __name__ == "__main__":
    main()
