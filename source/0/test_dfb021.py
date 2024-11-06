# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/qinspire/test.py
from getpass import getpass
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from quantuminspire.qiskit import QI


def get_authentication():
    """Gets the authentication for connecting to the
    Quantum Inspire API.
    """

    email = "jomera6869@civikli.com"

    password = "wellecacusci"
    return email, password


if __name__ == "__main__":

    if "authentication" not in vars().keys():
        authentication = get_authentication()

    QI.set_authentication_details(*authentication)
    qi_backend = QI.get_backend("QX single-node simulator")

    q = QuantumRegister(2)
    b = ClassicalRegister(2)
    circuit = QuantumCircuit(q, b)
    circuit.h(q[0])
    circuit.cx(q[0], q[1])
    circuit.measure(q, b)

    qi_job = execute(circuit, backend=qi_backend, shots=256)

    qi_result = qi_job.result()

    histogram = qi_result.get_counts(circuit)
    print("\nState\tCounts")
    [print("{0}\t{1}".format(state, counts)) for state, counts in histogram.items()]
