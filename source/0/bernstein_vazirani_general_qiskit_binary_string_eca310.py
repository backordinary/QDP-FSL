# https://github.com/UST-QuAntiL/nisq-analyzer-content/blob/63b9ee5c143c08239938794080df89a17284a4f4/example-implementations/Bernstein-Vazirani/bernstein-vazirani-general-qiskit-binary-string.py
from qiskit import QuantumCircuit

# n = 3 # number of qubits used to represent s
# s = '011'   # the hidden binary string

# https://qiskit.org/textbook/ch-algorithms/bernstein-vazirani.html


def get_circuit(**kwargs):
    n = kwargs["number_of_qubits"]
    s = kwargs["s"]

    # We need a circuit with n qubits, plus one ancilla qubit
    # Also need n classical bits to write the output to
    bv_circuit = QuantumCircuit(n + 1, n)

    # put ancilla in state |->
    bv_circuit.h(n)
    bv_circuit.z(n)

    # Apply Hadamard gates before querying the oracle
    for i in range(n):
        bv_circuit.h(i)

    # Apply barrier
    bv_circuit.barrier()

    # Apply the inner-product oracle
    s = s[::-1]  # reverse s to fit qiskit's qubit ordering
    for q in range(n):
        if s[q] == '0':
            bv_circuit.iden(q)
        else:
            bv_circuit.cx(q, n)

    # Apply barrier
    bv_circuit.barrier()

    # Apply Hadamard gates after querying the oracle
    for i in range(n):
        bv_circuit.h(i)

    # Measurement
    for i in range(n):
        bv_circuit.measure(i, i)

    return bv_circuit
