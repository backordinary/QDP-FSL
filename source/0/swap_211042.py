# https://github.com/riZZZhik/quantum_faces/blob/ce4b59d5fa73209531eb8f0532d867902a35deca/quantum/swap.py
from qiskit import execute


def swap(circuit, q0, q1, q2, classic, backend, num_of_shots):
    circuit.tdg(q0[0])
    circuit.h(q0[0])

    for i in range(len(q1)):
        circuit.cswap(q0[0], q1[i], q2[i])

    circuit.h(q0[0])
    circuit.tdg(q0[0])

    circuit.measure(q0[0], classic)

    result = execute(circuit, backend, shots=num_of_shots).result()

    return result.get_counts(circuit)
