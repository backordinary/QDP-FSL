# https://github.com/drobiu/quantum-project/blob/26a2a49cabb0868c9bf5ad68e6861062da7ad6ea/src/arithmetic/add.py
from math import pi
from qiskit import *
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit.circuit.library.basis_change import QFT


def add(circuit, a, b, control=None, apply_QFT=True, amount=1):
    # Function adds a to b Original add function build by
    # https://github.com/TimVroomans/Quantum-Mastermind/blob/master/src/mastermind/arithmetic/dradder.py
    number_a = len(a)
    number_b = len(b)
    num = number_b + number_a

    if number_a > number_b:
        raise ValueError("Amount of registers in b must be larger than a")

    # QFT(optional)
    if apply_QFT:
        circuit = circuit.compose(QFT(num_qubits=number_b, approximation_degree=0, do_swaps=True,
                                      inverse=False, insert_barriers=True, name='qft'), [*b])

    # Actual add loop
    for i in range(number_a):
        for j in range(number_b - i):
            if control:
                p_gate = PhaseGate(amount * pi / 2 ** (number_b - i - j - 1)).control(len(control) + 1)
                circuit.append(p_gate, [*control, a[i], b[j]])
            else:
                circuit.cp(amount * pi / 2 ** (number_b - i - j - 1), a[i], b[j])

    # Inverse QFT (optional)
    if apply_QFT:
        circuit = circuit.compose(QFT(num_qubits=number_b, approximation_degree=0, do_swaps=True,
                                      inverse=True, insert_barriers=True, name='iqft'), [*b])

    return circuit

# def c_add(circuit, a, b, control, apply_QFT=True, amount=1):
#


if __name__ == "__main__":
    # test
    a = QuantumRegister(3)
    b = QuantumRegister(5)
    qc = QuantumCircuit(a, b)
    test = add(qc, a, b)
    print(test.draw(output='text'))
