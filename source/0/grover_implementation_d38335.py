# https://github.com/alexnels43/grovers-quantum-implementation/blob/908091bfffd77f4a40b471bcb5b573e4bdc5c2a6/grover_implementation.py
# Alex Nelson

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, BasicAer, Aer, execute
from qiskit.visualization import plot_state_qsphere
from qiskit.tools.visualization import plot_histogram
from IPython.display import display
from math import sqrt, pow, pi
import numpy as np


def ket_notation(vector: str):
    ket = '\u27E9'
    return '|' + vector + ket


def print_qsphere(circuit):
    qsphere = execute(circuit, Aer.get_backend('statevector_simulator')).result().get_statevector(circuit)
    display(plot_state_qsphere(qsphere))


def print_unitary_matrix(circuit):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    unitary = execute(circuit, BasicAer.get_backend('unitary_simulator')).result().get_unitary(circuit)
    print("Unitary matrix:\n" + str(unitary.real))


def initialize_oracle(oracle, quantum_register, oracle_vector):
    num_qubits = oracle.num_qubits
    # Applying Pauli-X gates to 0 values of our oracle vector
    reversed_oracle_vector = oracle_vector[::-1]
    for n in range(0, len(oracle_vector)):
        if reversed_oracle_vector[n] == '0':
            oracle.x(quantum_register[n])
    # Applying a Hadamard gate to our final qubit
    oracle.h(quantum_register[num_qubits - 1])
    # Applying a controlled NOT gate with our final qubit as the target
    if num_qubits == 2: oracle.cx(quantum_register[0], quantum_register[1])
    if num_qubits == 3: oracle.ccx(quantum_register[0], quantum_register[1], quantum_register[2])
    if num_qubits == 4: oracle.mcx([quantum_register[0], quantum_register[1], quantum_register[2]], quantum_register[3])
    if num_qubits == 5: oracle.mcx([quantum_register[0], quantum_register[1], quantum_register[2], quantum_register[3]], quantum_register[4])
    # Applying a Hadamard gate to our final qubit
    oracle.h(quantum_register[num_qubits - 1])
    # Applying Pauli-X gates to 0 values of our oracle vector
    for n in range(0, len(oracle_vector)):
        if reversed_oracle_vector[n] == '0':
            oracle.x(quantum_register[n])
    # Adding a bar for ease of visualization
    oracle.barrier(quantum_register)
    return oracle


def initialize_amplifier(amplifier, quantum_register):
    num_qubits = amplifier.num_qubits
    # Applying a Hadamard to all qubits
    amplifier.h(quantum_register)
    # Applying a NOT to all qubits
    amplifier.x(quantum_register)
    # Applying a Hadamard to the final qubit
    amplifier.h(quantum_register[num_qubits - 1])
    # Applying a controlled NOT gate with our final qubit as the target
    if num_qubits == 2: amplifier.cx(quantum_register[0], quantum_register[1]);
    if num_qubits == 3: amplifier.ccx(quantum_register[0], quantum_register[1], quantum_register[2])
    if num_qubits == 4: amplifier.mcx([quantum_register[0], quantum_register[1], quantum_register[2]], quantum_register[3])
    if num_qubits == 5: amplifier.mcx([quantum_register[0], quantum_register[1], quantum_register[2], quantum_register[3]], quantum_register[4])
    # Applying a Hadamard to the final qubit
    amplifier.h(quantum_register[num_qubits - 1])
    # Applying a NOT to all qubits
    amplifier.x(quantum_register)
    # Applying a Hadamard to all qubits
    amplifier.h(quantum_register)
    # Adding a bar for ease of visualization
    amplifier.barrier(quantum_register)
    return amplifier


def initialize_grover(grover, quantum_register):
    # Initiate the Grover with Hadamards
    grover.h(quantum_register)
    grover.barrier(quantum_register)
    display(grover.draw(output="mpl"))
    print_qsphere(grover)
    return grover


def build_grover(grover, oracle, amplifier, quantum_register, classic_register):
    ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
    search_space = pow(2, oracle.num_qubits)
    number_of_iterations = int(pi / 4 * (sqrt(search_space)))
    step = 1
    for n in range(number_of_iterations):
        print('Step ' + str(step))
        print('Apply the oracle for the ' + ordinal(n + 1) + ' time')
        step += 1
        grover.compose(oracle, inplace=True)
        display(grover.draw(output="mpl"))
        print_qsphere(grover)
        print('=======================================================================')
        print('Step ' + str(step))
        print('Apply the amplifier for the ' + ordinal(n + 1) + ' time')
        step += 1
        grover.compose(amplifier, inplace=True)
        display(grover.draw(output="mpl"))
        print_qsphere(grover)
        print('=======================================================================')

    grover.measure(quantum_register, classic_register)
    return grover


def main():
    # Edit these variables to change the generated algorithm steps
    num_qubits = 4
    oracle_vector = '0101'

    quantum_register = QuantumRegister(num_qubits)
    classical_register = ClassicalRegister(num_qubits)

    oracle = QuantumCircuit(quantum_register, classical_register)
    oracle = initialize_oracle(oracle, quantum_register, oracle_vector)
    print("Oracle circuit for " + ket_notation(str(oracle_vector)))
    display(oracle.draw(output="mpl"))
    print_unitary_matrix(oracle)

    amplifier = QuantumCircuit(quantum_register, classical_register)
    amplifier = initialize_amplifier(amplifier, quantum_register)
    display(amplifier.draw(output="mpl"))
    print_unitary_matrix(amplifier)

    grover = QuantumCircuit(quantum_register, classical_register)
    display(grover.draw(output="mpl"))
    print_qsphere(grover)

    grover = initialize_grover(grover, quantum_register)
    grover = build_grover(grover, oracle, amplifier, quantum_register, classical_register)
    display(grover.draw(output="mpl"))

    print("Statistical distribution of our search for " + ket_notation(
        str(oracle_vector)) + " on a quantum computer emulator\n")
    simulation_results = execute(grover, Aer.get_backend('qasm_simulator')).result().get_counts()
    display(plot_histogram(simulation_results))


if __name__ == '__main__':
    main()
