# https://github.com/timothygao8710/Quantum-Computing-Course/blob/add45485678b80df2746491ad413ea32ac003275/QiskitExercises/Lab12/lab12.py
# Lab 12: Deutsch-Jozsa Algorithm
# Copyright 2021 The MITRE Corporation. All Rights Reserved.

#Not showing qiskit import
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import Aer


def always_zero(circuit, input, output):
    """
    This oracle always "returns" zero, so it never phase-flips the output
    qubit.

    Parameters:
        circuit (QuantumCircuit): The circuit being constructed.
        input (QuantumRegister): The input register to evaluate.
        output (QuantumRegister): The output (result) qubit to flip.
    """


def always_one(circuit, input, output):
    """
    This oracle always "returns" one, so it always phase-flips the output
    qubit.

    Parameters:
        circuit (QuantumCircuit): The circuit being constructed.
        input (QuantumRegister): The input register to evaluate.
        output (QuantumRegister): The output (result) qubit to flip.
    """

    circuit.z(output)


def exercise_1(circuit, input, output):
    """
    In this exercise, you are given a register with an unknown number of
    qubits, in an unknown state, and an output qubit in the |1> state.
    Your goal is to construct an oracle that will flip the phase of the
    output qubit if the register contains an odd number of qubits in the
    |1> state, and leave the output qubit alone if the register contains
    an even number of qubits in the |1> state.
    
    For example, if the register was in the state |10101>, you would
    phase-flip the output qubit because there was an odd number of |1>
    qubits. If the register was in the state |01010>, you would leave the
    output qubit alone.

    Parameters:
        circuit (QuantumCircuit): The circuit being constructed.
        input (QuantumRegister): A register of qubits in an unknown state.
            It could be in an arbitrary superposition.
        output (QuantumRegister): An output qubit that you must flip if the
            oracle's conditions are met. It will be in the |1> state to start;
            you must put it in the -|1> state if the register has an odd number 
            of qubits in the |1> state.
    """

    for i in input:
        circuit.cz(i, output)

def exercise_2(circuit, first_index, second_index, input, output):
    """
    In this exercise, you are given a register with an unknown number of
    qubits, in an unknown state, and an output qubit in the |1> state. You
    are also given two indices of qubits in the register. Your goal is to
    construct an oracle that will phase-flip the output qubit if the two
    qubits with the given indices have different parity; that is, if both
    qubits are in the same state, you should leave the output qubit alone.
    If the qubits are in opposite states, you should phase-flip the
    output.

    Parameters:
        circuit (QuantumCircuit): The circuit being constructed.
        first_index (int): The index in the register of the first qubit to 
            use in the parity check.
        second_index (int): The index in the register of the second qubit
            to use in the parity check.
        input (QuantumRegister): A register of qubits in an unknown state.
            It could be in an arbitrary superposition.
        output (QuantumRegister): An output qubit that you must flip if the
            oracle's conditions are met. It will be in the |1> state to start;
            you must put it in the -|1> state if the two qubits in the register
            at the given indices have opposite parity.
    """
    

    circuit.cz(input[first_index], output)
    circuit.cz(input[second_index], output)


def exercise_3(input_length, oracle):
    """
    In this exercise, you will implement the Deutsch-Jozsa algorithm. You
    are given an oracle, which is an operation that takes in a qubit
    register and an output qubit that will be phase-flipped if the
    register meets the oracle's condition. The register must be in the
    |+...+> state, and the output qubit must be in the |1> state. Your
    goal is to prepare the input register and output qubit, run the oracle,
    and use the resulting state of the register to determine whether the
    oracle represents a constant function or a balanced function.

    Parameters:
        input_length (int): The number of qubits that the oracle expects the
            input register to contain. You must allocate a register with this many 
            qubits to provide to the oracle.
        oracle (function): An operation that represents some quantum function.
            It will take in a circuit, an input register (that must be in the |+...+> 
            state) and an output qubit (that must be in the |1> state). It will phase-flip
            the output qubit for each state in the register's superposition that meets
            its criteria, causing that state to become phase-flipped as well.
    """

    qubits = QuantumRegister(input_length)
    measurement = ClassicalRegister(input_length)
    circuit = QuantumCircuit(qubits, measurement)
    ancilla = QuantumRegister(1, "ancilla")
    if not circuit.has_register(ancilla):
        circuit.add_register(ancilla)

    for i in qubits:
        circuit.h(i)
    circuit.x(ancilla)

    oracle(circuit, qubits, ancilla)

    for i in qubits:
        circuit.h(i)

    circuit.measure(qubits, measurement)

    simulator = Aer.get_backend('aer_simulator')
    simulation = execute(circuit, simulator, shots=1)
    result = simulation.result()

    counts = result.get_counts(circuit)

    for(measured_state, count) in counts.items():
        big_endian_state = measured_state[::-1]
        #True if constant
        if(int(big_endian_state) == 0):
            return True
        else:
            return False

    