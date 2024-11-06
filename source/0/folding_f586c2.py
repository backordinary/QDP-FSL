# https://github.com/GaiaBolognini/Zero-Noise-Extrapolation/blob/93cc2238cc33ff2e666c5c279670bacc2738309f/Projects/Coding/Folding.py
#Implementing the folding techniques for the circuit: layer and circuit folding
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
import numpy as np
import random

def circuit_folding (circuit, scaling=1):
    """
    Function that takes a circuit as a input and apply the circuit folding, considering the scaling factor
    Args: circuit = circuit to be folded
          scaling = scaling factor (1=do not fold), (3=fold one time the whole circuit)
    return: the circuit folded
    """
    #properties of the circuit
    n_qubits = circuit.width()
    qubits = list(np.arange(n_qubits))

    #retrieving from the scaling factor the number of times we need to apply the circuit
    ratio = (scaling-1)/2
    #number of times we need to fold the whole circuit
    n = int(ratio)
    #number of last layers to be folded again
    s = int((ratio-n)*circuit.size())

    circuit_folded = circuit.copy()
    c_inv = circuit_folded.inverse()
    circuit_folded.barrier()

    for i in range(n):
        circuit_folded = circuit_folded.compose(c_inv)
        circuit_folded.barrier()
        circuit_folded = circuit_folded.compose(circuit)

    #creating a partial circuit, made only of the last s layers
    if (s != 0):
        c_partial = QuantumCircuit(QuantumRegister(n_qubits))
        for j in range(circuit.size() - s, circuit.size(), 1):
            tuple = circuit.data[j][0]
            arg = circuit.data[j][1]
            args=[]
            for k in range(len(arg)):
              args.append(arg[k].index)
            c_partial.append(tuple, qargs=list(args))

        circuit_folded = circuit_folded.compose(c_partial.inverse())
        circuit_folded.barrier()
        circuit_folded = circuit_folded.compose(c_partial)

    return circuit_folded

def gate_folding (circuit, scaling=1, way='Left'):
    """
    Function that takes a circuit as a input and apply the gate folding, considering the scaling factor
    Args: circuit = circuit to be folded
          scaling = scaling factor (1=do not fold), (3=fold one time the whole circuit)
          way = which way to fill the circuit for s values: Left, Right, Random
    return: the circuit folded
    """
    # properties of the circuit
    n_qubits = circuit.width()
    n_gates = circuit.size()
    qubits = list(np.arange(n_qubits))

    # retrieving from the scaling factor the number of times we need to fold completely each single gate
    ratio = (scaling - 1) / 2
    # number of times we need to fold the whole circuit
    n = int(ratio)
    # number layers to be folded again one more time
    s = int((ratio - n) * n_gates)


    list_gates = list(np.arange(n_gates))
    random_gates = random.sample(list_gates, s)

    circuit_folded = circuit.copy()
    count = 1

    for i in range(n_gates):
        instruction = circuit.data[i]

        # computing the inverse and storing it into another instruction
        instruction_inverse = list(instruction)
        instruction_inverse[0] = instruction[0].inverse()
        instruction_inverse = tuple(instruction_inverse)

        for k in range(n):

            #folding the circuit for n times
            circuit_folded.data.insert(i+count, instruction_inverse)
            count += 1
            circuit_folded.data.insert(i + count, instruction)
            count += 1

        if (way=='Left'):
            if (i < s):
                circuit_folded.data.insert(i + count, instruction_inverse)
                count += 1
                circuit_folded.data.insert(i + count, instruction)
                count += 1

        if (way=='Right'):
            if (i >= (n_gates-s)):
                circuit_folded.data.insert(i + count, instruction_inverse)
                count += 1
                circuit_folded.data.insert(i + count, instruction)
                count += 1

        if (way=='Random'):
            if (i in random_gates):
                circuit_folded.data.insert(i + count, instruction_inverse)
                count += 1
                circuit_folded.data.insert(i + count, instruction)
                count += 1


    return circuit_folded

