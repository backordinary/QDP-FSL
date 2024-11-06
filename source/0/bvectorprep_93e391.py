# https://github.com/Cmollan/QuantumAlgorithms/blob/63a94deed901d8de913aaef6c6ee96f4ac86669d/bVectorPrep.py
# Code created by Calahan Mollan

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from QuantumFourierTransform import quantum_register_swap

## Remove these imports once testing is done
from qiskit import *
from qiskit.visualization import plot_histogram


### Functions for creating the amplitude vector
def State_Vector(bin_order, basis):
    # SUMMARY: Recursively create the state vector for the specified bit string in the provided basis
    # INPUTS: bin_order - bit string corresponding to the number of the state e.g. '01101' in the standard basis is the
    #                     13th state vector, |13>, |01101>
    #         basis - Dictionary of keys of 0 or 1 and corresponding values in numpy arrays
    # OUTPUTS: State_Vector - Numpy array corresponding to the specified bit string in the provided basis

    if len(bin_order) > 1:
        return np.kron(
            State_Vector(bin_order[:-1], basis),
            basis[bin_order[-1]]
        )
    else:
        return basis[bin_order]


def Standard_State_Vector(bin_order):
    # SUMMARY: Wrapper function for the State_Vector using the standard basis
    # INPUTS: bin_order - bit string corresponding to the number of the state e.g. '01101' in the standard basis is the
    #                     13th state vector, |13>, |01101>
    # OUTPUTS: State_Vector - Numpy array corresponding to the specified bit string in the provided basis

    basis = {
        '0': np.array([1, 0]),
        '1': np.array([0, 1])
    }
    return State_Vector(bin_order, basis)


def Create_Standard_State_Vectors(bits):
    # SUMMARY: Create all state vectors in the standard basis in the number of bits provided (2**number of bits)
    # INPUTS: bits - number of bits
    # OUTPUTS: state_vectors - dictionary of keys of the bit string of the state vector and values of the numpy array

    state_vectors = {}

    for i in range(2 ** bits):
        binary_num = format(i, '0' + str(bits) + 'b')  # key
        state_vectors[str(binary_num)] = Standard_State_Vector(binary_num)  # value

    return state_vectors


def AmplitudeEncode(b):
    # SUMMARY: High-level function for giving the encoding of the b vector provided as defined by HHL
    # INPUTS: b - numpy array which to encode
    # OUTPUTS: ans - numpy array of amplitudes

    n_bits = math.ceil(math.log(b.size, 2))

    state_vectors = Create_Standard_State_Vectors(n_bits)

    ans = np.zeros(1)  # initializing

    magnitude = np.linalg.norm(b)

    for i in range(b.size):
        dict_key = format(i, '0' + str(n_bits) + 'b')
        temp_vector = b[i] * state_vectors[dict_key] / magnitude
        ans = ans + temp_vector

    return ans


### Functions for implementing Long and Sun's procedure
def U_theta(circuit, theta, index):
    # SUMMARY: Adds a unitary gate to the given quantum circuit as defined by Long and Sun
    # INPUTS: circuit - Qiskit QuantumCircuit object to add the gate to
    #         theta - Angle of rotation, parameter of the gate in radians
    #         index - index of the qubit to which to apply the gate
    # OUTPUTS: circuit - Qiskit QuantumCircuit with gate applied

    cx = Operator([
        [math.cos(theta), math.sin(theta)],
        [math.sin(theta), -math.cos(theta)]
    ])
    circuit.unitary(cx, [index], label='U_theta')


def Controlled_U_theta(circuit, control_indices, rotation_index, theta, control_int):
    # SUMMARY: Adds a controlled^k unitary gate to the given quantum circuit as defined by Long and Sun
    # INPUTS: circuit - Qiskit QuantumCircuit object to add the gate to
    #         control_indices - a list of qubit indices to use to control the gate
    #         rotation_index - list of the index of the qubit to be effected by the gate
    #         theta - Angle of rotation, parameter of the gate in radians
    #         control_int - either a bitstring or its corresponding integer representing if the control is 0 or 1
    # OUTPUTS: circuit - Qiskit QuantumCircuit object with gate applied

    temp = QuantumCircuit(1)
    U_theta(temp, theta, 0)

    Ck_U_theta = temp.to_gate().control(num_ctrl_qubits=len(control_indices), label='Ck_U', ctrl_state=control_int)

    circuit.append(Ck_U_theta, control_indices + rotation_index)


def U_alpha(circuit, a0, a1, index):
    # SUMMARY: Adds a unitary gate to the given quantum ciruit as defined by Long and Sun
    # INPUTS: circuit - Qiskit QuantumCircuit object to add the gate to
    #         a0 - parameter for the gate that is related to the main diagonal, from a bitstring with a 0 at the end
    #         a1 - parameter for the gate that is related to the off diagonal, from a bitstring with a 1 at the end
    #         index - index of the qubit to which to apply the gate
    # OUTPUTS: circuit - Qiskit QuantumCircuit object with gate applied

    denom = math.sqrt(abs(a0)**2 + abs(a1)**2)  # denominator, the 2 norm of a0 and a1

    cx = Operator([
        [a0/denom, a1/denom],
        [np.conj(a1)/denom, -np.conj(a0)/denom]
    ])
    circuit.unitary(cx, [index], label='U_alpha')


def Controlled_U_alpha(circuit, contol_indices, rotation_index, a0, a1, control_int):
    # SUMMARY: Adds a controlled^k unitary gate to the given quantum circuit as defined by Long and Sun
    # INPUTS: circuit - Qiskit QuantumCircuit object to add the gate to
    #         control_indices - a list of qubit indices to use to control the gate
    #         rotation_index - list of the index of the qubit to be effected by the gate
    #         a0 - parameter for the gate that is related to the main diagonal, from a bitstring with a 0 at the end
    #         a1 - parameter for the gate that is related to the off diagonal, from a bitstring with a 1 at the end
    #         control_int - either a bitstring or its corresponding integer representing if the control is 0 or 1
    # OUTPUTS: circuit - Qiskit QuantumCircuit object with gate applied

    temp = QuantumCircuit(1)
    U_alpha(temp, a0, a1, 0)

    Ck_U_alpha = temp.to_gate().control(num_ctrl_qubits=len(contol_indices), label='Ck_U_a', ctrl_state=control_int)

    circuit.append(Ck_U_alpha, contol_indices + rotation_index)


def AddEncoding(circuit, start_qubit, stop_qubit, b):
    # SUMMARY: High-level function to apply the encoding of the amplitudes of the b vector (HHL) by using the process
    #          defined by Long and Sun
    # INPUTS: circuit - Qiskit QuantumCircuit object to add the gate to
    #         start_qubit - index of the start of the encoding
    #         stop_qubit - index of the end of the encoding
    #         b - numpy array which to encode
    # OUTPUTS: circuit - Qiskit QuantumCircuit object with the process added

    b_amplitudes = AmplitudeEncode(b)  # Convert to the normalized amplitudes as defined by HHL

    n_qubits = stop_qubit - start_qubit + 1

    # Error Checking
    if 2 ** n_qubits != b_amplitudes.size:
        raise qiskit.QiskitError('Size of b matrix does not match number of qubits.')
        return

    for qubit in range(n_qubits):  # Iterate over the number of qubits to apply the encoding over

        if qubit == 0:  # 1st qubit has just a rotation, not a controlled rotation
            ## Theta Generation
            ssq_num = 0
            ssq_denom = 0

            for i in range(int(b_amplitudes.size/2)):
                ssq_num += b_amplitudes[b_amplitudes.size - i - 1]**2
                ssq_denom += b_amplitudes[i]**2

            if ssq_num == 0 and ssq_denom == 0:  # 0/0 corresponds to an angle of 0 according to Long and Sun
                theta = 0
            else:
                theta = math.atan(math.sqrt(
                    ssq_num / ssq_denom
                ))

            ## Making the Gate
            U_theta(circuit, theta, start_qubit)

        elif qubit != n_qubits - 1:  # Intermediate qubits

            ## Making the control indices list
            control_indices = []
            for temp in range(qubit + start_qubit, 0, -1):
                control_indices.append(temp - 1)  # Putting them in backwards because qiskit is weird

            for j in range(2**qubit):  # make the k controlled^k rotation gates

                ## Theta Generation
                # Bit strings are used because of the way the summations are writen
                prefix = format(j, '0' + str(qubit) + 'b')  # Start of the bit string, should match the control indices

                len_suffix = n_qubits - (len(prefix) + 1)  # length(prefix + (0 or 1) + suffix) = number of qubits

                start_num = prefix + '1' + format(0, '0' + str(len_suffix) + 'b')
                end_num = prefix + '1' + format(2 ** (len_suffix - 1), '0' + str(len_suffix) + 'b')
                start_denom = prefix + '0' + format(0, '0' + str(len_suffix - 1) + 'b')
                end_denom = prefix + '0' + format(2 ** (len_suffix - 1), '0' + str(len_suffix) + 'b')

                ssq_num = 0
                ssq_denom = 0
                for i in range(int(start_num, 2), int(end_num, 2) + 1):
                    ssq_num += b_amplitudes[i]**2
                for i in range(int(start_denom, 2), int(end_denom, 2) + 1):
                    ssq_denom += b_amplitudes[i]**2

                if ssq_num == 0 and ssq_denom == 0:  # 0/0 corresponds to an angle of 0 according to Long and Sun
                    theta = 0
                else:
                    theta = math.atan(math.sqrt(
                        ssq_num / ssq_denom
                    ))

                ## Making the Gate
                Controlled_U_theta(circuit, control_indices, [qubit + start_qubit], theta, j)

        else:  # Last qubit

            ## Making the control indices
            control_indices = []
            for temp in range(stop_qubit, start_qubit, -1):
                control_indices.append(temp - 1)  # Putting them in backwards because qiskit is weird

            for j in range(2**qubit):  # make the k controlled^k rotation gates

                ## a0 and a1 generation
                prefix = format(j, '0' + str(qubit) + 'b')

                a0 = b_amplitudes[int(prefix + '0', 2)]
                a1 = b_amplitudes[int(prefix + '1', 2)]

                ## Making the Gate
                if a0 == 0 and a1 == 0:  # if both a0 and a1 are 0, the matrix for U_alpha is not unitary.
                    a0 = 1  # a0 = 1 and a1 = 0 is the identity matrix

                Controlled_U_alpha(circuit, control_indices, [stop_qubit], a0, a1, j)

    ## Swapping MSB to LSB because qiskit
    quantum_register_swap(circuit, start_qubit, stop_qubit)

if __name__ == "main":
    # test = np.array([1, 1, 1, 1])

    test = np.array([2, 4, 3, 1])

    # test = np.array([1, 5, 4, 3, 1, 2, 7, 8])

    # test = np.array([58, 65, 17, 54])

    # test = np.array([85, 87, 117, 76, 110, 140, 104, 87,  68,  76, 102, 78, 56, 48, 97, 74])

    prob = AmplitudeEncode(test)**2
    qc = QuantumCircuit(2)
    AddEncoding(qc, 0, 1, test)
    qc.measure_all()
    qc.draw('mpl')

    backend = Aer.get_backend('qasm_simulator')
    shots = 262144
    results = execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()
    plot_histogram(answer)

