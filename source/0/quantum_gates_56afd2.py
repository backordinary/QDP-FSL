# https://github.com/joeyp722/Enigma/blob/d586f6c0411bc2b2b0d2a5c8d525fb3823d837ed/enigma/quantum_gates.py
# Defining several quantum gates.
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import HGate, XGate, ZGate, rz

from math import pi, ceil, log
from numpy import absolute

import enigma.sat as sat


# Diffuser gate for Grover algorithm.
def diffuser(target):

    # Define quantum circuit.
    qc = QuantumCircuit(len(target), name = 'Diffuser')

    # Execute Hadamard and X gates for all target qubits.
    for i in range(len(target)):
        qc.h(i)
        qc.x(i)

    # Perform a phase flip when all target qubits are in the 1 state.
    gate = ZGate().control(len(target)-1)
    qc.append(gate, list(range(0, len(target))))

    # Execute Hadamard and X gates for all target qubits.
    for i in range(len(target)):
        qc.x(i)
        qc.h(i)

    return qc.to_gate()

# Performs a controlled X gate based on the output for n-bit OR gate for the input literals.
def or_gate_x(input, target):

    # Define quantum circuit.
    qc = QuantumCircuit(len(input)+1, name = 'Or_gate_x')

    # Perform a bit flip on target qubit.
    qc.x(len(input))

    # Performing necessary negations.
    for i in range(len(input)):
        qc.i(i) if input[i] < 0 else qc.x(i)

    # Peform the required controlled X gate.
    gate = XGate().control(len(input))
    qc.append(gate, list(range(0, len(input)+1)))

    # Performing necessary negations. (reset)
    for i in range(len(input)):
        qc.i(i) if input[i] < 0 else qc.x(i)

    return qc.to_gate()

# Performs a controlled X gate based on the output for n-bit AND gate for the input literals.
def and_gate_x(input, target):

    # Define quantum circuit.
    qc = QuantumCircuit(len(input)+1, name = 'And_gate_x')

    # Performing necessary negations.
    for i in range(len(input)):
        qc.x(i) if input[i] < 0 else qc.i(i)

    # Peform the required controlled X gate.
    gate = XGate().control(len(input))
    qc.append(gate, list(range(0, len(input)+1)))

    # Performing necessary negations. (reset)
    for i in range(len(input)):
        qc.x(i) if input[i] < 0 else qc.i(i)

    return qc.to_gate()

# Performs a controlled Z gate based on the output for n-bit OR gate for the input literals.
def or_gate_z(input, target):

    # Define quantum circuit.
    qc = QuantumCircuit(len(input)+1, name = 'Or_gate_z')

    # Perform a phase flip on target qubit.
    qc.z(len(input))

    # Performing necessary negations.
    for i in range(len(input)):
        qc.i(i) if input[i] < 0 else qc.x(i)

    # Peform the required controlled Z gate.
    gate = ZGate().control(len(input))
    qc.append(gate, list(range(0, len(input)+1)))

    # Performing necessary negations. (reset)
    for i in range(len(input)):
        qc.i(i) if input[i] < 0 else qc.x(i)

    return qc.to_gate()

# Performs a controlled Z gate based on the output for n-bit AND gate for the input literals.
def and_gate_z(input, target):

    # Define quantum circuit.
    qc = QuantumCircuit(len(input)+1, name = 'And_gate_z')

    # Performing necessary negations.
    for i in range(len(input)):
        qc.x(i) if input[i] < 0 else qc.i(i)

    # Peform the required controlled Z gate.
    gate = ZGate().control(len(input))
    qc.append(gate, list(range(0, len(input)+1)))

    # Performing necessary negations. (reset)
    for i in range(len(input)):
        qc.x(i) if input[i] < 0 else qc.i(i)

    return qc.to_gate()

# Performs a controlled adder gate based on the output for n-bit OR gate for the input literals.
def or_gate_adder(input, ancilla_qubits, number, invert):

    # Define quantum circuit.
    qc = QuantumCircuit(len(input)+len(ancilla_qubits), name = 'Or_gate_adder')

    # Performing necessary negations.
    for i in range(len(input)):
        qc.i(i) if input[i] < 0 else qc.x(i)

    # Peform the required controlled X gate.
    gate = adder(ancilla_qubits, number).control(len(input))
    qc.append(gate, list(range(0, len(input)+len(ancilla_qubits))))

    # Performing necessary negations. (reset)
    for i in range(len(input)):
        qc.i(i) if input[i] < 0 else qc.x(i)

    # Invert QFT if applicable.
    if invert: qc = qc.inverse()

    return qc.to_gate()

# Oracle gate for cnf.
def cnf_oracle(cnf):

    # Determine number of required qubits for oracle.
    req = req_qubits_oracle(cnf, False, None)

    # Define quantum circuit.
    qc = QuantumCircuit(req, name = 'Cnf_oracle')

    # Determine literals.
    literals = []
    for j in range(len(cnf)):
        literals = list(set(literals) | set([abs(i) for i in cnf[j]]))

    # Determine ancilla qubits and literals.
    ancillas = []
    for i in range(len(cnf)+1):
        ancillas.append(len(literals)+i)

    ancilla_literals = list(set([abs(i)+1 for i in ancillas]))

    # OR gate for each clause.
    for j in range(len(cnf)):

        # Get clause literals.
        literals_clause = list(set([abs(i) for i in cnf[j]]))

        # Get clause literal qubits.
        qubits_clause = list(set([literals_clause.index(i) for i in literals_clause]))

        # Get target qubit.
        qubits_clause.append(len(literals)+j)

        # Peforming the OR gate.
        gate = or_gate_x(cnf[j], len(literals)+j)
        qc.append(gate, qubits_clause)

    # AND gate for phase kickback.
    gate = and_gate_z(ancilla_literals[:-1], ancillas[-1])
    qc.append(gate, ancillas)

    # OR gate for each clause. (reset)
    for j in range(len(cnf)):

        # Get clause literals.
        literals_clause = list(set([abs(i) for i in cnf[j]]))

        # Get clause literal qubits.
        qubits_clause = list(set([literals_clause.index(i) for i in literals_clause]))

        # Get target qubit.
        qubits_clause.append(len(literals)+j)

        # Peforming the OR gate.
        gate = or_gate_x(cnf[j], len(literals)+j)
        qc.append(gate, qubits_clause)

    return qc.to_gate()

# Get number of required qubits for oracle.
def req_qubits_oracle(cnf, adder, number):

    # Determine literals.
    literals = []
    for j in range(len(cnf)):
        literals = list(set(literals) | set([abs(i) for i in cnf[j]]))

    if adder: return len(literals) + ceil(log(sum(number))/log(2)) + 1
    return len(literals) + len(cnf) + 1

# Grover gate for cnf.
def cnf_grover(cnf, iterations):

    # Determine number of required qubits for oracle.
    req = req_qubits_oracle(cnf, False, None)

    # Defining the required qubits for oracle.
    qubits = list(range(0, req_qubits_oracle(cnf, False, None)))

    # Define quantum circuit.
    qc = QuantumCircuit(req, name = 'Cnf_grover^'+str(iterations))

    # Determine target qubits for diffuser.
    target = []
    for j in range(len(cnf)):

        # Get clause literals.
        literals_clause = list(set([abs(i) for i in cnf[j]]))

        # Get target qubits.
        target = list(set([literals_clause.index(i) for i in literals_clause]))

    # Define oracle and diffuser gates.
    cnf_oracle_gate = cnf_oracle(cnf)
    diffuser_gate = diffuser(target)

    # Constructing the circuit for a specific number of iterations.
    for i in range(0, iterations):
        qc.append(cnf_oracle_gate, qubits)
        qc.append(diffuser_gate, target)

    return qc.to_gate()

# Quantum Fourier Transformation (QFT), and inverse QFT.
def qft(qubits, invert):

    # Define quantum circuit.
    qc = QuantumCircuit(len(qubits), name = 'QFT')

    # Peform quantum circuit for Quantum Fourier Transform.
    for i in range(len(qubits)):
        qc.h(len(qubits)-1-i)

        # Perform controlled phase gates.
        for j in range(i+1, len(qubits)):
            qc.cp(2*pi*(2 ** (i-j-1)), len(qubits)-1-j, len(qubits)-1-i)

    # Invert QFT if applicable.
    if invert: qc = qc.inverse()

    return qc.to_gate()

# Peforming addition with the qubit register in the Hadamard basis.
def adder(qubits, number):

    # Define quantum circuit.
    qc = QuantumCircuit(len(qubits), name = '+' + str(number))

    # Convert number to bit string.
    bit_array = sat.decimal2binary(qubits, number)

    # Peform quantum circuit for quantum adder.
    for i in range(len(qubits)):

        # Perform controlled phase gates.
        for j in range(i+1):
            if bit_array[j] == '1': qc.rz(2*pi*(2 ** (-i+j-1)), i)

    return qc.to_gate()

# Oracle gate for cnf.
def cnf_oracle_adder(cnf, number, compare_string):

    # Determine number of required qubits for oracle.
    req = req_qubits_oracle(cnf, True, number)

    # Define quantum circuit.
    qc = QuantumCircuit(req, name = 'Cnf_oracle')

    # Determine literals.
    literals = []
    for j in range(len(cnf)):
        literals = list(set(literals) | set([abs(i) for i in cnf[j]]))


    # Determine ancilla qubits and literals.
    ancillas = []
    for i in range(ceil(log(sum(number))/log(2))+1):
        ancillas.append(len(literals)+i)


    ancilla_literals = list(set([abs(i)+1 for i in ancillas]))
    ancilla_adder_clause = sat.get_ancilla_adder_clause(ancilla_literals, compare_string)


    # OR gate for each clause.

    gate = qft(ancillas[:-1], False)
    qc.append(gate, ancillas[:-1])

    for j in range(len(cnf)):

        # Get clause literals.
        literals_clause = list(set([abs(i) for i in cnf[j]]))

        # Get clause literal qubits.
        qubits_clause = list(set([literals_clause.index(i) for i in literals_clause]))

        # Get target qubit.
        qubits_clause = qubits_clause + ancillas[:-1]

        # Peforming the OR gate.
        gate = or_gate_adder(cnf[j], ancillas[:-1], number[j], False)
        qc.append(gate, qubits_clause)

    gate = qft(ancillas[:-1], True)
    qc.append(gate, ancillas[:-1])

    # Concatenate ancilla clause and phasekick qubit lists.
    relevant_ancillas = list(set([abs(i)-1 for i in ancilla_adder_clause]) | set([ancillas[-1]]))

    # AND gate for phase kickback.
    gate = and_gate_z(ancilla_adder_clause, ancillas[-1])
    qc.append(gate, relevant_ancillas)

    gate = qft(ancillas[:-1], True)
    qc.append(gate, ancillas[:-1])

    # OR gate for each clause. (reset)
    for j in range(len(cnf)):

        # Get clause literals.
        literals_clause = list(set([abs(i) for i in cnf[j]]))

        # Get clause literal qubits.
        qubits_clause = list(set([literals_clause.index(i) for i in literals_clause]))

        # Get target qubit.
        qubits_clause = qubits_clause + ancillas[:-1]

        # Peforming the OR gate.
        gate = or_gate_adder(cnf[j], ancillas[:-1], number[j], True)
        qc.append(gate, qubits_clause)

    gate = qft(ancillas[:-1], False)
    qc.append(gate, ancillas[:-1])

    return qc.to_gate()


# Grover gate for cnf.
def cnf_grover_adder(cnf, number, compare_string, iterations):

    # Determine number of required qubits for oracle.
    req = req_qubits_oracle(cnf, True, number)

    # Defining the required qubits for oracle.
    qubits = list(range(0, req_qubits_oracle(cnf, True, number)))

    # Define quantum circuit.
    qc = QuantumCircuit(req, name = 'Cnf_grover^'+str(iterations))

    # Determine target qubits for diffuser.
    target = []
    for j in range(len(cnf)):

        # Get clause literals.
        literals_clause = list(set([abs(i) for i in cnf[j]]))

        # Get target qubits.
        target = list(set([literals_clause.index(i) for i in literals_clause]))

    # Define oracle and diffuser gates.
    cnf_oracle_gate = cnf_oracle_adder(cnf, number, compare_string)
    diffuser_gate = diffuser(target)

    # Constructing the circuit for a specific number of iterations.
    for i in range(0, iterations):
        qc.append(cnf_oracle_gate, qubits)
        qc.append(diffuser_gate, target)

    return qc.to_gate()
