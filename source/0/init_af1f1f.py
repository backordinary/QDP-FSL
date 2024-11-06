# https://github.com/KerlinMichel/Quantum-Copy-Machine-Algorithm/blob/3ed16cb171d641fc72ff34318f927989b45dd079/qca/quantum_computing/__init__.py
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Instruction
from qiskit.quantum_info import Statevector

def build_random_integer_circuit(num_bits):
    q = QuantumRegister(num_bits, 'q')
    c = ClassicalRegister(num_bits, 'c')
    circuit = QuantumCircuit(q, c)

    # Apply hadamard gate to put all qubits in balanced superposition
    circuit.h(q)
    circuit.measure(q, c)

    return circuit

def get_circuit_statevector(circuit: QuantumCircuit):
    # Statevector.from_instruction doesn't work if the circuit has measure gates
    for i in range(len(circuit.data) - 1, -1, -1):
        if circuit.data[i][0].name == 'measure':
            circuit.data.pop(i)

    state = Statevector.from_instruction(circuit)
    return state

def choice(iter, sim=Aer.get_backend('aer_simulator')):
    num_states = len(iter)

    # iterable only has one possible state
    if num_states == 1:
        return iter[0]

    num_bits = int(math.log2(num_states - 1)) + 1
    choice_index = -1

    random_int_circuit = build_random_integer_circuit(num_bits=num_bits)

    while choice_index < 0 or choice_index >= len(iter):
        result = sim.run(random_int_circuit, shots=1).result()
        counts = result.results[0].data.counts

        # results are stored by counting which state the circuit collapses to
        hex_value = list(counts.keys())[0]

        # states are represented in hex
        choice_index = int(hex_value, 16)

    return iter[choice_index]

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

def to_binary(integer, num_bits, msb='first'):
    if msb not in ['first', 'last']:
        raise ValueError('Invalid most significant bit mode')
    binary = []

    num = integer
    for _ in range(num_bits):
        bit = int(num % 2)
        if msb == 'first':
            binary.insert(0, bit)
        elif msb == 'last':
            binary.append(bit)
        num = num / 2
    return binary

import numpy as np

# states is a list of states
# neighborhood is a list where each element is a tuple with a size of the number of dimension in output minus 1
# output of size d1 x ... x len(states)
# output_mask of size d1 x ...
def build_quantum_collapse_algorithm_circuit(states, neighborhood, output, output_mask=None):
    num_states = len(states)

    if num_states == 0:
        raise ValueError('states is empty')

    qubits_per_output_part = math.ceil(math.log2(num_states))

    # output_mask masks parts of the output to ignore
    num_masked_output = 0 if output_mask is None else np.count_nonzero(output_mask == 0)
    num_output_parts = (output.size // num_states) - num_masked_output

    output_qrs = [QuantumRegister(qubits_per_output_part, f'output_{i} ') for i in range(num_output_parts)]
    output_part_indexes = np.ndindex(output.shape[:-1])
    output_part_qr_mapping = []

    # populate output_part_qr_mapping where the index points to a quantum register in output_qrs and the value is the output index
    for o_i in output_part_indexes:
        # skip masked output parts
        if output_mask is not None and output_mask[o_i] == 0:
            continue

        output_part_qr_mapping.append(o_i)

    neighborhood_constraint_qrs = []
    neighborhood_constraint_output_mapping = []

    for o_i, o in enumerate(output_part_qr_mapping):
        for n_i, n in enumerate(neighborhood):
            neighbor_index = np.array(o) + np.array(n)

            if np.any(neighbor_index < 0) or np.any(neighbor_index > np.array(output.shape[:-1]) - 1):
                    continue

            neighborhood_constraint_qr = QuantumRegister(1, f'neighborhood_constraint_{len(neighborhood_constraint_qrs)} ')

            neighborhood_constraint_qrs.append(neighborhood_constraint_qr)

            # (output part 1 quantum register, output part 2 quantum register, neighborhood index)
            neighborhood_constraint_output_mapping.append((output_qrs[o_i], output_qrs[output_part_qr_mapping.index(tuple(neighbor_index))], n_i))

    feasible_qr = QuantumRegister(1, 'feasible')

    circuit = QuantumCircuit(*output_qrs, *neighborhood_constraint_qrs, feasible_qr)

    return circuit, (output_qrs, output_part_qr_mapping), (neighborhood_constraint_qrs, neighborhood_constraint_output_mapping), feasible_qr

def set_quantum_collapse_algorithm_superposition(circuit, output_qrs):
    for output_qr in output_qrs:
        circuit.h(output_qr)

def build_neighbor_constraint_circuit(quantum_collapse_algorithm_circuit, output, neighborhood_constraint_circuit, to_gate=False, clone_circuit=True):
    (output_qr_1, state_1), (output_qr_2, state_2) = output
    if clone_circuit:
        quantum_collapse_algorithm_circuit = quantum_collapse_algorithm_circuit.copy()

        # remove all instructions
        quantum_collapse_algorithm_circuit.data = [d for d in quantum_collapse_algorithm_circuit.data if not isinstance(d[0], Instruction)]

        # remove feasiable circuit
        quantum_collapse_algorithm_circuit.qubits.pop()

    state_num_qubits = len(output_qr_1._bits)

    s1 = to_binary(state_1, state_num_qubits, msb='last')
    s2 = to_binary(state_2, state_num_qubits, msb='last')

    quantum_collapse_algorithm_circuit.x(list(b for i, b in enumerate(output_qr_1._bits) if s1[i] == 0) + list(b for i, b in enumerate(output_qr_2._bits) if s2[i] == 0))
    quantum_collapse_algorithm_circuit.mct(output_qr_1._bits + output_qr_2._bits, neighborhood_constraint_circuit)
    quantum_collapse_algorithm_circuit.x(list(b for i, b in enumerate(output_qr_1._bits) if s1[i] == 0) + list(b for i, b in enumerate(output_qr_2._bits) if s2[i] == 0))

    if to_gate:
        gate = quantum_collapse_algorithm_circuit.to_gate()
        gate.name = f'C_n(s1: {state_1}, s2: {state_2}, C_n_i: {neighborhood_constraint_circuit.name[-2:-1]})'
        return gate

    return quantum_collapse_algorithm_circuit

def build_feasible_check_circuit(quantum_collapse_algorithm_circuit, neighborhood_constraint_qrs, to_gate=False, clone_circuit=True):
    feasible_circuit = quantum_collapse_algorithm_circuit.qubits[-1]

    if clone_circuit:
        quantum_collapse_algorithm_circuit = quantum_collapse_algorithm_circuit.copy()

        # remove all instructions
        quantum_collapse_algorithm_circuit.data = [d for d in quantum_collapse_algorithm_circuit.data if not isinstance(d[0], Instruction)]

        for _ in range(len(quantum_collapse_algorithm_circuit.qubits) - (len(neighborhood_constraint_qrs) + 1)):
            quantum_collapse_algorithm_circuit.qubits.pop(0)

    quantum_collapse_algorithm_circuit.mct(neighborhood_constraint_qrs, feasible_circuit)

    if to_gate:
        gate = quantum_collapse_algorithm_circuit.to_gate()
        gate.name = f'feasibility_check'
        return gate

    return quantum_collapse_algorithm_circuit

def build_quantum_collapse_neighborhood_constraint(quantum_collapse_algorithm_circuit, C_n, neighborhood_constraint_qrs, neighborhood_constraint_output_mapping, to_gate=False, clone_circuit=False):
    if clone_circuit:
        quantum_collapse_algorithm_circuit = quantum_collapse_algorithm_circuit.copy()

        # remove all instructions
        quantum_collapse_algorithm_circuit.data = [d for d in quantum_collapse_algorithm_circuit.data if not isinstance(d[0], Instruction)]

    for nc_qr_i, (output_qr_1, output_qr_2, neighborhood_i) in enumerate(neighborhood_constraint_output_mapping):
        for s1 in range(C_n.shape[0]):
            for n_i in range(C_n.shape[1]):
                for s2 in range(C_n.shape[2]):
                    if n_i == neighborhood_i and C_n[s1, n_i, s2] == 1:
                        build_neighbor_constraint_circuit(quantum_collapse_algorithm_circuit, ((output_qr_1, s1), (output_qr_2, s2)), neighborhood_constraint_qrs[nc_qr_i], clone_circuit=False)

    if to_gate:
        gate = quantum_collapse_algorithm_circuit.to_gate()
        gate.name = f'neighborhood_constraint'
        return gate

    return quantum_collapse_algorithm_circuit

def build_quantum_collapse_oracle(quantum_collapse_algorithm_circuit, C_n, neighborhood_constraint_output_mapping, neighborhood_constraint_qrs, to_gate=False):
    quantum_collapse_algorithm_circuit = quantum_collapse_algorithm_circuit.copy()

    # remove all instructions
    quantum_collapse_algorithm_circuit.data = [d for d in quantum_collapse_algorithm_circuit.data if not isinstance(d[0], Instruction)]

    build_quantum_collapse_neighborhood_constraint(quantum_collapse_algorithm_circuit, C_n, neighborhood_constraint_qrs, neighborhood_constraint_output_mapping, clone_circuit=False)
    build_feasible_check_circuit(quantum_collapse_algorithm_circuit, neighborhood_constraint_qrs, clone_circuit=False)
    # uncompute
    build_quantum_collapse_neighborhood_constraint(quantum_collapse_algorithm_circuit, C_n, neighborhood_constraint_qrs, neighborhood_constraint_output_mapping, clone_circuit=False)

    if to_gate:
        gate = quantum_collapse_algorithm_circuit.to_gate()
        gate.name = f'QC Oracle'
        return gate

    return quantum_collapse_algorithm_circuit

def flatten(t):
        return [item for sublist in t for item in sublist]

def build_quantum_collapse_diffuser(quantum_collapse_algorithm_circuit, output_qrs, neighborhood_constraint_qrs, to_gate=False):
    quantum_collapse_algorithm_circuit = quantum_collapse_algorithm_circuit.copy()

    # remove all instructions
    quantum_collapse_algorithm_circuit.data = [d for d in quantum_collapse_algorithm_circuit.data if not isinstance(d[0], Instruction)]

    # remove neighborhood contraint circuit and feasible circuit
    for _ in range(len(neighborhood_constraint_qrs) + 1):
            quantum_collapse_algorithm_circuit.qubits.pop()

    quantum_collapse_algorithm_circuit.append(diffuser(len(output_qrs) * output_qrs[0].size), flatten([qubits for qubits in output_qrs]))

    if to_gate:
        gate = quantum_collapse_algorithm_circuit.to_gate()
        gate.name = f'U_s'
        return gate

    return quantum_collapse_algorithm_circuit

def build_quantum_collapse_grover_search_circuit(quantum_collapse_algorithm_circuit, C_n, output_qrs, neighborhood_constraint_qrs, neighborhood_constraint_output_mapping, num_iterations=1, to_gate=False):
    quantum_collapse_algorithm_circuit = quantum_collapse_algorithm_circuit.copy()

    # remove all instructions
    quantum_collapse_algorithm_circuit.data = [d for d in quantum_collapse_algorithm_circuit.data if not isinstance(d[0], Instruction)]

    for _ in range(num_iterations):
        oracle = build_quantum_collapse_oracle(quantum_collapse_algorithm_circuit, C_n, neighborhood_constraint_output_mapping, neighborhood_constraint_qrs, to_gate=True)

        diffuser_gate = build_quantum_collapse_diffuser(quantum_collapse_algorithm_circuit, output_qrs, neighborhood_constraint_qrs, to_gate=True)

        quantum_collapse_algorithm_circuit.append(oracle, quantum_collapse_algorithm_circuit.qubits)

        quantum_collapse_algorithm_circuit.append(diffuser_gate, flatten([qubits for qubits in output_qrs]))

    if to_gate:
        gate = quantum_collapse_algorithm_circuit.to_gate()
        gate.name = f'QC Grover'
        return gate

    return quantum_collapse_algorithm_circuit