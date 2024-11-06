# https://github.com/MoizAhmedd/quantumtesting/blob/e0643e26628ef1cdc6d0c7309264bddd38004905/quantumhashing.py
from qiskit import  QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer, execute
from qiskit.quantum_info import Pauli, state_fidelity, basis_state, process_fidelity

def initialize_program():
    qubit = QuantumRegister(2)
    A = qubit[0]
    B = qubit[1]

    bit = ClassicalRegister(2)
    a = bit[0]
    b = bit[1]

    qc = QuantumCircuit(qubit,bit)

    return A, B, a, b, qc

def hash2bit(variable, hash, bit, qc):
    if hash == 'H':
        qc.h(variable)

    qc.measure(variable,bit)

def calculate_P(backend):
    """Calculates probability """

    P = {}
    program = {}
    for hashes in ['VV','VH','HV','HH']:
        A, B , a , b , program[hashes] = initialize_program()

        setup_variables(A, B, program[hashes])

        hash2bit(A, hashes[0], a, program[hashes])
        hash2bit(B,hashes[1],b,program[hashes])

    job = execute(list(program.values()), backend, shots=shots)

    for hashes in ['VV','VH','HV','HH']:
        stats = job.result().get_counts(program[hashes])

        P[hashes] = 0
        for string in stats.keys():
            a = string[-1]
            b = string[-2]

            if a!=b:
                P[hashes] += stats[string] / shots

    return P

#Registers
q0 = QuantumRegister(2,'q0')
c0 = ClassicalRegister(2,'c0')
q1 = QuantumRegister(2,'q1')
c1 = ClassicalRegister(2,'c1')
q_test = QuantumRegister(2,'q0')


#Circuits, made using registers
circ = QuantumCircuit(q0,q1)
circ.x(q0[1]) #Add x-gate
circ.x(g1[0])

#Registers can be added using add register method

#Concatenating circuits
qc = circ + meas

#Not sure about concatenation


