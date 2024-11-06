# https://github.com/sabinadragoi/Grover/blob/f847ea049b2937747dc67a73794af5e0ba3dee5d/global_gate.py
import numpy as np
import time
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# def CZ(qbit1,qbit2):
#     # qbit1 = [c_0,c_1]
#     # qbit2 = [d_0,d_1]
#     return [qbit1[0]*qbit2[0],qbit1[0]*qbit2[1],-qbit1[1]*qbit2[0],-qbit1[1]*qbit2[1]]
#
# def CNOT(qbit1,qbit2):
#     # qbit1 = [c_0,c_1]
#     # qbit2 = [d_0,d_1]
#     return [qbit1[0]*qbit2[0],qbit1[0]*qbit2[1],qbit1[1]*qbit2[1],qbit1[1]*qbit2[0]]

def total_CNOT(init_state, i1,i2, normalized):
    # init state is a list of 2**n entries, each corresponding to the coefficient of that basis state
    # n = total number of qubits considered
    n = int(np.log2(len(init_state)))
    # i1 = index of control qubit, i2 = index of target qubit
    new_state = init_state.copy()
    for i in range(len(init_state)):
        basis_state = "{0:b}".format(i).zfill(n)
        basis_state_entry = init_state[i]
        if basis_state[i1] == '1':
            basis_state_list = list(basis_state)
            basis_state_list[i2] = str(1 - int(basis_state[i2]))
            new_basis_state= ''.join([str(item) for item in basis_state_list])
            new_basis_state_index = int(new_basis_state,2)
            new_state[new_basis_state_index] += basis_state_entry
            new_state[i] = new_state[i] - basis_state_entry

    if normalized == False:
        return new_state



def total_CZ(init_state, i1):
    # init state is a list of 2**n entries, each corresponding to the coefficient of that basis state
    # n = total number of qubits considered
    n = int(np.log2(len(init_state)))
    # i1 = index of control qubit, i2 = index of target qubit
    new_state = init_state.copy()
    for i in range(len(init_state)):
        basis_state = "{0:b}".format(i).zfill(n)
        basis_state_entry = init_state[i]

        if basis_state[i1] == '1':
            new_state[i] = - basis_state_entry

    return new_state


def add_two_qbits(state1, state2):
    # qbit = [c_0,c_1]
    n = int(np.log2(len(state1)))
    m = int(np.log2(len(state2)))
    total_state = np.zeros(2**(n+m)) # 2**(n+m) = len(state1)+len(state1)
    for i in range(len(state1)):
        basis_state_1 = "{0:b}".format(i).zfill(n)
        basis_state_1_entry = state1[i]
        for j in range((len(state2))):
            basis_state_2 = "{0:b}".format(j).zfill(m)
            basis_state_2_entry = state2[j]

            basis_state_total = basis_state_1 + basis_state_2
            basis_state_total_index = int(basis_state_total,2)
            total_state[basis_state_total_index] = basis_state_1_entry*basis_state_2_entry
    return total_state


# add_qbits(*argv)
#    A wrapper around `add_two_qbits`.
#    Takes an arbitrary number of arguments, and performs
#    addition on all of them.
#
#    Should not be called with fewer than 2 arguments.
#
#    Should not be called with an extremely large number
#    of arguments, as the recursion depth limit might be
#    exceeded.

def add_qbits(*argv):
    assert len(argv) >= 2, ("add_qbits cannot be called "
                            "with fewer than 2 arguments")

    if len(argv) == 2:
        return add_two_qbits(*argv)

    return add_qbits(add_two_qbits(argv[0], argv[1]), *argv[2:])


# c_0 = np.sqrt(1/2)
# c_1 = np.sqrt(1/2)
# d_0 = np.sqrt(1/3)
# d_1 = np.sqrt(2/3)
# q_1 = [c_0,c_1]
# q_2 = [d_0,d_1]
# a1a2 = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
# q1a1a2 = add_qbits(q_1,a1a2)
# q1a1a2q2 = add_qbits(q1a1a2,q_2)
# inter_state = total_CNOT(q1a1a2q2,0,1,False)
# inter_state = total_CZ(inter_state,1)
# inter_state = total_CNOT(inter_state,)
# inter_state = total_CNOT(inter_state,3,2,False)

if __name__ == "__main__":
    anc = QuantumRegister(2, 'a')
    qr = QuantumRegister(2, 'q')
    # cr = ClassicalRegister(1, 'c')
    cr = QuantumRegister(1, 'c')
    circuit = QuantumCircuit(anc,qr, cr)

    circuit.cx(qr[0],anc[0])
    circuit.cx(qr[1],anc[1])
    circuit.cz(qr[0],anc[0],label = 'CZ')

    circuit.cx(qr[0],cr[0])
    circuit.cx(qr[1],cr[0])
    circuit.fredkin(cr[0],anc[0],anc[1])
    circuit.cz(qr[0],anc[0],label = 'CZ')

    circuit.barrier()

    circuit.cx(qr[0],anc[0])
    circuit.cx(qr[1],anc[1])

    circuit.draw(output='mpl')
    plt.show()
