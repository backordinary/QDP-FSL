# https://github.com/weiwenjiang/qc_mnist/blob/c337d72c207fad7b998528e77795cbf0bf1f8fba/Revision/MNIST_Demo_2/qiskit_library.py
import numpy as np
from random import randrange
import qiskit as qk
from qiskit import Aer
from qiskit import execute
import math
import sys
import random
import time
from tqdm import tqdm


# ========================
# Preformatting Start
# ========================

# Generate a list of array size binaries
def generateStates(size):
    state = []
    sizeRoot = int(math.log(size, 2))
    for i in range(size):
        state.append(bin(i))
    for i in range(size):
        state[i] = state[i][1:]
        state[i] = state[i].replace('b', '')
        if len(state[i]) < sizeRoot: state[i] = (add0(sizeRoot - (len(state[i])))) + state[i]
    return state


def markStates(state, matrix):
    size = len(matrix)
    for i in range(size):
        if matrix[i] == 1:
            state[i] = '-' + state[i]
        else:
            state[i] = '+' + state[i]
    return state


def add0(n):
    if n == 0:
        return ('')
    else:
        return ('0' + (add0(n - 1)))


def allPositive(state, size):
    for i in range(size):
        state[i] = '+' + state[i]
    return state


def howMany1(string):
    count = 0
    for i in range(len(string)):
        if string[i] == '1': count = count + 1
    return count


def singFlip(string):
    if string[0] == '-':
        return ('+' + string[1:])
    else:
        return ('-' + string[1:])


def whereIs1(string):
    locations = ''
    string = string[1:]
    for i in range(len(string)):
        if string[i] == '1': locations = locations + str(i)
    return locations


def flipSingOn(state, location, maxIndex):
    for i in range(maxIndex):
        if state[i][location] == '1':
            state[i] = singFlip(state[i])
    return state


# singFlip em todos os elementos com '1' em duas posições especificas
def flipSingOn2(state, qubits, maxIndex):
    q0 = int(qubits[0]) + 1
    q1 = int(qubits[1]) + 1
    for i in range(maxIndex):
        if state[i][q0] == '1' and state[i][q1] == '1':
            state[i] = singFlip(state[i])
    return state


# singFlip em todos os elementos com '1' em três posições especificas
def flipSingOn3(state, qubits, maxIndex):
    q0 = int(qubits[0]) + 1
    q1 = int(qubits[1]) + 1
    q2 = int(qubits[2]) + 1
    for i in range(maxIndex):
        if state[i][q0] == '1' and state[i][q1] == '1' and state[i][q2] == '1':
            state[i] = singFlip(state[i])
    return state


# singFlip em todos os elementos com '1' em quatro posições especificas
def flipSingOn4(state, qubits, maxIndex):
    q0 = int(qubits[0]) + 1
    q1 = int(qubits[1]) + 1
    q2 = int(qubits[2]) + 1
    q3 = int(qubits[3]) + 1
    for i in range(maxIndex):
        if state[i][q0] == '1' and state[i][q1] == '1' and state[i][q2] == '1' and state[i][q3] == '1':
            state[i] = singFlip(state[i])
    return state

# singFlip em todos os elementos com '1' em quatro posições especificas
def flipSingOn5(state, qubits, maxIndex):
    q0 = int(qubits[0]) + 1
    q1 = int(qubits[1]) + 1
    q2 = int(qubits[2]) + 1
    q3 = int(qubits[3]) + 1
    q4 = int(qubits[4]) + 1
    for i in range(maxIndex):
        if state[i][q0] == '1' and state[i][q1] == '1' and state[i][q2] == '1' and state[i][q3] == '1' and state[i][q4] == '1':
            state[i] = singFlip(state[i])
    return state


def flipSingOn6(state, qubits, maxIndex):
    q0 = int(qubits[0]) + 1
    q1 = int(qubits[1]) + 1
    q2 = int(qubits[2]) + 1
    q3 = int(qubits[3]) + 1
    q4 = int(qubits[4]) + 1
    q5 = int(qubits[5]) + 1
    for i in range(maxIndex):
        if state[i][q0] == '1' and state[i][q1] == '1' and state[i][q2] == '1' and state[i][q3] == '1' and state[i][q4] == '1' and state[i][q5] == '1':
            state[i] = singFlip(state[i])
    return state



# Generates circuit for 1 | 1⟩ bits
def runTo1(circuit, statusVector, n, q, maxIndex):
    # print("*"*40,"in run to 1","*"*40)
    # print_status(statusVector,"status_in_run_1-1")
    theQubits = whereIs1(statusVector[n])
    # print(n,statusVector[n],theQubits,type(theQubits))
    circuit.z(q[int(theQubits[0])])  # apply z to qubit
    # print(circuit)
    # multiplies by -1 on the status vector
    statusVector = flipSingOn(statusVector, int(theQubits[0]) + 1, maxIndex)
    # print_status(statusVector,"status_in_run_1-2")

    return statusVector


# Gera circuito para bits com 2 |1⟩
def runTo2(circuit, statusVector, n, q, maxIndex):
    # print("*"*40,"runTo2","*"*40)
    theQubits = whereIs1(statusVector[n])
    # print_status(statusVector,"status_run2")
    # print(n,statusVector[n],theQubits)
    q0 = int(theQubits[0])
    q1 = int(theQubits[1])
    circuit.cz(q[q0], q[q1])
    # print(circuit)
    statusVector = flipSingOn2(statusVector, theQubits, maxIndex)
    # print_status(statusVector,"status_run2")
    return statusVector


# Gera circuito para bits com 3 |1⟩
def runTo3(circuit, statusVector, n, q, aux, maxIndex):
    theQubits = whereIs1(statusVector[n])
    q0 = int(theQubits[0])
    q1 = int(theQubits[1])
    q2 = int(theQubits[2])

    circuit = ccz(circuit, q[q0], q[q1], q[q2], aux[0])
    statusVector = flipSingOn3(statusVector, theQubits, maxIndex)
    return statusVector


# Gera circuito para bits com 4 |1⟩
def runTo4(circuit, statusVector, n, q, aux, maxIndex):
    theQubits = whereIs1(statusVector[n])
    q0 = int(theQubits[0])
    q1 = int(theQubits[1])
    q2 = int(theQubits[2])
    q3 = int(theQubits[3])

    circuit = cccz(circuit, q[q0], q[q1], q[q2], q[q3], aux[0], aux[1])
    statusVector = flipSingOn4(statusVector, theQubits, maxIndex)
    return statusVector

# Gera circuito para bits com 4 |1⟩
def runTo5(circuit, statusVector, n, q, aux, maxIndex):
    theQubits = whereIs1(statusVector[n])
    q0 = int(theQubits[0])
    q1 = int(theQubits[1])
    q2 = int(theQubits[2])
    q3 = int(theQubits[3])
    q4 = int(theQubits[4])

    circuit = ccccz(circuit, q[q0], q[q1], q[q2], q[q3], q[q4], aux[0], aux[1], aux[2])
    statusVector = flipSingOn5(statusVector, theQubits, maxIndex)
    return statusVector


def runTo6(circuit, statusVector, n, q, aux, maxIndex):
    theQubits = whereIs1(statusVector[n])
    q0 = int(theQubits[0])
    q1 = int(theQubits[1])
    q2 = int(theQubits[2])
    q3 = int(theQubits[3])
    q4 = int(theQubits[4])
    q5 = int(theQubits[5])

    circuit = cccccz(circuit, q[q0], q[q1], q[q2], q[q3], q[q4], q[q5], aux[0], aux[1], aux[2], aux[3])
    statusVector = flipSingOn6(statusVector, theQubits, maxIndex)
    return statusVector

# =======================
# MULTI-CONTROLLED GATES
# =======================

def ccz(circ, q1, q2, q3, aux1):
    # Apply Z-gate to a state controlled by 3 qubits
    circ.ccx(q1, q2, aux1)
    circ.cz(aux1, q3)
    # cleaning the aux bit
    circ.ccx(q1, q2, aux1)
    return circ


def cccz(circ, q1, q2, q3, q4, aux1, aux2):
    # Apply Z-gate to a state controlled by 4 qubits
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, aux1, aux2)
    circ.cz(aux2, q4)
    # cleaning the aux bits
    circ.ccx(q3, aux1, aux2)
    circ.ccx(q1, q2, aux1)
    return circ

def ccccz(circ, q1, q2, q3, q4, q5, aux1, aux2, aux3):
    # Apply Z-gate to a state controlled by 4 qubits
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, aux1, aux2)
    circ.ccx(q4, aux2, aux3)
    circ.cz(aux3, q5)
    # cleaning the aux bits
    circ.ccx(q4, aux2, aux3)
    circ.ccx(q3, aux1, aux2)
    circ.ccx(q1, q2, aux1)
    return circ

def cccccz(circ, q1, q2, q3, q4, q5, q6, aux1, aux2, aux3, aux4):
    # Apply Z-gate to a state controlled by 4 qubits
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, aux1, aux2)
    circ.ccx(q4, aux2, aux3)
    circ.ccx(q5, aux3, aux4)
    circ.cz(aux4, q6)
    # cleaning the aux bits
    circ.ccx(q5, aux3, aux4)
    circ.ccx(q4, aux2, aux3)
    circ.ccx(q3, aux1, aux2)
    circ.ccx(q1, q2, aux1)
    return circ

def cccx(circ, q1, q2, q3, q4, aux1):
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, aux1, q4)
    # cleaning the aux bits
    circ.ccx(q1, q2, aux1)
    return circ


def ccccx(circ, q1, q2, q3, q4, q5, aux1, aux2):
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, q4, aux2)
    circ.ccx(aux2, aux1, q5)
    # cleaning the aux bits
    circ.ccx(q3, q4, aux2)
    circ.ccx(q1, q2, aux1)
    return circ


def ccccccx(circ, q, q_out, aux):
    circ.ccx(q[0], q[1], aux[0])
    circ.ccx(q[2], aux[0], aux[1])
    circ.ccx(q[3], aux[1], aux[2])
    circ.ccx(q[4], aux[2], aux[3])
    circ.ccx(q[5], aux[3], q_out)
    # cleaning the aux bits
    circ.ccx(q[4], aux[2], aux[3])
    circ.ccx(q[3], aux[1], aux[2])
    circ.ccx(q[2], aux[0], aux[1])
    circ.ccx(q[0], q[1], aux[0])
    return circ


def print_status(vector, name="status_vector"):
    print("=" * 40, name, "=" * 40)
    for v in vector:
        print(v, end=" ")
    print()
    print("-" * 100)


# Gera circuito para operador U, baseado em sua matriz representativa
def generateU(circuit, matrix, q, aux):

    maxIndex = len(matrix)
    # print(maxIndex)
    goalVector = generateStates(maxIndex)
    goalVector = markStates(goalVector, matrix)
    # print_status(goalVector,"goal_vector")
    statusVector = generateStates(maxIndex)
    statusVector = allPositive(statusVector, maxIndex)

    # Checking if goalVector[0] is -1
    if goalVector[0][0] == '-':
        for i in range(maxIndex):
            goalVector[i] = singFlip(goalVector[i])

    # print_status(goalVector, "goal_vector")
    # Weiwen: Add Z gate
    for n in range(maxIndex):
        if (statusVector[n] != goalVector[n]) and (howMany1(goalVector[n]) == 1):
            # print(n,end=" ")
            statusVector = runTo1(circuit, statusVector, n, q, maxIndex)
    # print()
    # print_status(statusVector)

    # Weiwen: Add cZ on two qbits whose sign is not consistent with the goal
    for n in range(maxIndex):
        if (statusVector[n] != goalVector[n]) and (howMany1(goalVector[n]) == 2):
            statusVector = runTo2(circuit, statusVector, n, q, maxIndex)

    # print(circuit)
    # print_status(statusVector)

    for n in range(maxIndex):
        if (statusVector[n] != goalVector[n]) and (howMany1(goalVector[n]) == 3):
            statusVector = runTo3(circuit, statusVector, n, q, aux, maxIndex)

    # print(circuit)
    # print_status(statusVector)

    for n in range(maxIndex):
        if (statusVector[n] != goalVector[n]) and (howMany1(goalVector[n]) == 4):
            statusVector = runTo4(circuit, statusVector, n, q, aux, maxIndex)



    # IMPORTANT: This function limit sthe number of qubits: Now Support 6 qubite 2^6 = 64 pixels
    for n in range(maxIndex):
        if (statusVector[n] != goalVector[n]) and (howMany1(goalVector[n]) == 5):
            statusVector = runTo5(circuit, statusVector, n, q, aux, maxIndex)

    for n in range(maxIndex):
        if (statusVector[n] != goalVector[n]) and (howMany1(goalVector[n]) == 6):
            statusVector = runTo6(circuit, statusVector, n, q, aux, maxIndex)

    # print(circuit)
    # print_status(statusVector)

    # circuit.barrier()


def encoder2_1(circuit, q, o):
    circuit.cx(q[0], q[1])
    circuit.h(o)

    circuit.cz(q[1], o)

    circuit.barrier()
    return o

def encoder4_2(circuit, q, aux):
    circuit.cx(q[0], q[1])
    circuit.cx(q[0], q[2])
    circuit.cx(q[0], q[3])

    for i in [4, 5]:
        circuit.h(q[i])

    circuit.cz(q[1], q[5])
    ccz(circuit, q[1], q[5], q[4], aux[0])

    circuit.cz(q[2], q[4])
    ccz(circuit, q[2], q[4], q[5], aux[0])

    ccz(circuit, q[3], q[4], q[5], aux[0])

    circuit.barrier()
    return q[4:6]


def encoder8_3(circuit, q, aux):
    circuit.cx(q[0], q[1])
    circuit.cx(q[0], q[2])
    circuit.cx(q[0], q[3])
    circuit.cx(q[0], q[4])
    circuit.cx(q[0], q[5])
    circuit.cx(q[0], q[6])
    circuit.cx(q[0], q[7])

    for i in [8, 9, 10]:
        circuit.h(q[i])

    circuit.cz(q[1], q[10])
    ccz(circuit, q[1], q[10], q[9], aux[0])
    ccz(circuit, q[1], q[10], q[8], aux[0])
    cccz(circuit, q[1], q[10], q[9], q[8], aux[0], aux[1])

    circuit.cz(q[2], q[9])
    ccz(circuit, q[2], q[9], q[8], aux[0])
    ccz(circuit, q[2], q[9], q[10], aux[0])
    cccz(circuit, q[2], q[9], q[8], q[10], aux[0], aux[1])

    ccz(circuit, q[3], q[9], q[10], aux[0])
    cccz(circuit, q[3], q[9], q[10], q[8], aux[0], aux[1])

    circuit.cz(q[4], q[8])
    ccz(circuit, q[4], q[8], q[10], aux[0])
    ccz(circuit, q[4], q[8], q[9], aux[0])
    cccz(circuit, q[4], q[8], q[9], q[10], aux[0], aux[1])

    ccz(circuit, q[5], q[8], q[10], aux[0])
    cccz(circuit, q[5], q[8], q[9], q[10], aux[0], aux[1])

    ccz(circuit, q[6], q[8], q[9], aux[0])
    cccz(circuit, q[6], q[8], q[9], q[10], aux[0], aux[1])

    cccz(circuit, q[7], q[8], q[9], q[10], aux[0], aux[1])

    circuit.barrier()
    return q[8:11]




def encoder16_4(circuit, q, o, aux):
    for i in range(1, 16):
        circuit.cx(q[0], q[i])
    circuit.h(o[0])
    circuit.h(o[1])
    circuit.h(o[2])
    circuit.h(o[3])

    circuit.cz(q[1], o[3])
    ccz(circuit, q[1], o[3], o[2], aux[0])
    ccz(circuit, q[1], o[3], o[1], aux[0])
    ccz(circuit, q[1], o[3], o[0], aux[0])
    cccz(circuit, q[1], o[3], o[2], o[1], aux[0], aux[1])
    cccz(circuit, q[1], o[3], o[2], o[0], aux[0], aux[1])
    cccz(circuit, q[1], o[3], o[1], o[0], aux[0], aux[1])
    ccccz(circuit, q[1], o[3], o[2], o[1], o[0], aux[0], aux[1], aux[2])

    circuit.cz(q[2], o[2])
    ccz(circuit, q[2], o[2], o[3], aux[0])
    ccz(circuit, q[2], o[2], o[1], aux[0])
    ccz(circuit, q[2], o[2], o[0], aux[0])
    cccz(circuit, q[2], o[2], o[3], o[1], aux[0], aux[1])
    cccz(circuit, q[2], o[2], o[3], o[0], aux[0], aux[1])
    cccz(circuit, q[2], o[2], o[1], o[0], aux[0], aux[1])
    ccccz(circuit, q[2], o[2], o[3], o[1], o[0], aux[0], aux[1], aux[2])

    circuit.cz(q[4], o[1])
    ccz(circuit, q[4], o[1], o[3], aux[0])
    ccz(circuit, q[4], o[1], o[2], aux[0])
    ccz(circuit, q[4], o[1], o[0], aux[0])
    cccz(circuit, q[4], o[1], o[3], o[2], aux[0], aux[1])
    cccz(circuit, q[4], o[1], o[3], o[0], aux[0], aux[1])
    cccz(circuit, q[4], o[1], o[2], o[0], aux[0], aux[1])
    ccccz(circuit, q[4], o[1], o[3], o[2], o[0], aux[0], aux[1], aux[2])

    circuit.cz(q[8], o[0])
    ccz(circuit, q[8], o[0], o[3], aux[0])
    ccz(circuit, q[8], o[0], o[2], aux[0])
    ccz(circuit, q[8], o[0], o[1], aux[0])
    cccz(circuit, q[8], o[0], o[3], o[2], aux[0], aux[1])
    cccz(circuit, q[8], o[0], o[3], o[1], aux[0], aux[1])
    cccz(circuit, q[8], o[0], o[2], o[1], aux[0], aux[1])
    ccccz(circuit, q[8], o[0], o[3], o[2], o[1], aux[0], aux[1], aux[2])


    ccz(circuit, q[3], o[2], o[3], aux[0])
    cccz(circuit, q[3], o[2], o[3], o[0], aux[0], aux[1])
    cccz(circuit, q[3], o[2], o[3], o[1], aux[0], aux[1])
    ccccz(circuit, q[3], o[2], o[3], o[0], o[1], aux[0], aux[1], aux[2])

    ccz(circuit, q[5], o[1], o[3], aux[0])
    cccz(circuit, q[5], o[1], o[3], o[0], aux[0], aux[1])
    cccz(circuit, q[5], o[1], o[3], o[2], aux[0], aux[1])
    ccccz(circuit, q[5], o[1], o[3], o[0], o[2], aux[0], aux[1], aux[2])

    ccz(circuit, q[6], o[1], o[2], aux[0])
    cccz(circuit, q[6], o[1], o[2], o[0], aux[0], aux[1])
    cccz(circuit, q[6], o[1], o[2], o[3], aux[0], aux[1])
    ccccz(circuit, q[6], o[1], o[2], o[0], o[3], aux[0], aux[1], aux[2])

    ccz(circuit, q[9], o[0], o[3], aux[0])
    cccz(circuit, q[9], o[0], o[3], o[1], aux[0], aux[1])
    cccz(circuit, q[9], o[0], o[3], o[2], aux[0], aux[1])
    ccccz(circuit, q[9], o[0], o[3], o[1], o[2], aux[0], aux[1], aux[2])

    ccz(circuit, q[10], o[0], o[2], aux[0])
    cccz(circuit, q[10], o[0], o[2], o[1], aux[0], aux[1])
    cccz(circuit, q[10], o[0], o[2], o[3], aux[0], aux[1])
    ccccz(circuit, q[10], o[0], o[2], o[1], o[3], aux[0], aux[1], aux[2])

    ccz(circuit, q[12], o[0], o[1], aux[0])
    cccz(circuit, q[12], o[0], o[1], o[2], aux[0], aux[1])
    cccz(circuit, q[12], o[0], o[1], o[3], aux[0], aux[1])
    ccccz(circuit, q[12], o[0], o[1], o[2], o[3], aux[0], aux[1], aux[2])


    cccz(circuit, q[7], o[1], o[2], o[3], aux[0], aux[1])
    ccccz(circuit, q[7], o[1], o[2], o[3], o[0], aux[0], aux[1], aux[2])

    cccz(circuit, q[11], o[0], o[2], o[3], aux[0], aux[1])
    ccccz(circuit, q[11], o[0], o[2], o[3], o[1], aux[0], aux[1], aux[2])

    cccz(circuit, q[13], o[0], o[1], o[3], aux[0], aux[1])
    ccccz(circuit, q[13], o[0], o[1], o[3], o[2], aux[0], aux[1], aux[2])

    cccz(circuit, q[14], o[0], o[1], o[2], aux[0], aux[1])
    ccccz(circuit, q[14], o[0], o[1], o[2], o[3], aux[0], aux[1], aux[2])

    ccccz(circuit, q[15], o[0], o[1], o[2], o[3], aux[0], aux[1], aux[2])

    circuit.barrier()
    return o




from qiskit.tools.monitor import job_monitor


def simulate(circuit, shots, iter, printable=True):
    if printable:
        print(circuit)

    count_set = []

    for it in range(iter):
        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute(circuit, backend, shots=shots)
        # job_monitor(job_sim)
        result_sim = job_sim.result()
        counts = result_sim.get_counts()
        count_set.append(counts)

    return count_set


def single_priceptro(circuit, q, input, weights, aux=[]):
    circuit.h(q[0])
    circuit.h(q[1])
    circuit.h(q[2])
    circuit.h(q[3])
    generateU(input, q, aux)
    circuit.barrier()
    generateU(weights, q, aux)
    circuit.barrier()
    circuit.h(q[0])
    circuit.x(q[0])
    circuit.h(q[1])
    circuit.x(q[1])
    circuit.h(q[2])
    circuit.x(q[2])
    circuit.h(q[3])
    circuit.x(q[3])
    circuit = ccccx(circuit, q[0], q[1], q[2], q[3], q[4], aux[0], aux[1])
    # circuit.measure(q[4], c[0])
    circuit.barrier()




# %%
# MINST Circuit Generate


def init(circuit, input, work):
    # print("\t", len(input))
    for idx in range(len(input)):
        # if input[idx]<0.5:
        #     circuit.x(work[idx])
        y_v = input[idx].item()
        if y_v > 0:
            alpha = np.arccos(y_v)
        elif y_v < 0:
            alpha = np.pi - np.arccos(-y_v)
        else:
            alpha = np.pi / 2
        if(len(input)==1):
            circuit.ry(alpha, work)
        else:
            circuit.ry(alpha, work[idx])
    # circuit.barrier()


def reset_qbits(circuit, q_set):
    for q in q_set:
        circuit.reset(q)
    circuit.barrier()



def SLP_2_encoding(circuit, q_in, q_en, input=[]):
    if len(input) != 0:
        init(circuit, input, q_in)
    encoder2_1(circuit, q_in, q_en)

def SLP_2_Uw(circuit, q_en, w, aux=[]):
    beg = len(circuit.data)
    generateU(circuit, w, q_en, aux)
    end = len(circuit.data)
    return beg, end

def SLP_4_encoding(circuit, q_in, q_en, input=[], aux=[]):
    if len(input) != 0:
        init(circuit, input, q_in)

    encoder_q_set = []
    for q in q_in:
        encoder_q_set.append(q)
    for q in q_en:
        encoder_q_set.append(q)
    encoder4_2(circuit, encoder_q_set, aux)


def SLP_4_Uw(circuit, q_en, w, aux=[]):
    beg = len(circuit.data)
    generateU(circuit, w, q_en, aux)
    end = len(circuit.data)

    # circuit.barrier()
    # for qbit in q_en[0:2]:
    #     circuit.h(qbit)
    #     circuit.x(qbit)
    #
    # circuit.ccx(q_en[0], q_en[1], q_out)
    # circuit.barrier()

    return beg, end


def SLP_8_encoding(circuit, q_in, q_en, input=[], aux=[]):
    if len(input) != 0:
        init(circuit, input, q_in)

    encoder_q_set = []
    for q in q_in:
        encoder_q_set.append(q)
    for q in q_en:
        encoder_q_set.append(q)
    encoder8_3(circuit, encoder_q_set, aux)

def SLP_8_Uw(circuit, q_en, w, aux=[]):
    beg = len(circuit.data)
    generateU(circuit, w, q_en, aux)
    end = len(circuit.data)

    # circuit.barrier()
    # for qbit in q_en[0:3]:
    #     circuit.h(qbit)
    #     circuit.x(qbit)
    #
    # cccx(circuit, q_en[0], q_en[1], q_en[2], q_out, aux[0])
    # circuit.barrier()

    return beg, end


def SLP_16_encoding(circuit, q_in, q_en, input=[], aux=[]):
    if len(input) != 0:
        init(circuit, input, q_in)

    encoder16_4(circuit, q_in, q_en, aux)


def SLP_16_Uw(circuit, q_en, w, aux=[]):
    beg = len(circuit.data)
    generateU(circuit, w, q_en, aux)
    end = len(circuit.data)



    return beg, end


def reverse_part_circuit(circuit, beg, end):
    for inst, qargs, cargs in reversed(circuit.data[beg:end]):
        circuit.data.append((inst.inverse(), qargs, cargs))
