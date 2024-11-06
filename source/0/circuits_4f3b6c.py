# https://github.com/jblue1/Quantum-Information-Final-Project/blob/074e264190fbb6445d0b2d19d9f4f48424f82011/heisenberg/circuits.py
import qiskit as qi


backend = qi.Aer.get_backend("qasm_simulator")


def expectation_zzz(counts, shots, z_index_list=None):
    """
    Function taken from Davit Khachatryan's stack overflow answer here:
    https://quantumcomputing.stackexchange.com/questions/11408/qiskit-z-expectation-value-from-counts/11411#11411
    :param shots: shots of the experiment
    :param counts: counts obtained from Qiskit's Result.get_counts()
    :param z_index_list: a list of indexes
    :return: the expectation value of ZZ...Z operator for given z_index_list
    """

    if z_index_list is None:
        z_counts = counts
    else:
        z_counts = cut_counts(counts, z_index_list)

    expectation = 0
    for key in z_counts:
        sign = -1
        if key.count("1") % 2 == 0:
            sign = 1
        expectation += sign * z_counts[key] / shots

    return expectation


def cut_counts(counts, bit_indexes):
    """
    Function taken from Davit Khachatryan's stack overflow answer here:
    https://quantumcomputing.stackexchange.com/questions/11408/qiskit-z-expectation-value-from-counts/11411#11411
    :param counts: counts obtained from Qiskit's Result.get_counts()
    :param bit_indexes: a list of indexes
    :return: new_counts for the  specified bit_indexes
    """
    bit_indexes.sort(reverse=True)
    new_counts = {}
    for key in counts:
        new_key = ""
        for index in bit_indexes:
            new_key += key[-1 - index]
        if new_key in new_counts:
            new_counts[new_key] += counts[key]
        else:
            new_counts[new_key] = counts[key]

    return new_counts


def measureCircuit(prepGate, pauliString):
    reverseString = pauliString[::-1]
    circ = qi.QuantumCircuit(4, 4)
    circ.append(prepGate, [0, 1, 2, 3])
    qubitsToMeasure = []

    for i, gate in enumerate(reverseString):
        if gate == "X":
            circ.h(i)
            qubitsToMeasure.append(i)
        if gate == "Y":
            circ.sdg(i)
            circ.h(i)
            qubitsToMeasure.append(i)
        if gate == "Z":
            qubitsToMeasure.append(i)

    circ.measure([0, 1, 2, 3], [0, 1, 2, 3])
    shots = 200



    result = qi.execute(circ, backend=backend, shots=shots).result()
    counts = result.get_counts(circ)
    return expectation_zzz(counts, shots, z_index_list=qubitsToMeasure)


def expectedH(j, b, prepGate):
    first = (
        measureCircuit(prepGate, "XXII")
        + measureCircuit(prepGate, "YYII")
        + measureCircuit(prepGate, "ZZII")
    )
    second = (
        measureCircuit(prepGate, "IXIX")
        + measureCircuit(prepGate, "IYIY")
        + measureCircuit(prepGate, "IZIZ")
    )
    third = (
        measureCircuit(prepGate, "IIXX")
        + measureCircuit(prepGate, "IIYY")
        + measureCircuit(prepGate, "IIZZ")
    )
    fourth = (
        measureCircuit(prepGate, "XIXI")
        + measureCircuit(prepGate, "YIYI")
        + measureCircuit(prepGate, "ZIZI")
    )

    return j * (first + second + third + fourth) + b * (
        measureCircuit(prepGate, "ZIII")
        + measureCircuit(prepGate, "IZII")
        + measureCircuit(prepGate, "IIZI")
        + measureCircuit(prepGate, "IIIZ")
    )


def prepGate(thetas, depth):
    circ = qi.QuantumCircuit(4)
    circ.ry(thetas[0], 0)
    circ.rz(thetas[4 + 0], 0)
    circ.ry(thetas[1], 1)
    circ.rz(thetas[4 + 1], 1)
    circ.ry(thetas[2], 2)
    circ.rz(thetas[4 + 2], 2)
    circ.ry(thetas[3], 3)
    circ.rz(thetas[4 + 3], 3)
    
    for j in range(depth):
        circ.cx(0, 1)
        circ.cx(0, 2)
        circ.cx(0, 3)
        circ.cx(1, 2)
        circ.cx(1, 3)
        circ.cx(2, 3)

        for i in range(4):
            circ.ry(thetas[8*(j+1) + i], i)
            circ.rz(thetas[8*(j+1) + 4 + i], i)

    return circ.to_gate()


def costFnx(x, *args):
    j = args[0]
    b = 1
    depth = args[1]
    gate = prepGate(x, depth)
    return expectedH(j, b, gate)


