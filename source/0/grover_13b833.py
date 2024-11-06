# https://github.com/GizmoBill/QuantumComputing/blob/c379a941f269271c8ce5a8b3d47fa6b85d2af41a/Grover.py
# Copyright (c) 2021 Bill Silver. Licence granted to public under terms of MIT licence
# at https://github.com/GizmoBill/QuantumComputing/blob/main/LICENSE

# ************************
# *                      *
# *  Grover's Algorithm  *
# *                      *
# ************************

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from numpy import pi, sqrt, arcsin

# ***********************************************
# *                                             *
# *  Create Multi-Control X from Toffoli Gates  *
# *                                             *
# ***********************************************
#
# Add a controlled X gate to the specified QuantumCircuit, using the specified
# list of control qubits, the specified target qubit, and the specified list
# of ancilla qubits. Any number of controls >= 0 can be specified. The number
# of ancillas must be at least two fewer than the number of controls; these
# qubits will be left in their original state. A minimum depth circuit is
# created.
def multiX(qc, controls, target, ancilla) :
    n = len(controls)
    uncompute = []
    if n > 2 :
        ai = 0  # ancilla index

        while n > 2 :
            newControls = [];
            for i in range(0, n, 2):
                if i == n - 1 :
                    newControls.append(controls[i])
                else :
                    qc.toffoli(controls[i], controls[i + 1], ancilla[ai])
                    uncompute.insert(0, [controls[i], controls[i + 1], ancilla[ai]])
                    newControls.append(ancilla[ai])
                    ai = ai + 1
            controls = newControls
            n = len(controls)

    if n == 0 :
        qc.x(target)
    elif n == 1 :
        qc.cx(controls[0], target)
    else :
        qc.toffoli(controls[0], controls[1], target)
    for t in uncompute :
        qc.toffoli(t[0], t[1], t[2])

# **********************************
# *                                *
# *  Select Oracle's Winner State  *
# *                                *
# **********************************
#
# An oracle is created with a multi-controlled X gate (MCX), one control qubit for
# each oracle qubit. With just the MCX, the winning state would br all 1's. To select
# any winner, this function adds X gates to every qubit whose corresponding bit in
# winner is 0. The X gates must be added both before and after the MCX. so that the
# desired winner is shifted to the all 1's state before the MCX, and back after the
# MCX.

def oracle(qc, qubits, winner) :
    for q in qubits :
        if (winner & 1) == 0 :
            qc.x(q)
        winner = winner >> 1

# ************************************
# *                                  *
# *  Make and Return Grover Circuit  *
# *                                  *
# ************************************

def grover(numQbits, winner) :
    # Number of iterations
    n = int(pi / (4 * arcsin(2 ** (-numQbits / 2))))

    # Create circuit. There are quantum and classical registers whose sizes are the
    # specified numQbits. There are numQbits-2 ancillas for the MCX gates (oracle
    # and diffuser), and one more for the oracle target qubit for phase kickback.
    qc = QuantumCircuit(numQbits, numQbits, name = "Grover:q={0},i={1},w={2}".format(numQbits, n, winner))
    qList = list(qc.qregs[0])
    areg = AncillaRegister(numQbits - 1)
    qc.add_register(areg)
    target = areg[0]
    ancilla = [areg[i] for i in range(1, areg.size)]

    # Initialize the primary qubits to |+> and the target ancilla to |->
    for q in range(numQbits) :
        qc.h(qList[q])
    qc.x(target)
    qc.h(target)

    for s in range(n) :
        # Oracle
        oracle(qc, qList, winner)
        multiX(qc, qList, target, ancilla)
        oracle(qc, qList, winner)

        # Diffuse (amplitude amplification)
        for q in range(1, numQbits) :
            qc.h(qList[q])
            qc.x(qList[q])
        qc.z(qList[0])
        multiX(qc, [qList[i] for i in range(1, numQbits)], qList[0], ancilla)
        qc.z(qList[0])
        for q in range(1, numQbits) :
            qc.x(qList[q])
            qc.h(qList[q])

    # Measure the result
    for q in range(numQbits) :
        qc.measure(qList[q], qc.cregs[0][q])

    print("{0} qubits, {1} iterations, winner = {2}".format(numQbits, n, winner))

    return qc
