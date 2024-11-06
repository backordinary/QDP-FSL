# https://github.com/AndersHR/quantum_error_mitigation/blob/8aa0806b1433ad420251bc9b6dd25f47f8e08e15/Pauli_twirling.py
from qiskit import QuantumCircuit

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller, Optimize1qGates

import random

PHYSICAL_GATE_CONVERSION = {"X": "u3(pi,0,pi)", "Z": "u1(pi)", "Y": "u3(pi,pi/2,pi/2)"}

def pauli_twirl_cnots(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Pauli twirl CNOT-gates in a given quantum circuit to convert general CNOT-gate errors into
    stochastic Pauli errors.

    :param qc: quantum circuit for which to Pauli twirl all CNOT gates
    :return: Pauli twirled quantum circuit
    """

    # The circuit may be expressed in terms of various types of gates.
    # The 'Unroller' transpiler pass 'unrolls' the circuit to be expressed in terms of the
    # physical gate set [u1,u2,u3,cx]
    unroller = Unroller(["u1", "u2", "u3", "cx"])
    pm = PassManager(unroller)

    unrolled_qc = pm.run(qc)

    circuit_qasm = unrolled_qc.qasm()
    new_circuit_qasm_str = ""

    qreg_name = find_qreg_name(circuit_qasm)

    for i, line in enumerate(circuit_qasm.splitlines()):
        if line[0:2] == "cx":
            new_circuit_qasm_str += pauli_twirl_cnot_gate(qreg_name, line)
        else:
            new_circuit_qasm_str += line + "\n"

    new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

    # The "Optimize1qGates" transpiler pass optimizes chains of single-qubit gates by collapsing them into
    # a single, equivalent u3-gate

    # We want to avoid that the transpiler optimizes CNOT-gates, as the ancillary CNOT-gates must be kept
    # to keep the noise amplification

    optimize1qates = Optimize1qGates()
    pm = PassManager(optimize1qates)

    return pm.run(new_qc)

def noise_amplify_and_pauli_twirl_cnots(qc: QuantumCircuit, amp_factor: int = 1, pauli_twirl: bool = True) -> QuantumCircuit:
    """
    Pauli twirl CNOT-gates and amplify CNOT-noise by extending each CNOT-gate as CNOT^amp_factor.

    Using CNOT*CNOT = I, the identity, and an amp_factor = (2*n + 1) for an integer n, then the
    extended CNOT will have the same action as a single CNOT, but with the noise amplified by
    a factor amp_factor.

    :param qc: Quantum circuit for which to Pauli twirl all CNOT gates and amplify CNOT-noise
    :param amp_factor: The noise amplification factor, must be (2n + 1) for n = 0,1,2,3,...
    :return: Pauli-twirled and noise-amplified Quantum Circuit
    """

    if (amp_factor - 1) % 2 != 0:
        print("Invalid amplification factor")

    # The circuit may be expressed in terms of various types of gates.
    # The 'Unroller' transpiler pass 'unrolls' the circuit to be expressed in terms of the
    # physical gate set [u1,u2,u3,cx]
    unroller = Unroller(["u1","u2","u3","cx"])
    pm = PassManager(unroller)

    unrolled_qc = pm.run(qc)

    circuit_qasm = unrolled_qc.qasm()
    new_circuit_qasm_str = ""

    qreg_name = find_qreg_name(circuit_qasm)

    for i, line in enumerate(circuit_qasm.splitlines()):
        if line[0:2] == "cx":
            for j in range(amp_factor):
                if pauli_twirl:
                    new_circuit_qasm_str += pauli_twirl_cnot_gate(qreg_name, line)
                else:
                    new_circuit_qasm_str += (line + "\n")
        else:
            new_circuit_qasm_str += line + "\n"

    new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

    # The "Optimize1qGates" transpiler pass optimizes chains of single-qubit gates by collapsing them into
    # a single, equivalent u3-gate

    # We want to avoid that the transpiler optimizes CNOT-gates, as the ancillary CNOT-gates must be kept
    # to keep the noise amplification

    optimize1qates = Optimize1qGates()
    pm = PassManager(optimize1qates)

    return pm.run(new_qc)


# HELP FUNCTIONS:

def is_cnot(qasm_line: str):
    if qasm_line[0:2] == "cx":
        return True
    else:
        return False

def find_qreg_name(circuit_qasm: str):
    for line in circuit_qasm.splitlines():
        if line[0:5] == "qreg ":
            qreg_name = ""
            for i in range(5,len(line)):
                if line[i] == "[" or line[i] == ";":
                    break
                elif line[i] != " ":
                    qreg_name += line[i]
            return qreg_name

def find_cnot_control_and_target(qasm_line: str):
    qubits = []
    for i, c in enumerate(qasm_line):
        if c == "[":
            qubit_nr = ""
            for j in range(i+1, len(qasm_line)):
                if qasm_line[j] == "]":
                    break
                qubit_nr += qasm_line[j]
            qubits.append(int(qubit_nr))
    return qubits[0], qubits[1]

def propagate(control_in: str, target_in: str):
    """
    Propagates Pauli gates through a CNOT in accordance with the following circuit identities:

    (X x I) CNOT = CNOT (X x X)
    (I x X) CNOT = CNOT (I x X)
    (Z x I) CNOT = CNOT (I x Z)
    (I x Z) CNOT = XNOT (Z x Z)

    We use the fact that Y = iXZ, where when applied on a state |psi> the factor i is simply a global phase
    which we can ignore. F.ex, I x iXZ |psi> x |phi> = |psi> x (i XZ |phi>) = i (|psi> x |phi'>)

    :param control_in: Pauli gates on control qubit before CNOT
    :param target_in: Pauli gates on target qubit before CNOT
    :return: Equivalent Pauli gates on control and target qubits after CNOT
    """

    control_out, target_out = '', ''
    if 'X' in control_in:
        control_out += 'X'
        target_out += 'X'
    if 'X' in target_in:
        target_out += 'X'
    if 'Z' in control_in:
        control_out += 'Z'
    if 'Z' in target_in:
        control_out += 'Z'
        target_out += 'Z'

    # Pauli gates square to the identity, i.e. XX = I, ZZ = I
    # Remove all such occurences from the control & target out Pauli gate strings
    if 'ZZ' in control_out:
        control_out = control_out[:-2]
    if 'ZZ' in target_out:
        target_out = target_out[:-2]
    if 'XX' in control_out:
        control_out = control_out[2:]
    if 'XX' in target_out:
        target_out = target_out[2:]

    # If no Pauli gates remain then we have the identity gate I
    if control_out == '':
        control_out = 'I'
    if target_out == '':
        target_out = 'I'

    return control_out[::-1], target_out[::-1]

"""
def collapse_pauli_gates(pauli_gates: str):
    decomposed_pauli_gates = ""
    for i, a in enumerate(pauli_gates):
        if a == "Y":
            decomposed_pauli_gates += "XZ"
        else:
            decomposed_pauli_gates += a
    indices_x = [i for i,a in enumerate(decomposed_pauli_gates) if a == "X"]
    indices_z = [i for i,a in enumerate(decomposed_pauli_gates) if a == "Z"]
"""


def apply_qasm_pauli_gate(qreg_name: str, qubit: int, pauli_gates: str):
    """


    :param qreg_name:
    :param qubit: Index of qubit
    :param pauli_gates:
    :return:
    """
    new_qasm_line = ''
    for gate in pauli_gates:
        if gate != 'I':
            u_gate = PHYSICAL_GATE_CONVERSION[gate]
            new_qasm_line += u_gate + ' ' + qreg_name + '[' + str(qubit) + '];' + '\n'
    return new_qasm_line


def pauli_twirl_cnot_gate(qreg_name: str, qasm_line_cnot: str, a: str = "", b: str = ""):
    control, target = find_cnot_control_and_target(qasm_line_cnot)

    pauli_gates = ["I", "X", "Z", "XZ"]

    if a == "":
        a = random.choice(pauli_gates)
    if b == "":
        b = random.choice(pauli_gates)

    # Find gates such that:
    # (a x b) CNOT (c x d) = CNOT for an ideal CNOT-gate,
    # by propagating the Pauli gates through the CNOT

    c, d = propagate(a, b)

    new_qasm_line = apply_qasm_pauli_gate(qreg_name, control, a)
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, target, b)
    new_qasm_line += qasm_line_cnot + '\n'
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, target, d)
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, control, c)

    return new_qasm_line


if __name__ == "__main__":
    qc = QuantumCircuit(2,2)
    qc.h(0)
    qc.cnot(0,1)

    #print(qc.qasm())

    #new_qc = pauli_twirl_cnots(qc)

    #print(new_qc.qasm())

    print(propagate("XZ","Z"))