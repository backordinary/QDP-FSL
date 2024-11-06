# https://github.com/akotil/qcircuit-optimization/blob/66416362d466cdae89c8cbcc5f2bfb1c29957067/converters.py
import re

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate


class Parser:

    def __init__(self, input_file):
        self.number_of_qubits = 0
        self.input_file = input_file

    def qc_to_netlist(self) -> list:
        netlist = []
        with open(self.input_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    self.number_of_qubits = int(line.split(" ")[-1][0]) + 1
                elif i == len(lines) - 1:
                    continue
                else:
                    tokens = line.split(" ")
                    gate = re.findall(r'"([^"]*)"', tokens[0])[0]
                    qubit_idx = int(tokens[0][tokens[0].find("(") + 1:tokens[0].find(")")])
                    dagger = True if tokens[0].find("*") != -1 else False
                    c_str = tokens[2]

                    control_indices = []
                    if c_str.startswith("controls"):
                        control_indices = c_str[c_str.find("[") + 1:c_str.find("]")].split(",")
                        control_indices = [int(s) for s in control_indices]
                    netlist.append((gate, qubit_idx, control_indices, dagger))
        return netlist

    def netlist_to_qiskit_circuit(self, netlist: list) -> QuantumCircuit:
        qc = QuantumCircuit(self.number_of_qubits)
        for i, block in enumerate(netlist):
            gate, qubit, controls, dagger = block
            if gate == "H":
                qc.h(qubit)
            elif gate == "not":
                if not controls:
                    qc.x(qubit)
                elif len(controls) == 1:
                    qc.cx(controls[0], qubit)
                elif len(controls == 2):
                    # qc.ccx(controls[0], controls[1], qubit)
                    self.apply_toffoli_transformation(qc, controls[0], controls[1], qubit)
            elif gate == "Z":
                if not controls:
                    qc.z(qubit)
                elif len(controls) == 1:
                    qc.cz(controls[0], qubit)
                elif len(controls) == 2:
                    # Doubly controlled Z-Gate is converted into a Toffoli Gate with Hadamard transformations
                    qc.h(qubit)
                    self.apply_toffoli_transformation(qc, controls[0], controls[1], qubit)
                    qc.h(qubit)
                    # Apply the Toffoli Gate transformation afterwards
                    # z = ZGate().control(2)
                    # qc.append(z, controls + [qubit])
            elif gate == "T":
                if dagger:
                    qc.rz(-np.pi / 4, qubit)
                else:
                    qc.rz(np.pi / 4, qubit)

            elif gate == "S":
                if dagger:
                    qc.rz(-np.pi / 2, qubit)
                else:
                    qc.rz(np.pi / 2, qubit)

        return qc

    @staticmethod
    def apply_toffoli_transformation(qc: QuantumCircuit, c1: int, c2: int, target: int):
        qc.h(target)
        qc.cnot(c2, target)
        qc.append(RZGate(-np.pi / 4), [target])
        qc.cnot(c1, target)
        qc.rz(np.pi / 4, target)
        qc.cnot(c2, target)
        qc.append(RZGate(-np.pi / 4), [target])
        qc.cnot(c1, target)
        qc.cnot(c1, c2)
        qc.append(RZGate(-np.pi / 4), [c2])
        qc.cnot(c1, c2)
        qc.rz(np.pi / 4, c1)
        qc.rz(np.pi / 4, c2)
        qc.rz(np.pi / 4, target)
        qc.h(target)
