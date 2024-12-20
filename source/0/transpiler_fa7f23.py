# https://github.com/maghamalyan/qfh-2019/blob/57e59705996b5ab1561918c359aa131e20ced993/web/dqc/transpiler.py
from math import pi
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.extensions import CnotGate, XGate, HGate, TGate, TdgGate

from dqc.BaseInstructions import BaseInstruction, RZInstruction, CNOTInstruction, \
    XInstruction, HInstruction, TInstruction


def to_base_instructions(inst: Gate, qubits: List[Tuple[str, int]]):
    if isinstance(inst, XGate):
        (conn_name, qubit), = qubits

        return [XInstruction(conn_name, qubit)]

    if isinstance(inst, HGate):
        (conn_name, qubit), = qubits

        return [HInstruction(conn_name, qubit)]

    if isinstance(inst, TGate):
        (conn_name, qubit), = qubits

        return [TInstruction(conn_name, qubit)]

    if isinstance(inst, TdgGate):
        (conn_name, qubit), = qubits

        return [RZInstruction(conn_name, qubit, 2*pi-pi/4)]

    if isinstance(inst, CnotGate):
        (c_name, control), (t_name, target) = qubits

        return [CNOTInstruction((c_name, control), (t_name, target))]


def get_node_and_qubit_for_index(nodes: List[str], index, qubits_per_node: int):
    return nodes[int(index/qubits_per_node)], index - int(index/qubits_per_node)*qubits_per_node


def from_qiskit_to_base_instructions(qc: QuantumCircuit, node_names: List[str]) -> Dict[str, List[BaseInstruction]]:
    qubits_per_node = int(len(qc.qubits)/len(node_names))
    transpiled = transpile(qc, basis_gates=['cx', 'x', 'h', 't', 'tdg'])

    base_instructions = {name: [] for name in node_names}

    for inst, q_reg, c_reg in transpiled:
        if inst.name in ['measure', 'reset', 'barrier', 'snapshot']:
            continue
        qubits = [get_node_and_qubit_for_index(node_names, index, qubits_per_node) for reg, index in q_reg]
        nodes = set([n for n, q in qubits])
        for node in nodes:
            insts = to_base_instructions(inst, qubits)
            base_instructions[node] += to_base_instructions(inst, qubits) if insts else []

    return base_instructions
