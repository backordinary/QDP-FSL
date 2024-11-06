# https://github.com/bartek-bartlomiej/master-thesis/blob/2217bde7eaac7aa882c921290f612b8d704ea6ee/utils/circuit_creation.py
from qiskit import QuantumRegister, QuantumCircuit

from utils.typing_ import Name, QRegsSpec


def create_circuit(regs: QRegsSpec, name: Name) -> QuantumCircuit:
    qregs = [QuantumRegister(size, name=name) for name, size in regs.items()]
    return QuantumCircuit(*qregs, name=name)
