# https://github.com/rajkk1/qindia-qa101-final-project/blob/f390bca954ab4884ec0fa39a69d574b5c1696830/u_init.py
# Following the import convention given here: https://stackoverflow.com/a/42701457/5013467

from typing import Union

from braket.circuits import Circuit
import cirq
from qiskit import QuantumCircuit, QuantumRegister

def u_init(quantum_language: str = "qiskit") -> Union[QuantumCircuit, cirq.Circuit, Circuit]:
    """Produces a quantum circuit with two qubits rotated by theta_0 and theta_1 respectively using Ry gates"""
    theta_0 = 0.2
    theta_1 = 1.6
    if quantum_language == "qiskit":
        qc = QuantumCircuit(2)
        qc.ry(theta_0,0)
        qc.ry(theta_1,1)
    elif quantum_language == "cirq":
        q0, q1 = cirq.LineQubit.range(2)
        qc = cirq.Circuit()
        qc.append([cirq.ry(theta_0)(q0), cirq.ry(theta_1)(q1)])
    elif quantum_language == "braket":
        qc = Circuit().ry(0, theta_0).ry(1, theta_1)
    else:
        raise ValueError("Desired quantum language not found")
    return qc
