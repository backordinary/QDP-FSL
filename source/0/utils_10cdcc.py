# https://github.com/G-Carneiro/GCQ/blob/a557c193c54c4f193c9ffde7f94c576b06972abe/src/utils/utils.py
from typing import Union, Tuple, List

from qiskit import QuantumCircuit, QuantumRegister

qubox_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJRdWFudHVsb29wIFNlcnZlciAo" \
                   "UUNMYWJzKSIsInN1YiI6IlF1YW50dWxvb3AgU2VydmVyIFVTRVIgdG9rZW4iLCJhdWQiOiJ1c" \
                   "2VyIiwiZXhwIjoxNjg0NjI2MDAwLCJuYmYiOjE2NTM1MjIwMDAsImlhdCI6MTY1MzUyMjAwMC" \
                   "wibmFtZSI6IkdhYnJpZWwgTWVkZWlyb3MgTG9wZXMgQ2FybmVpcm8iLCJlbWFpbCI6ImdhYnJ" \
                   "pZWwubWxjQGdyYWQudWZzYy5iciJ9.-boMNQgVIOogC1qVGHHBSbLHNo8XX1nSxcH5LWQ-Er8"

ibmq_token: str = "98e8a2739b6f1ef5e4855f3eb754f84ee240e8cd5370de304cf" \
                  "fa2f527f47fbe3a398efd986329d655f70a18d3607767474efc" \
                  "f32b72d0f3a7dbfadb3e34ee1f"


def create_bell_pair(quantum_circuit: QuantumCircuit,
                     control_qubit: Union[int, QuantumRegister],
                     target_qubit: Union[int, QuantumRegister],
                     state: Tuple[int, int] = (0, 0)
                     ) -> None:
    """
    Returns:
        QuantumCircuit: Circuit that produces a Bell pair
    """
    if state[0]:
        quantum_circuit.x(target_qubit)

    if state[1]:
        quantum_circuit.x(control_qubit)

    quantum_circuit.h(control_qubit)
    quantum_circuit.cx(control_qubit, target_qubit)
    return None


def multiple_control_gate(qc: QuantumCircuit,
                          control_qubits: Union[QuantumRegister, List[int]],
                          work_qubits: Union[QuantumRegister, List[int]],
                          target_qubit: Union[QuantumRegister, int]
                          ) -> None:

    num_ctrl_qubits: int = len(control_qubits)
    qc.ccx(control_qubits[0], control_qubits[1], work_qubits[0])

    for i in range(2, num_ctrl_qubits):
        qc.ccx(control_qubit1=control_qubits[i],
               control_qubit2=work_qubits[i - 2],
               target_qubit=work_qubits[i - 1])

    qc.cz(control_qubit=work_qubits[num_ctrl_qubits - 2], target_qubit=target_qubit[0])

    for i in reversed(range(2, num_ctrl_qubits)):
        qc.ccx(control_qubit1=control_qubits[i],
               control_qubit2=work_qubits[i - 2],
               target_qubit=work_qubits[i - 1])

    qc.ccx(control_qubits[0], control_qubits[1], work_qubits[0])

    return None
