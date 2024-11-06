# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/d2a86a20922811d2eae5efafa05268b0f249c995/multi_placing/allocate_qc_manually.py
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.compiler import transpile


def allocate_qc_manually(
    qc_map: List[Tuple[QuantumCircuit, Dict[Qubit, int]]]
) -> QuantumCircuit:
    """allocate multipul circuit to hardware qubits and produce it as
    a single combined circuit

    Args:
        qc_map: Pairs of QuantumCircuit and its mapping (dict).
                Only one circuit should have a measurements
                * virtual to physical::

                    {qr[0]: 0,
                    qr[1]: 1,
                    qr[2]: 2}
    Return:
        QuantumCircuit
    """

    comp_qc, layout_d = _compose_qc(qc_map)

    return comp_qc, layout_d


def _compose_qc(qc_layout):
    """Compose each qc and return new multitask qc"""
    qubit_counter = 0
    clbit_counter = 0
    layout_dict = {}
    composed_qc = QuantumCircuit()
    """FIXME
    QuantumCircuit.compose -> QuantumCircuit.extend
    に書き換える
    https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html#qiskit.circuit.QuantumCircuit.extend
    """
    for qc, layout in qc_layout:
        num_qubits = qc.num_qubits
        num_clbits = qc.num_clbits

        qubits = list(range(qubit_counter, qubit_counter + num_qubits))

        for qreg in qc.qregs:
            composed_qc.add_register(qreg)

        if num_clbits > 0:
            cr = ClassicalRegister(size=num_clbits, name=None)
            composed_qc.add_register(cr)
            clbits = list(range(clbit_counter, clbit_counter + num_clbits))

            composed_qc.compose(qc, qubits=qubits, clbits=clbits, inplace=True)
        else:
            composed_qc.compose(qc, qubits=qubits, inplace=True)

        qubit_counter += num_qubits
        clbit_counter += num_clbits

        layout_dict.update(layout)

    return composed_qc, layout_dict
