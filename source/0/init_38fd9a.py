# https://github.com/donn/donns-qiskit-tweaks/blob/0b92061e69ab4a715f01ed8284b4d33e4dc85d6a/donns_qiskit_tweaks/gates/__init__.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from typing import List, Optional


from qiskit.circuit.library.standard_gates import HGate, CXGate, CCXGate


def _apply_etof(qc: QuantumCircuit, controls: List[Optional[bool]]) -> QuantumCircuit:
    n = len(controls)

    for qubit, control in zip(range(n), controls):
        if control is not None and not control:
            qc.x(qubit)

    controlling_qubits = []
    for qubit, control in zip(range(n), controls):
        if control is not None:
            controlling_qubits.append(qubit)

    qc.mct(controlling_qubits, n, mode="noancilla")

    for qubit, control in zip(range(n), controls):
        if control is not None and not control:
            qc.x(qubit)

    return qc


class ETOF(Gate):
    def __init__(
        self,
        controls: List[Optional[bool]],
        label: Optional[str] = None,
    ) -> None:
        self.n = len(controls)
        circuit = QuantumCircuit(self.n + 1)

        _apply_etof(circuit, controls)
        parameter_dict = {p: p for p in circuit.parameters}
        super().__init__(
            name="ETOF",
            num_qubits=circuit.num_qubits,
            params=[*parameter_dict.values()],
            label=label,
        )
        self.condition = None

        target = circuit.assign_parameters(parameter_dict, inplace=False)

        if self.num_qubits > 0:
            q = QuantumRegister(self.num_qubits, "q")

        qubit_map = {bit: q[idx] for idx, bit in enumerate(circuit.qubits)}

        qc = QuantumCircuit(q, name=self.name, global_phase=target.global_phase)

        for instruction in target.data:
            qc._append(
                instruction.replace(
                    qubits=tuple(qubit_map[y] for y in instruction.qubits)
                )
            )
        self.definition = qc


H = HGate
CNOT = CXGate
TOF = CCXGate
