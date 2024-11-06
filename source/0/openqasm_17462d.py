# https://github.com/samu-magaz/quantum-factory-method/blob/b348ccbe44502730f3db869bf1f3d0e68701bfd9/examples/openqasm.py
import re
from typing import List
import qiskit
from qiskit_aer import Aer
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

class Circuit:
    pass


class QiskitCircuit(Circuit):
    def __init__(self, built_circuit: qiskit.QuantumCircuit):
        self.built_circuit = built_circuit


class CirqCircuit(Circuit):
    def __init__(self, built_circuit: cirq.Circuit):
        self.built_circuit = built_circuit


class Platform:
    @staticmethod
    def build(openqasm_str: str) -> Circuit:
        pass

    @staticmethod
    def build(circuit: Circuit) -> List[float]:
        pass


class QiskitPlatform(Platform):
    shots = 1024

    @staticmethod
    def build(openqasm_str: str) -> QiskitCircuit:
        return QiskitCircuit(qiskit.QuantumCircuit.from_qasm_str(openqasm_str))

    @staticmethod
    def run(quantum_circuit: QiskitCircuit) -> List[float]:
        circ = quantum_circuit.built_circuit

        simulator = Aer.get_backend("aer_simulator")
        circ = qiskit.transpile(circ, simulator)
        result = simulator.run(circ, shots=QiskitPlatform.shots).result()

        counts = result.get_counts()
        counts = {int(k[0]): v for k,v in counts.items()}

        values = list(range(len(counts)))
        for k,v in counts.items():
            values[k] = v / QiskitPlatform.shots
        return values


class CirqPlatform(Platform):
    shots = 1024

    @staticmethod
    def build(openqasm_str: str) -> CirqCircuit:
        return CirqCircuit(circuit_from_qasm(openqasm_str))

    @staticmethod
    def run(quantum_circuit: CirqCircuit) -> List[float]:
        circ = quantum_circuit.built_circuit

        result = cirq.Simulator().run(circ, repetitions=CirqPlatform.shots)

        counts = result.measurements

        values = list(range(len(counts)))
        temp = 0
        for [v] in list(counts.items())[0][1]:
            temp += v
        values[0] = temp / CirqPlatform.shots
        values[1] = (CirqPlatform.shots - temp) / CirqPlatform.shots
        return values
