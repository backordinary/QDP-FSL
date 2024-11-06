# https://github.com/UST-QuAntiL/QuantumTranspiler/blob/a7cf54a83f26da7ef758fe5a17f7b8cf98dcc1a7/test/third_party_converter/pytket_converter.py
from pytket.qasm import circuit_to_qasm_str, circuit_from_qasm_str
from pytket.qiskit import tk_to_qiskit, qiskit_to_tk
import cirq
from pytket.cirq import tk_to_cirq, cirq_to_tk
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pytket.pyquil import pyquil_to_tk, tk_to_pyquil
from pyquil import Program
from qiskit import QuantumCircuit
from circuit.qiskit_utility import qasm_to_dag, dag_to_qasm
from qiskit.dagcircuit import DAGCircuit

class PytketConverter: 
    @staticmethod
    def qiskit_to_pyquil(circuit: QuantumCircuit) -> Program:
        program = tk_to_qiskit(qiskit_to_tk(circuit))
        return program