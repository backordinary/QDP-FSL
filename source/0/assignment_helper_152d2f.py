# https://github.com/narghev/qiskit/blob/6d0e2e79bdc63579d00569543e7bb0bd7c71930c/assignment_helper.py
import itertools
import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute


def get_counts(circuit, num_shots=100):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=num_shots)
        result = job.result()
        counts = result.get_counts(circuit)
    else:
        raise ValueError("Unknown circuit type")
    return counts


def get_single_measurement_counts(circuit, num_shots=100):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=num_shots)
        result = job.result()
        counts = result.get_counts(circuit)
    else:
        raise ValueError("Unknown circuit type")
    return counts


def get_amplitudes(circuit):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        backend = BasicAer.get_backend('statevector_simulator')
        job = execute(circuit, backend)
        amplitudes = job.result().get_statevector(circuit)
    else:
        raise ValueError("Unknown circuit type")
    return amplitudes


def get_classical_bits(circuit):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        classical_bits = circuit.cregs[0].size
    else:
        raise ValueError("Unknown circuit type")
    return classical_bits


def get_circuit_length(circuit):
    if isinstance(circuit, qiskit.circuit.quantumcircuit.QuantumCircuit):
        program_length = sum(circuit.count_ops().values())
    else:
        raise ValueError("Unknown circuit type")
    return program_length


if __name__ == "__main__":
    try:
        import qiskit
        from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
        from qiskit import execute, BasicAer
        from qiskit.quantum_info import Pauli
        from qiskit_aqua import Operator, get_aer_backend
        from qiskit_aqua.components.initial_states import Custom
        is_qiskit = True
    except ImportError:
        is_qiskit = False
    if not is_qiskit:
        raise RuntimeError("No quantum computing framework available!")
    
