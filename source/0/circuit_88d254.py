# https://github.com/lionelchg/LesAtomesDeSavoie/blob/9e0c37ce2ea634446b0406ceea90d9432461ae86/qiskit/grover/circuit.py
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

def initialize_s(qc:QuantumCircuit, qubits:list) -> QuantumCircuit:
    """ Apply a H-gate to 'qubits' in qc 

    :param qc: QuantumCircuit object
    :type qc: QuantumCircuit
    :param qubits: the qubits that are concerned by the application of H
    :type qubits: list
    :return: The initialized H gate
    :rtype: QuantumCircuit
    """
    for q in qubits:
        qc.h(q)
    return qc

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

def grover(nqubits:int, niterations:int, oracle:Gate) -> QuantumCircuit:
    """ Apply grover algorithm with niterations of 
    amplification

    :param nqubits: number of qubits
    :type nqubits: int
    :param niterations: number of iterations
    :type niterations: int
    :return: The grover circuit
    :rtype: QuantumCircuit
    """
    grover_circuit = QuantumCircuit(nqubits)
    grover_circuit = initialize_s(grover_circuit, list(range(nqubits)))
    grover_circuit.append(oracle, list(range(nqubits)))
    for _ in range(niterations):
        grover_circuit.append(diffuser(nqubits), list(range(nqubits)))
    grover_circuit.measure_all()
    grover_circuit.draw(output='mpl', 
        filename=f'figures/{nqubits:d}bits_{niterations:d}_its')
    return grover_circuit
