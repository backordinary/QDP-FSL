# https://github.com/cochaviz/qubic/blob/a709fa44c856e4818dc42407df9b9c6dececc263/src/circuit_solver.py
import os

from qiskit import execute
from qiskit.circuit import QuantumRegister, QuantumCircuit
from quantuminspire.credentials import get_authentication
from quantuminspire.qiskit import QI

QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')

authentication = get_authentication()
QI.set_authentication(authentication, QI_URL)

qi_backend = QI.get_backend('QX single-node simulator')
starmon_qi_backend = QI.get_backend('Starmon-5')


def add_gate_to_circ(qubit_index, gate_char, qr, qc, control_qubit_index=None):
    """
    Adds gate to quantum circuit

    @param qubit_index: index of target qubit in the quantum register object qr
    @type qubit_index: int
    @param gate_char: gate encoded as character. ie: 'h', 'x', 'cx', 'z', 'zh' etc.
    @type gate_char: char
    @type qr: QuantumRegister
    @type qc: QuantumCircuit
    @param control_qubit_index: index of control qubit for gates such as: cx
    @type control_qubit_index: int

    @raises ValueError if the given gate is not implemented.
    """
    if gate_char == 'h':
        qc.h(qr[qubit_index])
    elif gate_char == 'x':
        qc.x(qr[qubit_index])
    elif gate_char == 'z':
        qc.z(qr[qubit_index])
    elif gate_char == 'zh':
        qc.z(qr[qubit_index])
        qc.h(qr[qubit_index])
    elif gate_char == 'cx':
        qc.cx(qr[control_qubit_index], qr[qubit_index])
    else:
        raise ValueError('gate is not implemented')


def resolve_circuit(gates_list):
    """
    Resolves the given list of gates on the QuantumInspire backend(QX single-node simulator).

    @param gates_list: list of Gate objects
    @type gates_list: list(Gate)

    @return: dictionary with states as keys and their measured bit as value.
    @rtype: dict
    """
    # array of qubits to be used in circuit
    qubits = []

    for gate in gates_list:
        if gate.target_state not in qubits:
            qubits.append(gate.target_state)

        if gate.control_state is not None and \
                gate.control_state not in qubits:
            qubits.append(gate.control_state)

    q = QuantumRegister(len(qubits), 'q')
    qc = QuantumCircuit(q, name="tic-tac-toe circuit")

    # add gates to quantum circuit
    for gate in gates_list:
        qubit_index = qubits.index(gate.target_state)
        control_qubit_index = None

        if gate.control_state is not None:
            control_qubit_index = qubits.index(gate.control_state)

        add_gate_to_circ(qubit_index, gate.gate_char, q, qc, control_qubit_index=control_qubit_index)

    qc.measure_all()

    qi_job = execute(qc, backend=qi_backend, shots=1)
    qi_result = qi_job.result()
    histogram = qi_result.get_counts(qc)

    # get run result
    result = next(iter(histogram.items()))[0]
    # dict with state on key and measured bit on value
    res = dict()

    for qubit in range(len(qubits)):
        # note that the result of qubit q0 is shown on the right-most classical result register
        # ie: q0: 0, q1: 1, q2: 1 -> result is seen as: 110 (q2, q1, q0)
        qubit_measured_state = result[len(result) - qubit - 1]
        res[qubits[qubit]] = qubit_measured_state

    return res


def quantum_coin_flip(list):
    """
    does a random measurement on an qubit in superposition, then
    returns a 0 or 1, with p(0) = p(1) = 50%
    """
    qc = QuantumCircuit(1, name='coin-flip')    # quantum circuit with 1 qubit

    qc.h(0)  # place hadamard on qubit

    qc.measure_all()

    qi_job = execute(qc, backend=starmon_qi_backend, shots=1)
    qi_result = qi_job.result()

    # return measurement result: 0 or 1
    list.append(next(iter(qi_result.get_counts(qc).items()))[0])


