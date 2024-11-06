# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/compiler/readout_error_mitigation.py
from qiskit import execute, Aer
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.measure import Measure
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)


def run_meas_mitigation(circuit, backend=None):
    qubits = _parse_measured_qubits(circuit)
    return _create_meas_cal(qubits, backend)


def _parse_measured_qubits(circuit):
    prog2hw = circuit._layout.get_virtual_bits()
    # print(prog2hw)
    # clbits = circuit.clbits
    # num_clbits = len(clbits)
    hw_qubits = []
    for op in circuit._data:
        # meas = circuit.__getitem__(-i-1)
        if isinstance(op[0], Measure):

            # print(op)
            hw_qubits.append(op[1][0].index)
            # hw_qubits.append(prog2hw[op[1][0]])
    return hw_qubits


def _create_meas_cal(qubits, backend=None):
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubits)
    if backend is None:
        backend = Aer.get_backend('qasm_simulator')
    job = execute(meas_calibs, backend=backend, shots=8192)
    return job, state_labels
