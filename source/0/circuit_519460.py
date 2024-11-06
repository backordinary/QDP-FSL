# https://github.com/Allenator/nusynth/blob/7d095cfed9e0cfd035e129e59ac49c9cb6ccba0e/nusynth/circuit.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

import nusynth.squander as s


def xz_rot_1q(input):
    theta, phi = input

    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)

    return qc


def u3_gate_3q(input):
    qc = QuantumCircuit(3)
    qc.u(*input[0:3], 0) # type: ignore
    qc.u(*input[3:6], 1) # type: ignore
    qc.u(*input[6:9], 2) # type: ignore

    return qc


def su2_3q(input, n_reps):
    ansatz = EfficientSU2(
        3, reps=n_reps,
        su2_gates=['ry', 'rz']
    ).decompose().bind_parameters(input)
    qc = QuantumCircuit(3)
    qc.compose(ansatz, inplace=True)

    return qc


def qgan_3q(input, n_reps):
    qc = QuantumCircuit(3)
    for r in range(n_reps):
        pad = 3 * r
        qc.ry(input[pad], 0)
        qc.ry(input[pad + 1], 1)
        qc.ry(input[pad + 2], 2)
        qc.barrier()
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.cz(2, 0)
        qc.barrier()
    pad = 3 * n_reps
    qc.ry(input[pad], 0)
    qc.ry(input[pad + 1], 1)
    qc.ry(input[pad + 2], 2)

    return qc


def squander(input, n_qubits, n_layers_dict=s.default_n_layers_dict):
    n_layers_dict = s.default_n_layers_dict | n_layers_dict
    n_params = s.n_params(n_qubits, n_layers_dict)
    assert len(input) == n_params

    u_gen = (input[i:i + 3] for i in range(0, len(input), 3))
    nu = lambda: next(u_gen)
    qc = QuantumCircuit(n_qubits)

    # initial rotations
    for q in range(n_qubits):
        qc.u(*nu(), q) # type: ignore
    qc.barrier()

    # layers
    for qn in range(n_qubits, 1, -1):
        n_groups = n_layers_dict[qn] // (qn - 1)
        for _ in range(n_groups):
            ctrl = qn - 1
            for targ in range(ctrl):
                qc.cnot(ctrl, targ)
                qc.u(*nu(), targ) # type: ignore
                qc.u(*nu(), ctrl) # type: ignore
        qc.barrier()

    # check params exhausted
    try:
        nu()
        raise RuntimeError('Input parameters not exhausted')
    except StopIteration:
        pass

    return qc
