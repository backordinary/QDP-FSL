# https://github.com/jiangtong1000/learn_QuantumCompute/blob/d4647e192f7321b83c8b28df9fb4aa940232a585/evolve_mix.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.aer import Aer
from scipy import linalg
from qiskit import execute
from qiskit.quantum_info import Statevector


def make_circ_m(thetas, theta_i, theta_j, qcidx):
    assert qcidx in [0, 1]
    circ = QuantumCircuit(9)
    circ.h(0)
    circ.h(1)
    circ.h(2)
    circ.h(5)
    circ.h(6)
    apply_dic = {0: [circ.crx, 1, 3], 1: [circ.crx, 2, 4],
                 2: [circ.rzz, 3, 4], 3: [circ.rx, 3],
                 4: [circ.rx, 4], 5: [circ.rzz, 3, 4]}
    apply_diff_dic = {0: circ.ccx, 1: circ.ccx,
                      2: circ.cz, 3: circ.cx,
                      4: circ.cx, 5: circ.cz}
    if 0 == theta_i:
        if qcidx == 1:
            circ.x(0)
        func = apply_diff_dic[0]
        if 0 in [0, 1]:
            func(0, apply_dic[0][1], apply_dic[0][2])
        elif 0 in [2, 5]:
            func(0, apply_dic[0][1])
            func(0, apply_dic[0][2])
        else:
            func(0, apply_dic[0][1])
        if qcidx == 1:
            circ.x(0)
    func = apply_dic[0][0]
    if len(apply_dic[0]) == 2:
        func(thetas[0], apply_dic[0][1])
    elif len(apply_dic[0]) == 3:
        func(thetas[0], apply_dic[0][1], apply_dic[0][2])
    if 1 == theta_i:
        if qcidx == 1:
            circ.x(0)
        func = apply_diff_dic[1]
        if 1 in [0, 1]:
            func(0, apply_dic[1][1], apply_dic[1][2])
        elif 1 in [2, 5]:
            func(0, apply_dic[1][1])
            func(0, apply_dic[1][2])
        else:
            func(0, apply_dic[1][1])
        if qcidx == 1:
            circ.x(0)
    func = apply_dic[1][0]
    if len(apply_dic[1]) == 2:
        func(thetas[1], apply_dic[1][1])
    elif len(apply_dic[1]) == 3:
        func(thetas[1], apply_dic[1][1], apply_dic[1][2])
    if 2 == theta_i:
        if qcidx == 1:
            circ.x(0)
        func = apply_diff_dic[2]
        if 2 in [0, 1]:
            func(0, apply_dic[2][1], apply_dic[2][2])
        elif 2 in [2, 5]:
            func(0, apply_dic[2][1])
            func(0, apply_dic[2][2])
        else:
            func(0, apply_dic[2][1])
        if qcidx == 1:
            circ.x(0)
    func = apply_dic[2][0]
    if len(apply_dic[2]) == 2:
        func(thetas[2], apply_dic[2][1])
    elif len(apply_dic[2]) == 3:
        func(thetas[2], apply_dic[2][1], apply_dic[2][2])
    if 3 == theta_i:
        if qcidx == 1:
            circ.x(0)
        func = apply_diff_dic[3]
        if 3 in [0, 1]:
            func(0, apply_dic[3][1], apply_dic[3][2])
        elif 3 in [2, 5]:
            func(0, apply_dic[3][1])
            func(0, apply_dic[3][2])
        else:
            func(0, apply_dic[3][1])
        if qcidx == 1:
            circ.x(0)
    func = apply_dic[3][0]
    if len(apply_dic[3]) == 2:
        func(thetas[3], apply_dic[3][1])
    elif len(apply_dic[3]) == 3:
        func(thetas[3], apply_dic[3][1], apply_dic[3][2])
    if 4 == theta_i:
        if qcidx == 1:
            circ.x(0)
        func = apply_diff_dic[4]
        if 4 in [0, 1]:
            func(0, apply_dic[4][1], apply_dic[4][2])
        elif 4 in [2, 5]:
            func(0, apply_dic[4][1])
            func(0, apply_dic[4][2])
        else:
            func(0, apply_dic[4][1])
        if qcidx == 1:
            circ.x(0)
    func = apply_dic[4][0]
    if len(apply_dic[4]) == 2:
        func(thetas[4], apply_dic[4][1])
    elif len(apply_dic[4]) == 3:
        func(thetas[4], apply_dic[4][1], apply_dic[4][2])
    if 5 == theta_i:
        if qcidx == 1:
            circ.x(0)
        func = apply_diff_dic[5]
        if 5 in [0, 1]:
            func(0, apply_dic[5][1], apply_dic[5][2])
        elif 5 in [2, 5]:
            func(0, apply_dic[5][1])
            func(0, apply_dic[5][2])
        else:
            func(0, apply_dic[5][1])
        if qcidx == 1:
            circ.x(0)
    func = apply_dic[5][0]
    if len(apply_dic[5]) == 2:
        func(thetas[5], apply_dic[5][1])
    elif len(apply_dic[5]) == 3:
        func(thetas[5], apply_dic[5][1], apply_dic[5][2])
    apply_dic = {0: [circ.crx, 5, 7], 1: [circ.crx, 6, 8],
                 2: [circ.rzz, 7, 8], 3: [circ.rx, 7],
                 4: [circ.rx, 8], 5: [circ.rzz, 7, 8]}
    apply_diff_dic = {0: circ.ccx, 1: circ.ccx,
                      2: circ.cz, 3: circ.cx,
                      4: circ.cx, 5: circ.cz}
    if 0 == theta_j:
        circ.x(0)
        func = apply_diff_dic[0]
        if 0 in [0, 1]:
            func(0, apply_dic[0][1], apply_dic[0][2])
        elif 0 in [2, 5]:
            func(0, apply_dic[0][1])
            func(0, apply_dic[0][2])
        else:
            func(0, apply_dic[0][1])
        circ.x(0)
    func = apply_dic[0][0]
    if len(apply_dic[0]) == 2:
        func(thetas[0], apply_dic[0][1])
    elif len(apply_dic[0]) == 3:
        func(thetas[0], apply_dic[0][1], apply_dic[0][2])
    if 1 == theta_j:
        circ.x(0)
        func = apply_diff_dic[1]
        if 1 in [0, 1]:
            func(0, apply_dic[1][1], apply_dic[1][2])
        elif 1 in [2, 5]:
            func(0, apply_dic[1][1])
            func(0, apply_dic[1][2])
        else:
            func(0, apply_dic[1][1])
        circ.x(0)
    func = apply_dic[1][0]
    if len(apply_dic[1]) == 2:
        func(thetas[1], apply_dic[1][1])
    elif len(apply_dic[1]) == 3:
        func(thetas[1], apply_dic[1][1], apply_dic[1][2])
    if 2 == theta_j:
        circ.x(0)
        func = apply_diff_dic[2]
        if 2 in [0, 1]:
            func(0, apply_dic[2][1], apply_dic[2][2])
        elif 2 in [2, 5]:
            func(0, apply_dic[2][1])
            func(0, apply_dic[2][2])
        else:
            func(0, apply_dic[2][1])
        circ.x(0)
    func = apply_dic[2][0]
    if len(apply_dic[2]) == 2:
        func(thetas[2], apply_dic[2][1])
    elif len(apply_dic[2]) == 3:
        func(thetas[2], apply_dic[2][1], apply_dic[2][2])
    if 3 == theta_j:
        circ.x(0)
        func = apply_diff_dic[3]
        if 3 in [0, 1]:
            func(0, apply_dic[3][1], apply_dic[3][2])
        elif 3 in [2, 5]:
            func(0, apply_dic[3][1])
            func(0, apply_dic[3][2])
        else:
            func(0, apply_dic[3][1])
        circ.x(0)
    func = apply_dic[3][0]
    if len(apply_dic[3]) == 2:
        func(thetas[3], apply_dic[3][1])
    elif len(apply_dic[3]) == 3:
        func(thetas[3], apply_dic[3][1], apply_dic[3][2])
    if 4 == theta_j:
        circ.x(0)
        func = apply_diff_dic[4]
        if 4 in [0, 1]:
            func(0, apply_dic[4][1], apply_dic[4][2])
        elif 4 in [2, 5]:
            func(0, apply_dic[4][1])
            func(0, apply_dic[4][2])
        else:
            func(0, apply_dic[4][1])
        circ.x(0)
    func = apply_dic[4][0]
    if len(apply_dic[4]) == 2:
        func(thetas[4], apply_dic[4][1])
    elif len(apply_dic[4]) == 3:
        func(thetas[4], apply_dic[4][1], apply_dic[4][2])
    if 5 == theta_j:
        circ.x(0)
        func = apply_diff_dic[5]
        if 5 in [0, 1]:
            func(0, apply_dic[5][1], apply_dic[5][2])
        elif 5 in [2, 5]:
            func(0, apply_dic[5][1])
            func(0, apply_dic[5][2])
        else:
            func(0, apply_dic[5][1])
        circ.x(0)
    func = apply_dic[5][0]
    if len(apply_dic[5]) == 2:
        func(thetas[5], apply_dic[5][1])
    elif len(apply_dic[5]) == 3:
        func(thetas[5], apply_dic[5][1], apply_dic[5][2])
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def psi_partial_thetai(thetas, theta_i, part):
    circ = QuantumCircuit(9)
    circ.h(0)
    assert part in ["real", "imag"]
    if part == "imag":
        circ.sdg(0)
    circ.h(1)
    circ.h(2)
    circ.h(5)
    circ.h(6)
    apply_dic = {0: [circ.crx, 1, 3], 1: [circ.crx, 2, 4],
                 2: [circ.rzz, 3, 4], 3: [circ.rx, 3],
                 4: [circ.rx, 4], 5: [circ.rzz, 3, 4]}
    apply_diff_dic = {0: circ.ccx, 1: circ.ccx,
                      2: circ.cz, 3: circ.cx,
                      4: circ.cx, 5: circ.cz}
    if 0 == theta_i:
        circ.x(0)
        func = apply_diff_dic[0]
        if 0 in [0, 1]:
            func(0, apply_dic[0][1], apply_dic[0][2])
        elif 0 in [2, 5]:
            func(0, apply_dic[0][1])
            func(0, apply_dic[0][2])
        else:
            func(0, apply_dic[0][1])
        circ.x(0)
    func = apply_dic[0][0]
    if len(apply_dic[0]) == 2:
        func(thetas[0], apply_dic[0][1])
    elif len(apply_dic[0]) == 3:
        func(thetas[0], apply_dic[0][1], apply_dic[0][2])
    if 1 == theta_i:
        circ.x(0)
        func = apply_diff_dic[1]
        if 1 in [0, 1]:
            func(0, apply_dic[1][1], apply_dic[1][2])
        elif 1 in [2, 5]:
            func(0, apply_dic[1][1])
            func(0, apply_dic[1][2])
        else:
            func(0, apply_dic[1][1])
        circ.x(0)
    func = apply_dic[1][0]
    if len(apply_dic[1]) == 2:
        func(thetas[1], apply_dic[1][1])
    elif len(apply_dic[1]) == 3:
        func(thetas[1], apply_dic[1][1], apply_dic[1][2])
    if 2 == theta_i:
        circ.x(0)
        func = apply_diff_dic[2]
        if 2 in [0, 1]:
            func(0, apply_dic[2][1], apply_dic[2][2])
        elif 2 in [2, 5]:
            func(0, apply_dic[2][1])
            func(0, apply_dic[2][2])
        else:
            func(0, apply_dic[2][1])
        circ.x(0)
    func = apply_dic[2][0]
    if len(apply_dic[2]) == 2:
        func(thetas[2], apply_dic[2][1])
    elif len(apply_dic[2]) == 3:
        func(thetas[2], apply_dic[2][1], apply_dic[2][2])
    if 3 == theta_i:
        circ.x(0)
        func = apply_diff_dic[3]
        if 3 in [0, 1]:
            func(0, apply_dic[3][1], apply_dic[3][2])
        elif 3 in [2, 5]:
            func(0, apply_dic[3][1])
            func(0, apply_dic[3][2])
        else:
            func(0, apply_dic[3][1])
        circ.x(0)
    func = apply_dic[3][0]
    if len(apply_dic[3]) == 2:
        func(thetas[3], apply_dic[3][1])
    elif len(apply_dic[3]) == 3:
        func(thetas[3], apply_dic[3][1], apply_dic[3][2])
    if 4 == theta_i:
        circ.x(0)
        func = apply_diff_dic[4]
        if 4 in [0, 1]:
            func(0, apply_dic[4][1], apply_dic[4][2])
        elif 4 in [2, 5]:
            func(0, apply_dic[4][1])
            func(0, apply_dic[4][2])
        else:
            func(0, apply_dic[4][1])
        circ.x(0)
    func = apply_dic[4][0]
    if len(apply_dic[4]) == 2:
        func(thetas[4], apply_dic[4][1])
    elif len(apply_dic[4]) == 3:
        func(thetas[4], apply_dic[4][1], apply_dic[4][2])
    if 5 == theta_i:
        circ.x(0)
        func = apply_diff_dic[5]
        if 5 in [0, 1]:
            func(0, apply_dic[5][1], apply_dic[5][2])
        elif 5 in [2, 5]:
            func(0, apply_dic[5][1])
            func(0, apply_dic[5][2])
        else:
            func(0, apply_dic[5][1])
        circ.x(0)
    func = apply_dic[5][0]
    if len(apply_dic[5]) == 2:
        func(thetas[5], apply_dic[5][1])
    elif len(apply_dic[5]) == 3:
        func(thetas[5], apply_dic[5][1], apply_dic[5][2])
    apply_dic = {0: [circ.crx, 5, 7], 1: [circ.crx, 6, 8],
                 2: [circ.rzz, 7, 8], 3: [circ.rx, 7],
                 4: [circ.rx, 8], 5: [circ.rzz, 7, 8]}
    func = apply_dic[0][0]
    if len(apply_dic[0]) == 2:
        func(thetas[0], apply_dic[0][1])
    elif len(apply_dic[0]) == 3:
        func(thetas[0], apply_dic[0][1], apply_dic[0][2])
    func = apply_dic[1][0]
    if len(apply_dic[1]) == 2:
        func(thetas[1], apply_dic[1][1])
    elif len(apply_dic[1]) == 3:
        func(thetas[1], apply_dic[1][1], apply_dic[1][2])
    func = apply_dic[2][0]
    if len(apply_dic[2]) == 2:
        func(thetas[2], apply_dic[2][1])
    elif len(apply_dic[2]) == 3:
        func(thetas[2], apply_dic[2][1], apply_dic[2][2])
    func = apply_dic[3][0]
    if len(apply_dic[3]) == 2:
        func(thetas[3], apply_dic[3][1])
    elif len(apply_dic[3]) == 3:
        func(thetas[3], apply_dic[3][1], apply_dic[3][2])
    func = apply_dic[4][0]
    if len(apply_dic[4]) == 2:
        func(thetas[4], apply_dic[4][1])
    elif len(apply_dic[4]) == 3:
        func(thetas[4], apply_dic[4][1], apply_dic[4][2])
    func = apply_dic[5][0]
    if len(apply_dic[5]) == 2:
        func(thetas[5], apply_dic[5][1])
    elif len(apply_dic[5]) == 3:
        func(thetas[5], apply_dic[5][1], apply_dic[5][2])
    return circ


def make_circ_xxzz(thetas, theta_i, oper_idx, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_diff_dic = {0: [circ.cx, 7], 1: [circ.cx, 8],
                      2: [circ.cz, 7, 8]}
    if qcidx == 0:
        circ.x(0)
    func = apply_diff_dic[oper_idx][0]
    func(0, apply_diff_dic[oper_idx][1])
    if len(apply_diff_dic[oper_idx]) == 3:
        func(0, apply_diff_dic[oper_idx][2])
    if qcidx == 0:
        circ.x(0)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_xy1(thetas, theta_i, oper_idx, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_qubit = {0: 7, 1: 8}
    if qcidx == 0:
        circ.cx(0, apply_qubit[oper_idx])
        circ.cy(0, apply_qubit[oper_idx])
    elif qcidx == 1:
        circ.x(0)
        circ.cy(0, apply_qubit[oper_idx])
        circ.cx(0, apply_qubit[oper_idx])
        circ.x(0)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_xx(thetas, theta_i, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "imag")
    apply_gates = {0: [circ.x, 7],
                   1: [circ.y, 7],
                   2: [circ.x, 8],
                   3: [circ.y, 8]}
    func, qubit = apply_gates[qcidx]
    func(qubit)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_xy(thetas, theta_i, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    if qcidx == 0:
        circ.x(0)
        circ.cx(0, 7)
        circ.x(0)
        circ.cy(0, 7)
    if qcidx == 1:
        circ.x(0)
        circ.cy(0, 7)
        circ.x(0)
        circ.cx(0, 7)
    if qcidx == 2:
        circ.x(0)
        circ.cx(0, 8)
        circ.x(0)
        circ.cy(0, 8)
    if qcidx == 3:
        circ.x(0)
        circ.cy(0, 8)
        circ.x(0)
        circ.cx(0, 8)
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def make_circ_rho(thetas, theta_i):
    circ = psi_partial_thetai(thetas, theta_i, "imag")
    circ.cswap(0, 3, 7)
    circ.cswap(0, 4, 8)
    circ.h(0)
    return circ


def measure_qc_statevec(qc):
    backend = Aer.get_backend('statevector_simulator')
    job_result = execute(qc, backend).result().get_statevector(qc)
    probs = Statevector(job_result).probabilities(qargs=[0])
    return probs[0] - probs[1]


def measure_z0(thetas):
    circ = QuantumCircuit(4)
    circ.h(0)
    circ.h(1)
    apply_dic = {0: [circ.crx, 0, 2], 1: [circ.crx, 1, 3],
                 2: [circ.rzz, 2, 3], 3: [circ.rx, 2],
                 4: [circ.rx, 3], 5: [circ.rzz, 2, 3]}
    func = apply_dic[0][0]
    if len(apply_dic[0]) == 2:
        func(thetas[0], apply_dic[0][1])
    elif len(apply_dic[0]) == 3:
        func(thetas[0], apply_dic[0][1], apply_dic[0][2])
    func = apply_dic[1][0]
    if len(apply_dic[1]) == 2:
        func(thetas[1], apply_dic[1][1])
    elif len(apply_dic[1]) == 3:
        func(thetas[1], apply_dic[1][1], apply_dic[1][2])
    func = apply_dic[2][0]
    if len(apply_dic[2]) == 2:
        func(thetas[2], apply_dic[2][1])
    elif len(apply_dic[2]) == 3:
        func(thetas[2], apply_dic[2][1], apply_dic[2][2])
    func = apply_dic[3][0]
    if len(apply_dic[3]) == 2:
        func(thetas[3], apply_dic[3][1])
    elif len(apply_dic[3]) == 3:
        func(thetas[3], apply_dic[3][1], apply_dic[3][2])
    func = apply_dic[4][0]
    if len(apply_dic[4]) == 2:
        func(thetas[4], apply_dic[4][1])
    elif len(apply_dic[4]) == 3:
        func(thetas[4], apply_dic[4][1], apply_dic[4][2])
    func = apply_dic[5][0]
    if len(apply_dic[5]) == 2:
        func(thetas[5], apply_dic[5][1])
    elif len(apply_dic[5]) == 3:
        func(thetas[5], apply_dic[5][1], apply_dic[5][2])
    backend = Aer.get_backend('statevector_simulator')
    job_result = execute(circ, backend).result().get_statevector(circ)
    probs = Statevector(job_result).probabilities(qargs=[2])
    return probs[0] - probs[1]


def get_grad(thetas):
    M = np.zeros((6, 6))
    res = []
    qc = make_circ_m(thetas, 0, 0, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 0, 0, 1)
    res.append(measure_qc_statevec(qc))
    M[0, 0] = 2 * res[0] - 2 * res[1]
    M[0, 0] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 0, 1, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 0, 1, 1)
    res.append(measure_qc_statevec(qc))
    M[0, 1] = 2 * res[0] - 2 * res[1]
    M[1, 0] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 0, 2, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 0, 2, 1)
    res.append(measure_qc_statevec(qc))
    M[0, 2] = 2 * res[0] - 2 * res[1]
    M[2, 0] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 0, 3, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 0, 3, 1)
    res.append(measure_qc_statevec(qc))
    M[0, 3] = 2 * res[0] - 2 * res[1]
    M[3, 0] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 0, 4, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 0, 4, 1)
    res.append(measure_qc_statevec(qc))
    M[0, 4] = 2 * res[0] - 2 * res[1]
    M[4, 0] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 0, 5, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 0, 5, 1)
    res.append(measure_qc_statevec(qc))
    M[0, 5] = 2 * res[0] - 2 * res[1]
    M[5, 0] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 1, 1, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 1, 1, 1)
    res.append(measure_qc_statevec(qc))
    M[1, 1] = 2 * res[0] - 2 * res[1]
    M[1, 1] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 1, 2, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 1, 2, 1)
    res.append(measure_qc_statevec(qc))
    M[1, 2] = 2 * res[0] - 2 * res[1]
    M[2, 1] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 1, 3, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 1, 3, 1)
    res.append(measure_qc_statevec(qc))
    M[1, 3] = 2 * res[0] - 2 * res[1]
    M[3, 1] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 1, 4, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 1, 4, 1)
    res.append(measure_qc_statevec(qc))
    M[1, 4] = 2 * res[0] - 2 * res[1]
    M[4, 1] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 1, 5, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 1, 5, 1)
    res.append(measure_qc_statevec(qc))
    M[1, 5] = 2 * res[0] - 2 * res[1]
    M[5, 1] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 2, 2, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 2, 2, 1)
    res.append(measure_qc_statevec(qc))
    M[2, 2] = 2 * res[0] - 2 * res[1]
    M[2, 2] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 2, 3, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 2, 3, 1)
    res.append(measure_qc_statevec(qc))
    M[2, 3] = 2 * res[0] - 2 * res[1]
    M[3, 2] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 2, 4, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 2, 4, 1)
    res.append(measure_qc_statevec(qc))
    M[2, 4] = 2 * res[0] - 2 * res[1]
    M[4, 2] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 2, 5, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 2, 5, 1)
    res.append(measure_qc_statevec(qc))
    M[2, 5] = 2 * res[0] - 2 * res[1]
    M[5, 2] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 3, 3, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 3, 3, 1)
    res.append(measure_qc_statevec(qc))
    M[3, 3] = 2 * res[0] - 2 * res[1]
    M[3, 3] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 3, 4, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 3, 4, 1)
    res.append(measure_qc_statevec(qc))
    M[3, 4] = 2 * res[0] - 2 * res[1]
    M[4, 3] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 3, 5, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 3, 5, 1)
    res.append(measure_qc_statevec(qc))
    M[3, 5] = 2 * res[0] - 2 * res[1]
    M[5, 3] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 4, 4, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 4, 4, 1)
    res.append(measure_qc_statevec(qc))
    M[4, 4] = 2 * res[0] - 2 * res[1]
    M[4, 4] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 4, 5, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 4, 5, 1)
    res.append(measure_qc_statevec(qc))
    M[4, 5] = 2 * res[0] - 2 * res[1]
    M[5, 4] = 2 * res[0] - 2 * res[1]
    res = []
    qc = make_circ_m(thetas, 5, 5, 0)
    res.append(measure_qc_statevec(qc))
    qc = make_circ_m(thetas, 5, 5, 1)
    res.append(measure_qc_statevec(qc))
    M[5, 5] = 2 * res[0] - 2 * res[1]
    M[5, 5] = 2 * res[0] - 2 * res[1]
    V = []
    res = 0
    coeff = 0.25 if 0 == 2 else 1
    qc = make_circ_xxzz(thetas, 0, 0, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 0, 0, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 1 == 2 else 1
    qc = make_circ_xxzz(thetas, 0, 1, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 0, 1, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 2 == 2 else 1
    qc = make_circ_xxzz(thetas, 0, 2, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 0, 2, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 0, 0, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 0, 0, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 0, 1, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 0, 1, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 0, 0)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 0, 1)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 0, 2)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 0, 3)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    coeff = (-1) ** 0
    qc = make_circ_xy(thetas, 0, 0)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 1
    qc = make_circ_xy(thetas, 0, 1)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 2
    qc = make_circ_xy(thetas, 0, 2)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 3
    qc = make_circ_xy(thetas, 0, 3)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_rho(thetas, 0)
    res = res + 2 * measure_qc_statevec(qc)
    V.append(res)
    res = 0
    coeff = 0.25 if 0 == 2 else 1
    qc = make_circ_xxzz(thetas, 1, 0, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 1, 0, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 1 == 2 else 1
    qc = make_circ_xxzz(thetas, 1, 1, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 1, 1, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 2 == 2 else 1
    qc = make_circ_xxzz(thetas, 1, 2, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 1, 2, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 1, 0, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 1, 0, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 1, 1, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 1, 1, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 1, 0)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 1, 1)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 1, 2)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 1, 3)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    coeff = (-1) ** 0
    qc = make_circ_xy(thetas, 1, 0)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 1
    qc = make_circ_xy(thetas, 1, 1)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 2
    qc = make_circ_xy(thetas, 1, 2)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 3
    qc = make_circ_xy(thetas, 1, 3)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_rho(thetas, 1)
    res = res + 2 * measure_qc_statevec(qc)
    V.append(res)
    res = 0
    coeff = 0.25 if 0 == 2 else 1
    qc = make_circ_xxzz(thetas, 2, 0, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 2, 0, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 1 == 2 else 1
    qc = make_circ_xxzz(thetas, 2, 1, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 2, 1, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 2 == 2 else 1
    qc = make_circ_xxzz(thetas, 2, 2, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 2, 2, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 2, 0, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 2, 0, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 2, 1, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 2, 1, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 2, 0)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 2, 1)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 2, 2)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 2, 3)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    coeff = (-1) ** 0
    qc = make_circ_xy(thetas, 2, 0)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 1
    qc = make_circ_xy(thetas, 2, 1)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 2
    qc = make_circ_xy(thetas, 2, 2)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 3
    qc = make_circ_xy(thetas, 2, 3)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_rho(thetas, 2)
    res = res + 2 * measure_qc_statevec(qc)
    V.append(res)
    res = 0
    coeff = 0.25 if 0 == 2 else 1
    qc = make_circ_xxzz(thetas, 3, 0, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 3, 0, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 1 == 2 else 1
    qc = make_circ_xxzz(thetas, 3, 1, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 3, 1, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 2 == 2 else 1
    qc = make_circ_xxzz(thetas, 3, 2, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 3, 2, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 3, 0, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 3, 0, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 3, 1, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 3, 1, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 3, 0)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 3, 1)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 3, 2)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 3, 3)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    coeff = (-1) ** 0
    qc = make_circ_xy(thetas, 3, 0)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 1
    qc = make_circ_xy(thetas, 3, 1)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 2
    qc = make_circ_xy(thetas, 3, 2)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 3
    qc = make_circ_xy(thetas, 3, 3)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_rho(thetas, 3)
    res = res + 2 * measure_qc_statevec(qc)
    V.append(res)
    res = 0
    coeff = 0.25 if 0 == 2 else 1
    qc = make_circ_xxzz(thetas, 4, 0, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 4, 0, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 1 == 2 else 1
    qc = make_circ_xxzz(thetas, 4, 1, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 4, 1, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 2 == 2 else 1
    qc = make_circ_xxzz(thetas, 4, 2, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 4, 2, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 4, 0, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 4, 0, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 4, 1, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 4, 1, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 4, 0)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 4, 1)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 4, 2)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 4, 3)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    coeff = (-1) ** 0
    qc = make_circ_xy(thetas, 4, 0)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 1
    qc = make_circ_xy(thetas, 4, 1)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 2
    qc = make_circ_xy(thetas, 4, 2)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 3
    qc = make_circ_xy(thetas, 4, 3)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_rho(thetas, 4)
    res = res + 2 * measure_qc_statevec(qc)
    V.append(res)
    res = 0
    coeff = 0.25 if 0 == 2 else 1
    qc = make_circ_xxzz(thetas, 5, 0, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 5, 0, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 1 == 2 else 1
    qc = make_circ_xxzz(thetas, 5, 1, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 5, 1, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    coeff = 0.25 if 2 == 2 else 1
    qc = make_circ_xxzz(thetas, 5, 2, 0)
    res = res - 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xxzz(thetas, 5, 2, 1)
    res = res + 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 5, 0, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 5, 0, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 5, 1, 0)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xy1(thetas, 5, 1, 1)
    res = res - 2 * 0.25 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 5, 0)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 5, 1)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 5, 2)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    qc = make_circ_xx(thetas, 5, 3)
    res = res - 0.25 * 2 * measure_qc_statevec(qc)
    coeff = (-1) ** 0
    qc = make_circ_xy(thetas, 5, 0)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 1
    qc = make_circ_xy(thetas, 5, 1)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 2
    qc = make_circ_xy(thetas, 5, 2)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    coeff = (-1) ** 3
    qc = make_circ_xy(thetas, 5, 3)
    res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
    qc = make_circ_rho(thetas, 5)
    res = res + 2 * measure_qc_statevec(qc)
    V.append(res)
    M = M + np.eye(6) * 1.e-8
    grad_vec = linalg.solve(M, V)
    return grad_vec
