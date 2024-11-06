# https://github.com/prat212/MANSO/blob/8785d22e4146bd5603a3365b6d44d402458acec0/objectives.py
import numpy as np
import scipy as sp
import random
import math
import csv
import ast
import os
import argparse
import sys
import pickle
from itertools import product
from functools import partial
from qiskit import QuantumCircuit, execute, Aer
import PointsGL


## QAOA By Jeff
def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2 * gamma, q2)
    qc.cx(q1, q2)


def get_cost_operator_circuit(elist, nnodes, gamma):
    qc = QuantumCircuit(nnodes, nnodes)
    for u, v in elist:
        append_zz_term(qc, u, v, gamma)  # here can instead append the optimized C_phi gate
    return qc


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def get_mixer_operator_circuit(nqubits, beta):
    qc = QuantumCircuit(nqubits, nnodes)
    for n in range(nqubits):
        append_x_term(qc, n, beta)
    return qc


def get_qaoa_circuit(elist, nnodes, beta, gamma, measure=True):
    assert len(beta) == len(gamma)
    p = len(beta)  # infering number of QAOA steps from the parameters passed
    qc = QuantumCircuit(nnodes, nnodes)
    # first, apply a layer of Hadamards
    qc.h(range(nnodes))
    # second, apply p alternating operators
    for i in range(p):
        qc += get_cost_operator_circuit(elist, nnodes, gamma[i])
        qc += get_mixer_operator_circuit(nnodes, beta[i])
    if measure:
        qc.barrier(range(nnodes))
        qc.measure(range(nnodes), range(nnodes))
    return qc


#
# Maxcut objective
#


def maxcut_obj(x, elist):
    cut = 0
    for i, j in elist:
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut


def brute_force(obj_f, num_variables):
    best_cost_brute = 0
    for b in range(2 ** num_variables):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(num_variables)))]
        try:
            cost = obj_f(x)
        except TypeError:
            cost = obj_f(np.array(x))
        if cost < best_cost_brute:
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute


#
# Utils for statevector and searching for optimal parameters
#


def state_num2str(basis_state_as_num, nqubits):
    return "{0:b}".format(basis_state_as_num).zfill(nqubits)


def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)


def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2 ** nqubits, dtype=complex)
    for basis_state in range(2 ** nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state


def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real ** 2 + val.imag ** 2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def invert_counts(counts):
    return {k[::-1]: v for k, v in counts.items()}


def compute_maxcut_energy(counts, elist):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, elist)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts


def compute_maxcut_energy_sv(sv, elist):
    """Compute objective from statevector
    For large number of qubits, this is slow.
    """
    counts = state_to_ampl_counts(sv)
    return sum(maxcut_obj(np.array([int(x) for x in k]), elist) * (np.abs(v) ** 2) for k, v in counts.items())


def get_black_box_objective_sv(elist, nnodes, p):
    backend = Aer.get_backend("statevector_simulator")

    def f(theta):
        # let's assume first half is betas, second half is gammas
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(elist, nnodes, beta, gamma, measure=False)
        sv = execute(qc, backend).result().get_statevector()
        # return the energy
        return compute_maxcut_energy_sv(get_adjusted_state(sv), elist)

    return f


def get_black_box_objective(elist, nnodes, p):
    backend = Aer.get_backend("qasm_simulator")

    def f(theta):
        # let's assume first half is betas, second half is gammas
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(elist, nnodes, beta, gamma)
        counts = execute(qc, backend).result().get_counts()
        # return the energy
        return compute_maxcut_energy(invert_counts(counts), elist)

    return f


elist = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 7], [5, 8], [6, 8], [6, 9], [7, 9]]
nnodes = 10

obj_f = partial(maxcut_obj, elist=elist)
true_opt = brute_force(obj_f, nnodes)
# print(f"True optimum: ", true_opt)

# p is the number of QAOA alternating operators
p = 5
obj_sv = get_black_box_objective_sv(elist, nnodes, p)
obj = get_black_box_objective(elist, nnodes, p)


minim = 100
iterate = 0


def Qtum(x):
    val = obj(np.hstack(x))
    PointsGL.Points.append([val] + x.tolist())
    return val


def Qtum_D(x):
    val = obj_sv(np.hstack(x))
    return val


def Branin_D(x, a=1, b=5.1 / (4 * np.pi ** 2), c=5.0 / np.pi, r=6, s=10, t=1.0 / (8 * np.pi)):
    # random.seed( 30 )
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: <http://www.sfu.ca/~ssurjano/branin.html>

    """
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


def Shekel_D(xx, d):
    """
    https://www.sfu.ca/~ssurjano/shekel.html
    """
    a = 2 ** (-d + 4)
    m = 10
    fac = 1
    b = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    Ct = np.array([[4] * d, [1] * d, [8] * d, [6] * d, [3, 7] * int(d / 2), [2, 9] * int(d / 2), [5, 3] * int(d / 2), [8, 1] * int(d / 2), [6, 2] * int(d / 2), [7, 3.6] * int(d / 2)])
    xxmat = [xx,] * m
    inner = np.sum(a * (xxmat - Ct) ** 2, 1)
    outer = -np.sum((inner + fac * b) ** -1)
    return outer
