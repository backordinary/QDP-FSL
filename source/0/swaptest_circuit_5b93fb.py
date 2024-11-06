# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/swaptest_circuit.py
from qiskit import QuantumCircuit
from qiskit.result.result import Result, ExperimentResult

import numpy as np

from typing import *

def create_swaptest_toffoli_circuit():
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.h(1)

    qc.toffoli(0, 1, 2)
    qc.toffoli(0, 2, 1)
    qc.toffoli(0, 1, 2)

    qc.h(0)

    qc.measure(0, 0)

    return qc


def add_toffoli(qc, c1, c2, t):
    qc.barrier()
    qc.h(t)
    qc.cx(c2, t)
    qc.tdg(t)
    qc.cx(c1, t)
    qc.t(t)
    qc.cx(c2, t)
    qc.tdg(t)
    qc.cx(c1, t)
    qc.t(c2)
    qc.t(t)
    qc.cx(c1, c2)
    qc.h(t)
    qc.t(c1)
    qc.tdg(c2)
    qc.cx(c1, c2)


def add_swap(qc, q1, q2, p):
    add_toffoli(qc, p, q1, q2)
    add_toffoli(qc, p, q2, q1)
    add_toffoli(qc, p, q1, q2)


def create_swap_circuit(state1_qubits, state2_qubits, probe, angle: float = np.pi/2, n_qubits: int = 5):
    qc = QuantumCircuit(n_qubits, 1)

    qc.h(probe)

    qc.u(angle, 0, 0, state1_qubits[0])
    for i in range(1, len(state1_qubits)):
        qc.cx(state1_qubits[0], state1_qubits[i])

    for i in range(len(state1_qubits)):
        add_swap(qc, state1_qubits[i], state2_qubits[i], probe)

    qc.barrier()

    qc.h(probe)

    qc.measure(probe, 0)

    return qc


def swaptest_exp_val_func(results: List[ExperimentResult], filter=None):
    exp_vals, variances = np.zeros(np.shape(results)), np.zeros(np.shape(results))
    for i, res in enumerate(results):
        shots = res.shots
        counts = res.data.counts
        exp_val, eigenval = 0, 0
        for key in counts.keys():
            if key == "0x0":
                eigenval = +1
            elif key == "0x1":
                eigenval = -1
            else:
                raise Exception("Did not recognise key: {:}".format(key))
            exp_val += eigenval*counts[key] / shots
        exp_vals[i] = exp_val
        variances[i] = 1 - exp_val**2
    return exp_vals, variances


def swaptest_exp_val_func_randompauli(results: List[ExperimentResult], filter=None):
    exp_val, _ = swaptest_exp_val_func(results, filter)
    return exp_val

qc_swaptest = create_swap_circuit([1],[2],0, np.pi/2, 3)

if __name__ == "__main__":
    qc = QuantumCircuit(2,1)
    qc.h(0)
    qc.measure(0,0)

    from qiskit import Aer, execute

    sim_backend = Aer.set_up_backend("qasm_simulator")

    job = execute(qc, sim_backend, shots=1024)

    res = job.result().results

    print(res)
    print(res[0].data)
    print(res[0].data.counts)

    print(swaptest_exp_val_func(res[0]))