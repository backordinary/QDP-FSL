# https://github.com/QCOL-LU/Bayesian-Error-Characterization-and-Mitigation/blob/0cbef5546bad6000150b2cd4ab7a0dac7a32fb66/Tutorial/expfuncs.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:52:17 2020

@author: Muqing Zheng
"""

import csv
import numpy as np

from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, transpile, execute, QuantumRegister
from qiskit.tools.monitor import job_monitor


def QAOAexp(backend, file_address=''):
    """
        QAOA from https://arxiv.org/abs/1804.03719

    Parameters
    ----------
    backend : IBMQBackend
        backend.
    file_address : String, optional
        address for save data. The default is ''. Ends with "/" if not empty.

    Returns
    -------
    None.

    """
    pi = np.pi
    g1 = 0.2 * pi
    g2 = 0.4 * pi
    b1 = 0.15 * pi
    b2 = 0.05 * pi

    num = 5
    QAOA = QuantumCircuit(num, num)
    QAOA.h(1)
    QAOA.h(2)
    QAOA.h(3)
    QAOA.h(4)
    QAOA.barrier()

    # k = 1
    QAOA.cx(3, 2)
    QAOA.u1(-g1, 2)
    QAOA.cx(3, 2)
    QAOA.barrier()

    QAOA.cx(4, 2)
    QAOA.u1(-g1, 2)
    QAOA.cx(4, 2)
    QAOA.barrier()

    QAOA.cx(1, 2)
    QAOA.u1(-g1, 2)
    QAOA.cx(1, 2)
    QAOA.cx(4, 3)
    QAOA.u1(-g1, 3)
    QAOA.cx(4, 3)
    QAOA.barrier()
    QAOA.u3(2 * b1, -pi / 2, pi / 2, 1)
    QAOA.u3(2 * b1, -pi / 2, pi / 2, 2)
    QAOA.u3(2 * b1, -pi / 2, pi / 2, 3)
    QAOA.u3(2 * b1, -pi / 2, pi / 2, 4)
    QAOA.barrier()

    # k = 2
    QAOA.cx(3, 2)
    QAOA.u1(-g2, 2)
    QAOA.cx(3, 2)
    QAOA.barrier()

    QAOA.cx(4, 2)
    QAOA.u1(-g2, 2)
    QAOA.cx(4, 2)
    QAOA.barrier()

    QAOA.cx(1, 2)
    QAOA.u1(-g2, 2)
    QAOA.cx(1, 2)
    QAOA.cx(4, 3)
    QAOA.u1(-g2, 3)
    QAOA.cx(4, 3)
    QAOA.barrier()
    QAOA.u3(2 * b2, -pi / 2, pi / 2, 1)
    QAOA.u3(2 * b2, -pi / 2, pi / 2, 2)
    QAOA.u3(2 * b2, -pi / 2, pi / 2, 3)
    QAOA.u3(2 * b2, -pi / 2, pi / 2, 4)
    QAOA.barrier()

    QAOA.barrier()
    QAOA.measure([1, 2, 3, 4], [1, 2, 3, 4])
    QAOA_trans = transpile(QAOA, backend, initial_layout=range(num))
    print('QAOA circuit depth is ', QAOA_trans.depth())

    # Run on simulator
    simulator = Aer.get_backend("qasm_simulator")
    simu_shots = 100000
    simulate = execute(QAOA, backend=simulator, shots=simu_shots)
    QAOA_results = simulate.result()
    with open(file_address + 'Count_QAOA_Simulator.csv', mode='w',
              newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in QAOA_results.get_counts().items():
            count_writer.writerow([key, val])

    # Run on real device
    shots = 8192
    job_exp = execute(QAOA_trans,
                      backend=backend,
                      shots=shots,
                      optimization_level=0)
    print("Job id:", job_exp.job_id())
    job_monitor(job_exp)
    exp_results = job_exp.result()
    with open(file_address + 'Count_QAOA.csv', mode='w', newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in exp_results.get_counts().items():
            count_writer.writerow([key, val])


def Groverexp(backend, file_address=''):
    """
        Gorver's search from https://arxiv.org/abs/1804.03719

    Parameters
    ----------
    backend : IBMQBackend
        backend.
    file_address : String, optional
        address for save data. The default is ''. Ends with "/" if not empty.

    Returns
    -------
    None.

    """
    num = 3
    Grover = QuantumCircuit(num, num)

    Grover.x(0)
    Grover.h(1)
    Grover.h(2)
    Grover.barrier()

    Grover.h(0)
    Grover.barrier()

    Grover.h(0)

    Grover.cx(1, 0)
    Grover.tdg(0)
    Grover.cx(2, 0)
    Grover.t(0)

    Grover.cx(1, 0)
    Grover.tdg(0)
    Grover.cx(2, 0)
    Grover.barrier()
    Grover.t(0)
    Grover.tdg(1)
    Grover.barrier()

    Grover.h(0)
    Grover.cx(2, 1)
    Grover.tdg(1)
    Grover.cx(2, 1)
    Grover.s(1)
    Grover.t(2)
    Grover.barrier()

    Grover.h(1)
    Grover.h(2)
    Grover.barrier()
    Grover.x(1)
    Grover.x(2)
    Grover.barrier()
    Grover.h(1)
    Grover.cx(2, 1)
    Grover.h(1)
    Grover.x(2)
    Grover.barrier()
    Grover.x(1)
    Grover.h(2)
    Grover.barrier()
    Grover.h(1)

    Grover.barrier()
    Grover.measure([1, 2], [1, 2])
    Grover_trans = transpile(Grover, backend, initial_layout=[0, 1, 2])
    print('Grover circuit depth is ', Grover_trans.depth())

    # Run on real device
    shots = 8192
    job_exp = execute(Grover_trans,
                      backend=backend,
                      shots=shots,
                      optimization_level=0)
    print("Job id:", job_exp.job_id())
    job_monitor(job_exp)
    exp_results = job_exp.result()
    with open(file_address + 'Count_Grover.csv', mode='w', newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in exp_results.get_counts().items():
            count_writer.writerow([key, val])
