# https://github.com/panbae59/smooth-optimal-control-2/blob/c90a7f97d24660b54a815726b938194adf376414/optimal_control/qiskit_control.py
import numpy as np
from numpy.lib.npyio import save
from optimal_control.optimal_control import OptimalControl
from util.utils import save_value, load_value, get_closest_multiple_of_16
from qiskit.pulse import library as pulse_lib
from qiskit.pulse import Play
from util.sweep import Sweep
from qiskit import QuantumCircuit
from qiskit import transpile, schedule as build_schedule
from qiskit.circuit.library import UGate

class QiskitControl(OptimalControl):
    def __init__(self, Q_setup):
        super().__init__()
        self.Q_setup = Q_setup
        self.set_gates()

    def set_gates(self, ):
        backend = self.Q_setup.backend

        # x
        x_circ = QuantumCircuit(1, 1)
        x_circ.x(0)
        x_circ = transpile(x_circ, backend)
        x_schedule = build_schedule(x_circ, backend)

        # y
        y_circ = QuantumCircuit(1, 1)
        y_circ.y(0)
        y_circ = transpile(y_circ, backend)
        y_schedule = build_schedule(y_circ, backend)

        # y halfpi
        xh_circ = QuantumCircuit(1, 1)
        xh_circ.h(0)
        xh_circ.x(0)
        backend = backend
        xh_circ = transpile(xh_circ, backend)
        xh_schedule = build_schedule(xh_circ, backend)

        #minus y halfpi
        zh_circ = QuantumCircuit(1, 1)
        zh_circ.h(0)
        zh_circ.z(0)
        zh_circ = transpile(zh_circ, backend)
        zh_schedule = build_schedule(zh_circ, backend)

        # minus x_half pi
        shs_circ = QuantumCircuit(1, 1)
        shs_circ.s(0)
        shs_circ.h(0)
        shs_circ.s(0)
        shs_circ = transpile(shs_circ, backend)
        shs_schedule = build_schedule(shs_circ, backend)
        # x_half pi
        xshs_circ = QuantumCircuit(1, 1)
        xshs_circ.s(0)
        xshs_circ.h(0)
        xshs_circ.s(0)
        xshs_circ.x(0)
        xshs_circ = transpile(xshs_circ, backend)
        xshs_schedule = build_schedule(xshs_circ, backend)

        self.gate['X'] = x_schedule
        self.gate['Y'] = y_schedule
        self.gate['Y/2'] = xh_schedule
        self.gate['X/2'] = xshs_schedule
        self.gate['-X/2'] = shs_schedule
        self.gate['-Y/2'] = zh_schedule
