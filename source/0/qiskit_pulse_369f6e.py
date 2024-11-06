# https://github.com/dberga/quantum-experiments/blob/4f13e993fa4382496b7be5caf35dd4898b8d271e/paper_repos/torchquantum/torchquantum/plugins/qiskit_pulse.py
import torch
import torchquantum as tq
from qiskit import pulse, QuantumCircuit
from qiskit.pulse import library
from qiskit.test.mock import FakeQuito, FakeArmonk, FakeBogota
from qiskit.compiler import assemble, schedule
from .qiskit_macros import IBMQ_PNAMES

def circ2pulse(circuits, name):
    if name in IBMQ_PNAMES:
        backend = name()
        with pulse.build(backend) as pulse_tq:
            qc = circuits
            qc.measure_all()
            pulse.call(qc)
        pulse_tq.draw()