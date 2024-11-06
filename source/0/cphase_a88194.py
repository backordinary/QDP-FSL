# https://github.com/carstenblank/Quantum-Adder-Draper/blob/930adc6c3917334c156b84d94aa6470bc5b9932b/extensions/qft_extended/cPhase.py
from typing import Tuple

import math

from qiskit.extensions.standard.s import SGate
from qiskit.extensions.standard.t import TGate

from extensions.qft_extended.gencU import GencUGate
from qiskit import QuantumCircuit, CompositeGate, QuantumRegister

from extensions.qft_extended.rz_ibm import RZGate_ibm


class ControlledPhase(CompositeGate):

    def __init__(self, λ: float, target: Tuple[QuantumRegister,int],
                 control: Tuple[QuantumRegister,int], circ, cnot_back: bool = False):
        super().__init__("cPhase", [λ],[target, control], circ)
        if math.fabs(λ - math.pi/2) < 1e-3:
            self.t(target)
            self._cx(control,target,cnot_back)
            super()._attach(TGate(target, circ).inverse())
            self.t(control)
            self._cx(control, target, cnot_back)
        elif math.fabs(λ - math.pi) < 1e-3:
            self.s(target)
            self._cx(control,target,cnot_back)
            super()._attach(SGate(target, circ).inverse())
            self.s(control)
            self._cx(control, target, cnot_back)
        else:
            super()._attach(RZGate_ibm(λ, target))
            self._cx(control, target, cnot_back)
            super()._attach(RZGate_ibm(-λ, target))
            super()._attach(RZGate_ibm(λ, control))
            self._cx(control, target, cnot_back)

    def _cx(self, control, target, backward):
        if backward:
            self.h(control)
            self.h(target)

        if backward:
            self.cx(target, control)
        else:
            self.cx(control, target)

        if backward:
            self.h(control)
            self.h(target)

def cphase(self,  λ: float, tgt, ctl, cnot_back: bool = False):
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(ControlledPhase(λ, tgt, ctl, self, cnot_back))

QuantumCircuit.cphase = cphase
CompositeGate.cphase = cphase