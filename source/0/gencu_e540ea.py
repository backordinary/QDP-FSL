# https://github.com/carstenblank/Quantum-Adder-Draper/blob/930adc6c3917334c156b84d94aa6470bc5b9932b/extensions/extended_simplified/gencU.py
from typing import Tuple

import math
from qiskit import CompositeGate, QuantumCircuit, QuantumRegister, InstructionSet


class GencUGate(CompositeGate):

    def __init__(self, α: float, β: float, γ: float, δ: float, target: Tuple[QuantumRegister,int],
                 control: Tuple[QuantumRegister,int], circ, cnot_back: bool = False):
        super().__init__("gencU", [α, β, γ, δ], [target, control], circ)
        # self.comment(["BEGIN gencU(α=%.4f,β=%.4f,γ=%.4f,δ=%.4f) %s[%d],%s[%d]" %
        #               (α,β,γ,δ,target[0].name,target[1], control[0].name, control[1])])
        self._c(target)
        self._cx(control, target, cnot_back)
        self._b(target)
        self._cx(control, target, cnot_back)
        self._a(target)
        self.u1(α, control)
        #self.comment(["# gates: %d" % len(self.data), "END gencU(...)"])

    def _cx(self, control, target, backward):
        #self.comment(["BEGIN cnot"])
        if backward:
            self.u3(math.pi/2,0.0,0.0,control) # == self.h(control)
            self.u3(math.pi / 2, 0.0, 0.0, target)  # == self.h(target)

        if backward:
            self.cx(target, control)
        else:
            self.cx(control, target)

        if backward:
            self.u3(math.pi / 2, 0.0, 0.0, control)  # == self.h(control)
            self.u3(math.pi / 2, 0.0, 0.0, target)  # == self.h(target)
        #self.comment(["END cnot"])

    def _a(self, target):
        #self.comment(["BEGIN A"])
        β = self.param[1]
        γ = self.param[2]
        self.ry_ibm(γ / 2, target)
        self.rz_ibm(β, target)
        #self.comment(["END A"])

    def _b(self, target):
        #self.comment(["BEGIN B"])
        β = self.param[1]
        γ = self.param[2]
        δ = self.param[3]
        self.rz_ibm(-(δ + β) / 2, target)
        self.ry_ibm(-γ / 2, target)
        #self.comment(["END B"])

    def _c(self, target):
        #self.comment(["BEGIN C"])
        β = self.param[1]
        δ = self.param[3]
        self.rz_ibm((δ - β) / 2, target)
        #self.comment(["BEGIN C"])


def gencU(self,  α: float, β: float, γ: float, δ: float, tgt, ctl, cnot_back: bool = False):
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(GencUGate(α, β, γ, δ, tgt, ctl, self, cnot_back))

QuantumCircuit.gencU = gencU
CompositeGate.gencU = gencU