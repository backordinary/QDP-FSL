# https://github.com/carstenblank/Quantum-Adder-Draper/blob/930adc6c3917334c156b84d94aa6470bc5b9932b/extensions/extended_simplified/cr.py
from extensions.qft_extended.gencU import GencUGate
from qiskit import QuantumCircuit, CompositeGate


class CRGate(GencUGate):

    def __init__(self, λ, target, control, circ, cnot_back: bool = False):
        α = λ / 2
        β = λ / 2
        γ = λ / 2
        δ = λ / 2
        super().__init__(α, β, γ, δ, target, control, circ, cnot_back)

def cr(self,  λ: float, tgt, ctl, cnot_back: bool = False):
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CRGate(λ, tgt, ctl, self, cnot_back))

QuantumCircuit.cr = cr
CompositeGate.cr = cr