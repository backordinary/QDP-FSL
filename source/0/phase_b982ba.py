# https://github.com/carstenblank/Quantum-Adder-Draper/blob/930adc6c3917334c156b84d94aa6470bc5b9932b/extensions/qft_extended/phase.py
from qiskit import QuantumCircuit, QuantumRegister, CompositeGate, InstructionSet


class PhaseGate(CompositeGate):

    def __init__(self, θ: float, q: QuantumRegister, circ: QuantumCircuit):
        super().__init__("phase", [θ], [q], circ)
        self.u1(θ, q)
        self.x(q)
        self.u1(θ, q)
        self.x(q)


def phase(self,θ: float, q):
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.phase(θ, (q, j)))
        return gs
    else:
        self._check_qubit(q)
        return self._attach(PhaseGate(θ, q, self))


QuantumCircuit.phase = phase
CompositeGate.phase = phase