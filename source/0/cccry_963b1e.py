# https://github.com/hoang-ho/Quantum_Computing/blob/d494582589439ef0bf43ffcde27b711e2cfba2f2/quantum_gates/cccry.py
from qiskit import CompositeGate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class CccryGate(CompositeGate):
    """Controlled^3 R_y gate."""

    def __init__(self, theta, ctl1, ctl2, ctl3, tgt, circ=None):
        """Create new Controlled^3 R_y gate from ctl1, ctl2, ctl3 to tgt with angle theta."""
        """We will apply R_y gate of ctl1*ctl2*ctl3 = 1, i.e., all of them be in the state 1
        know that ctl1 + ctl2 + ctl3 - (ctl1 XOR ctl2) - (ctl1 XOR ctl3) - (ctl2 XOR ctl3) + (ctl1 XOR ctl2 XOR ctl3) = 4*ctl1*ctl2*ctl3
        """
        super().__init__("cry", [theta], [ctl1, ctl2, ctl3, tgt], circ)
        # Check if both ctl1 and ctl2 are in state 1
        self.cry(theta / 4, ctl1, tgt)
        self.cx(ctl1, ctl2)
        self.cry(-theta / 4, ctl2, tgt)
        self.cx(ctl1, ctl2)
        self.cry(theta / 4, ctl2, tgt)
        # this is effectively the Controlled^2 R_y gate from ctl1, ctl2 to tgt

        self.cx(ctl2, ctl3)
        self.cry(-theta / 4, ctl3, tgt)
        self.cx(ctl1, ctl3)
        self.cry(theta / 4, ctl3, tgt)

        self.cx(ctl2, ctl3)
        self.cry(theta / 4, ctl3, tgt)
        self.cx(ctl1, ctl3)
        self.cry(theta / 4, ctl3, tgt)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cry(self.param[0], self.arg[0], self.arg[1], self.arg[2], self.arg[3]))


def cccry(self, theta, ctl1, ctl2, ctl3, tgt):
    """Apply Controlled^3 R_y from ctl1, ctl2, ctl3 to tgt with angle theta."""

    if isinstance(ctl1, QuantumRegister) and \
            isinstance(ctl2, QuantumRegister) and \
            isinstance(ctl3, QuantumRegister) and \
            len(ctl1) == len(tgt) and len(ctl2) == len(tgt) and len(ctl3) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cry(theta, (ctl1, i), (ctl2, i), (ctl3, i), (tgt, i)))
        return instructions

    self._check_qubit(ctl1)
    self._check_qubit(ctl2)
    self._check_qubit(ctl3)

    self._check_qubit(tgt)
    self._check_dups([ctl1, ctl2, ctl3, tgt])

    return self._attach(CccryGate(theta, ctl1, ctl2, ctl3, tgt, self))


QuantumCircuit.cccry = cccry
CompositeGate.cccry = cccry
