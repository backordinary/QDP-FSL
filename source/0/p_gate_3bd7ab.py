# https://github.com/Baccios/CTC_iterator/blob/2828be62b31c36237a05c04ba9f8cf5b97a21737/ctc/gates/p_gate.py
"""
This module implements a Gate that initializes a qubit |0⟩ to sqrt(p)|0⟩ + sqrt(1-p)|1⟩
"""
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.extensions import Initialize

from math import sqrt


class PGate(Gate):
    """
    This gate initializes a qubit |0⟩ to sqrt(p)|0⟩ + sqrt(1-p)|1⟩
    """

    def __init__(self, p, label=None):
        """
        :param p: The gate output is sqrt(p)|0⟩ + sqrt(1-p)|1⟩ when the input is |0⟩
        :type p: float
        :param label: The gate label. Defaults to None
        """
        if p < 0 or p > 1:
            raise ValueError("p must be between zero and one.")
        self._p = p
        super().__init__('P gate', 1, [], label)

    def _define(self):
        """
        define gate behavior
        """

        qubit = QuantumRegister(1)
        p_circuit = QuantumCircuit(qubit)

        state_p = [sqrt(self._p), sqrt(1-self._p)]
        init_p_gate = Initialize(state_p)
        p_circuit.append(init_p_gate, [qubit[0]])

        self.definition = p_circuit
