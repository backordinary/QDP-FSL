# https://github.com/donn/donns-qiskit-tweaks/blob/0b92061e69ab4a715f01ed8284b4d33e4dc85d6a/test/test_etof.py
from donns_qiskit_tweaks.gates import ETOF
from donns_qiskit_tweaks.matrices import Bases

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Optional

import unittest


class Test(QuantumCircuit):
    def __init__(
        self, controls: List[Optional[bool]], initial_control_states: List[int]
    ):
        n = len(controls)

        if n != len(initial_control_states):
            raise ValueError("n != len(initial_control_states)")

        controllers = QuantumRegister(n, "x")
        target = QuantumRegister(1, "xt")
        measured = ClassicalRegister(1, "measured")
        super().__init__(controllers, target, measured)

        for qubit, bra in enumerate(initial_control_states):
            if bra == 0:
                self.initialize(Bases.bras.zero, qubit)
            else:
                self.initialize(Bases.bras.one, qubit)
        self.initialize(Bases.bras.zero, n)

        self.barrier()

        self.append(ETOF(controls), [*range(0, n + 1)])

        self.barrier()
        self.measure(n, 0)


class TestETOF(unittest.TestCase):
    def test_etof(self):
        for state, expected in [
            ([0, 0, 0], {"0": 1024}),
            ([0, 0, 1], {"0": 1024}),
            ([0, 1, 0], {"1": 1024}),
            ([0, 1, 1], {"1": 1024}),
            ([1, 0, 0], {"0": 1024}),
            ([1, 0, 1], {"0": 1024}),
            ([1, 1, 0], {"0": 1024}),
            ([1, 1, 1], {"0": 1024}),
        ]:
            test = Test([False, True, None], state)
            results = test.simulate()
            counts = results.get_counts()
            for value in ["0", "1"]:
                assert expected.get(value) == counts.get(value)


if __name__ == "__main__":
    unittest.main()
