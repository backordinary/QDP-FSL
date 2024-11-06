# https://github.com/nahumsa/volta/blob/196947ed3ae596c6d94324b7fdcd4d2dc18fa1a3/tests/test_ssvqe.py
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import unittest

import qiskit
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import Z, I


from volta.ssvqe import SSVQE
from volta.utils import classical_solver


class TestSSVQD(unittest.TestCase):
    def setUp(self):

        optimizer = qiskit.algorithms.optimizers.COBYLA()

        backend = QuantumInstance(
            backend=Aer.get_backend("qasm_simulator"), shots=10000
        )

        hamiltonian = 1 / 2 * (Z ^ I) + 1 / 2 * (Z ^ Z)
        ansatz = TwoLocal(hamiltonian.num_qubits, ["ry", "rz"], "cx", reps=2)

        self.Algo = SSVQE(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            optimizer=optimizer,
            n_excited=2,
            backend=backend,
        )

        self.energy = []
        self.energy.append(self.Algo.run(index=0)[0])
        self.energy.append(self.Algo.run(index=1)[0])
        self.eigenvalues, _ = classical_solver(hamiltonian)

    def test_energies(self):
        want = self.eigenvalues[0]
        got = self.energy[0]
        decimalPlace = 1
        message = "SSVQE not working for the ground state of 1/2*((Z^I) + (Z^Z))"
        self.assertAlmostEqual(want, got, decimalPlace, message)

        decimalPlace = 1
        want = self.eigenvalues[1]
        got = self.energy[1]
        message = "SSVQE not working for the first excited state of 1/2*((Z^I) + (Z^Z))"
        self.assertAlmostEqual(want, got, decimalPlace, message)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
