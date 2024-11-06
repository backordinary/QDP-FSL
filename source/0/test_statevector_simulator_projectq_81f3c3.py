# https://github.com/yaelbh/qiskit-projectq-provider/blob/7b87bd750e33945362c63445ef7b8aae6ffc0497/test/test_statevector_simulator_projectq.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except

from test.common import QiskitProjectQTestCase

import unittest
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_addon_projectq import ProjectQProvider


class StatevectorSimulatorProjectQTest(QiskitProjectQTestCase):
    """Test ProjectQ C++ statevector simulator."""

    def setUp(self):
        ProjectQ = ProjectQProvider()
        self.projectq_sim = ProjectQ.get_backend('projectq_statevector_simulator')

    def test_sv_simulator_projectq(self):
        """Test final state vector for single circuit run."""

        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        result = execute(qc, backend=self.projectq_sim).result()
        self.assertEqual(result.status, 'COMPLETED')
        actual = result.get_statevector(qc)

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0]))**2, 1/2)
        self.assertAlmostEqual(abs(actual[1]), 0)
        self.assertAlmostEqual(abs(actual[2]), 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1/2)

    def test_qubit_order(self):
        """Verify Qiskit qubit ordering in state vector"""

        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.x(qr[0])

        result = execute(qc, backend=self.projectq_sim).result()
        self.assertEqual(result.status, 'COMPLETED')
        actual = result.get_statevector(qc)

        # state is |01> (up to a global phase), because qubit 0 is LSB
        self.assertAlmostEqual(abs(actual[0]), 0)
        self.assertAlmostEqual((abs(actual[1]))**2, 1)
        self.assertAlmostEqual(abs(actual[2]), 0)
        self.assertAlmostEqual(abs(actual[3]), 0)


if __name__ == '__main__':
    unittest.main()
