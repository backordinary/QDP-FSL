# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/test/python/test_skip_transpiler.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

import unittest

from qiskit import QuantumProgram

from .common import QiskitTestCase


class CompileSkipTranslationTest(QiskitTestCase):
    """Test compilaton with skip translation."""

    def test_simple_compile(self):
        """
        Compares with and without skip_transpiler
        """
        name = 'test_simple'
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit(name, [qr], [cr])
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.measure(qr, cr)

        rtrue = qp.compile([name], backend='local_qasm_simulator', shots=1024,
                           skip_transpiler=True)
        rfalse = qp.compile([name], backend='local_qasm_simulator', shots=1024,
                            skip_transpiler=False)
        self.assertEqual(rtrue.config, rfalse.config)
        self.assertEqual(rtrue.experiments, rfalse.experiments)

    def test_simple_execute(self):
        name = 'test_simple'
        seed = 42
        qp = QuantumProgram()
        qr = qp.create_quantum_register('qr', 2)
        cr = qp.create_classical_register('cr', 2)
        qc = qp.create_circuit(name, [qr], [cr])
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.measure(qr, cr)

        rtrue = qp.execute(name, seed=seed, skip_transpiler=True)
        rfalse = qp.execute(name, seed=seed, skip_transpiler=False)
        self.assertEqual(rtrue.get_counts(), rfalse.get_counts())


if __name__ == '__main__':
    unittest.main()
