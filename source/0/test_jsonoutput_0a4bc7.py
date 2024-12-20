# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/test/python/test_jsonoutput.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring

"""Quick program to test json backend
"""
import unittest

from qiskit import qasm, unroll, QuantumProgram

from .common import QiskitTestCase, Path


class TestJsonOutput(QiskitTestCase):
    """Test Json output.

    This is mostly covered in test_quantumprogram.py but will leave
    here for convenience.
    """
    def setUp(self):
        self.QASM_FILE_PATH = self._get_resource_path(
            'qasm/entangled_registers.qasm', Path.EXAMPLES)

    def test_json_output(self):
        qp = QuantumProgram()
        qp.load_qasm_file(self.QASM_FILE_PATH, name="example")

        basis_gates = []  # unroll to base gates, change to test
        unroller = unroll.Unroller(qasm.Qasm(data=qp.get_qasm("example")).parse(),
                                   unroll.JsonBackend(basis_gates))
        circuit = unroller.execute()
        self.log.info('test_json_ouptut: %s', circuit)


if __name__ == '__main__':
    unittest.main()
