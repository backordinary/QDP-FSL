# https://github.com/epiqc/PartialCompilation/blob/50d80f56efdf754e40a0b1dd00404788a03fdf3d/qiskit-terra/test/python/transpiler/test_trivial_layout.py
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the TrivialLayout pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.mapper import CouplingMap
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeTenerife, FakeRueschlikon


class TestDenseLayout(QiskitTestCase):
    """Tests the TrivialLayout pass"""

    def setUp(self):
        self.cmap5 = FakeTenerife().configuration().coupling_map
        self.cmap16 = FakeRueschlikon().configuration().coupling_map

    def test_3q_circuit_5q_coupling(self):
        """Test finds trivial layout for 3q circuit on 5q device.
        """
        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = TrivialLayout(CouplingMap(self.cmap5))
        pass_.run(dag)
        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr[0]], 0)
        self.assertEqual(layout[qr[1]], 1)
        self.assertEqual(layout[qr[2]], 2)

    def test_9q_circuit_16q_coupling(self):
        """Test finds trivial layout for 9q circuit with 2 registers on 16q device.
        """
        qr0 = QuantumRegister(4, 'q0')
        qr1 = QuantumRegister(5, 'q1')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr0, qr1, cr)
        circuit.cx(qr0[1], qr0[2])
        circuit.cx(qr0[0], qr1[3])
        circuit.cx(qr1[4], qr0[2])
        circuit.measure(qr1[1], cr[0])
        circuit.measure(qr0[2], cr[1])

        dag = circuit_to_dag(circuit)
        pass_ = TrivialLayout(CouplingMap(self.cmap16))
        pass_.run(dag)
        layout = pass_.property_set['layout']
        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr0[2]], 2)
        self.assertEqual(layout[qr0[3]], 3)
        self.assertEqual(layout[qr1[0]], 0+4)
        self.assertEqual(layout[qr1[1]], 1+4)
        self.assertEqual(layout[qr1[2]], 2+4)
        self.assertEqual(layout[qr1[3]], 3+4)
        self.assertEqual(layout[qr1[4]], 4+4)

    def test_raises_wider_circuit(self):
        """Test error is raised if the circuit is wider than coupling map.
        """
        qr0 = QuantumRegister(3, 'q0')
        qr1 = QuantumRegister(3, 'q1')
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0, qr1)

        dag = circuit_to_dag(circuit)
        with self.assertRaises(TranspilerError):
            pass_ = TrivialLayout(CouplingMap(self.cmap5))
            pass_.run(dag)


if __name__ == '__main__':
    unittest.main()
