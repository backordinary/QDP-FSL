# https://github.com/daehiff/gate-optimisation/blob/1c3aad5de730d35726620b6963ef729fc785c460/tests/test_utils.py
import random
import unittest
import pyzx as zx
from qiskit import QuantumCircuit
from qiskit.circuit.library import *

from utils import *


class TestUtils(unittest.TestCase):

    def test_is_clifford(self):
        """
        Test the is_clifford method against Clifford+T group plus CZ
        :return:
        """
        self.assertTrue(is_clifford(XGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(YGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(ZGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(HGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(CXGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(CZGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(SGate()), "Gate was incorrectly classified")
        self.assertTrue(is_clifford(SdgGate()), "Gate was incorrectly classified")
        self.assertFalse(is_clifford(TGate()), "Gate was incorrectly classified")
        self.assertFalse(is_clifford(TdgGate()), "Gate was incorrectly classified")

    def test_is_clifford_t_generation(self):
        """
        All random circuits, that are created are clifford+T
        => Create 5 random circs
        Also create 5 counter examples with cz gate
        :return:
        """
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        self.assertTrue(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        self.assertTrue(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        self.assertTrue(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        self.assertTrue(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        self.assertTrue(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        qc.cz(0, 1)
        self.assertFalse(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        qc.cz(0, 1)
        self.assertFalse(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        qc.cz(0, 1)
        self.assertFalse(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        qc.cz(0, 1)
        self.assertFalse(is_clifford_t(qc))
        qc = generate_random_circuit(random.randint(2, 8), random.randint(10, 20))
        qc.cz(0, 1)
        self.assertFalse(is_clifford_t(qc))

    def test_circuit_stats(self):
        """
        Given, this circuit:
          ┌───┐ ┌───┐┌─────┐
    q_0: ─┤ H ├─┤ T ├┤ TDG ├
          ├───┤ └───┘└─────┘
    q_1: ─┤ X ├─────────────
          ├───┤
    q_2: ─┤ Y ├─────────────
          ├───┤
    q_3: ─┤ Z ├─────────────
          └───┘
    q_4: ───■───────────────
          ┌─┴─┐
    q_5: ─┤ X ├─────────────
          ├───┤
    q_6: ─┤ S ├─────────────
         ┌┴───┴┐
    q_7: ┤ SDG ├────────────
         └─────┘
    One would expect
    Is Clifford+T: True
    Num Cliffords. 7
    Num Non-Cliffords: 2
    1-Qubit: 8
    2-Qubit: 1
    Overlall 9
    depth: 3 (topline)
        :return:
        """
        qc = QuantumCircuit(8)
        qc.h(0)
        qc.x(1)
        qc.y(2)
        qc.z(3)
        qc.cx(4, 5)
        qc.s(6)
        qc.sdg(7)
        qc.t(0)
        qc.tdg(0)
        print(qc)
        clifford_count, non_clifford, two_qubit_count, overall_gate_count, depth = get_circuit_stats(qc)
        self.assertEqual(clifford_count, 7)
        self.assertEqual(non_clifford, 2)
        self.assertEqual(two_qubit_count, 1)
        self.assertEqual(overall_gate_count, 9)
        self.assertEqual(depth, 3)

        qc1 = generate_random_circuit(8, 20)
        clifford_count, non_clifford, two_qubit_count, overall_gate_count, depth = get_circuit_stats(qc1)
        self.assertEqual(clifford_count + non_clifford, overall_gate_count)

    def test_formatting(self):
        """
        Test the self-written qc formatter
        :return:
        """
        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc = zx_circuit_to_qiskit_circuit(qc_)
        qc = qiskit_circuit_to_zx_circuit(qc)
        self.assertTrue(qc_.verify_equality(qc))

        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc_str = to_qc_format(qc)
        qc__ = zx.Circuit.from_qc(qc_str)
        self.assertTrue(qc_.verify_equality(qc__))
        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc = zx_circuit_to_qiskit_circuit(qc_)
        qc = qiskit_circuit_to_zx_circuit(qc)
        self.assertTrue(qc_.verify_equality(qc))

        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc_str = to_qc_format(qc)
        qc__ = zx.Circuit.from_qc(qc_str)
        self.assertTrue(qc_.verify_equality(qc__))
        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc = zx_circuit_to_qiskit_circuit(qc_)
        qc = qiskit_circuit_to_zx_circuit(qc)
        self.assertTrue(qc_.verify_equality(qc))

        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc_str = to_qc_format(qc)
        qc__ = zx.Circuit.from_qc(qc_str)
        self.assertTrue(qc_.verify_equality(qc__))
        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc = zx_circuit_to_qiskit_circuit(qc_)
        qc = qiskit_circuit_to_zx_circuit(qc)
        self.assertTrue(qc_.verify_equality(qc))

        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc_str = to_qc_format(qc)
        qc__ = zx.Circuit.from_qc(qc_str)
        self.assertTrue(qc_.verify_equality(qc__))
        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc = zx_circuit_to_qiskit_circuit(qc_)
        qc = qiskit_circuit_to_zx_circuit(qc)
        self.assertTrue(qc_.verify_equality(qc))

        qc = generate_random_circuit(8, 20)
        qc_ = qiskit_circuit_to_zx_circuit(qc)
        qc_str = to_qc_format(qc)
        qc__ = zx.Circuit.from_qc(qc_str)
        self.assertTrue(qc_.verify_equality(qc__))
