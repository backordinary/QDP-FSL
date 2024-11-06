# https://github.com/ChitambarLab/QiskitSummerJam2020/blob/cff5abb69997759f043dd1ca690735718c5c66bc/tests/incompatible_measurement.py
import unittest
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import IBMQ

from device_independent_test import incompatible_measurement
from device_independent_test import quantum_communicator

class module_test_cases(unittest.TestCase):
	def test_run_test(self):
		provider = IBMQ.load_account()
		backend = provider.get_backend('ibmq_qasm_simulator')
		communicator = quantum_communicator.LocalDispatcher([backend])

		test = incompatible_measurement.run_test(communicator, 0.1, 1000)

		self.assertTrue(test[0])

	def test_conditional_probs(self):
		shots = 1000
		counts = {
			'1011': 78, '0000': 11, '0001': 103, '1001': 523, '1111': 17,
			'1100': 14, '0101': 19, '1010': 16, '0111': 3, '1101': 103,
 			'0110': 1, '1110': 5, '1000': 91, '0011': 16
		}

		test_probs = incompatible_measurement.conditional_probs(counts, shots)
		match_probs = [[0.138,0.864,0.838,0.153],[0.862,0.136,0.162,0.847]]

		self.assertTrue(np.alltrue(test_probs == match_probs))

		shots = 10
		counts = {
			'1000': 1, '0100': 2, '0010': 3, '0001': 4
		}

		test_probs = incompatible_measurement.conditional_probs(counts, shots)
		match_probs = [[.6,.7,.8,.9],[.4,.3,.2,0.1]]

		self.assertTrue(np.alltrue(test_probs == match_probs))

	def test_bell_violation(self):
		shots = 1000
		counts_y0 = {
			'1010': 515, '0000': 13, '0001': 4, '0010': 90, '0011': 16, '0100': 4, '0110': 20,
			'0111': 3, '1000': 103, '1001': 13, '1011': 87, '1100': 18, '1101': 3, '1110': 99, '1111': 12
		}

		counts_y1 = {
			'1011': 78, '0000': 11, '0001': 103, '1001': 523, '1111': 17,
			'1100': 14, '0101': 19, '1010': 16, '0111': 3, '1101': 103,
			'0110': 1, '1110': 5, '1000': 91, '0011': 16
		}

		self.assertAlmostEqual(incompatible_measurement.bell_violation(counts_y0, counts_y1, shots, shots), 6.806)

	def test_bell_score(self):
		y0_probs = np.array([[1,0,1,0],[0,1,0,1]])
		y1_probs = np.array([[0,1,1,0],[1,0,0,1]])

		bell_score = incompatible_measurement.bell_score(y0_probs, y1_probs)

		self.assertEqual(bell_score, 8)

	def test_alice_circuit(self):
		qc = incompatible_measurement.bb84_states()
		test_state = Statevector.from_instruction(qc)
		real_state = [0,0,0.5,0,0,0,0.5,0,0,0,-0.5,0,0,0,-0.5,0]
		error = abs(test_state.data-real_state)
		epsilon = 1.0E-4
		self.assertFalse(any(x>epsilon for x in error))
