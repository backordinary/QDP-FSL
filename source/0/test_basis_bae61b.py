# https://github.com/timcp/StabRankSearcher/blob/6fdd9b0ce3b69459a713424a0a787edd10e6b0c5/test/test_basis.py
import numpy as np
import unittest
import qiskit
import netsquid.qubits.qubitapi as qapi
from netsquid.qubits.stabtools import StabRepr
from stabranksearcher.basis import Basis, get_basis_copy
from stabranksearcher.quantum_state_tools import ket_to_qstate


class TestBasis(unittest.TestCase):

    def test_to_projector_case_orthogonal_states(self):
        # Case: basis consists of orthogonal states

        # create basis
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_00 = ket_to_qstate(ket_00)
        ket_11 = np.array([[0], [0], [0], [1]])
        qstate_11 = ket_to_qstate(ket_11)
        basis = Basis(qstates=[qstate_00, qstate_11])

        # create projector and test if equals expected projector
        actual_projector = basis.to_projector()
        expected_projector = np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 1]])
        self.assertTrue(np.array_equal(actual_projector, expected_projector))

    def test_to_projector_case_nonorthogonal_states(self):
        # Case: basis consists of non-orthogonal states

        # create basis
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_00 = ket_to_qstate(ket_00)
        ket_11 = np.array([[0], [0], [0], [1]])
        qstate_11 = ket_to_qstate(ket_11)
        ket_phi_plus = np.array([[1], [0], [0], [1]]) / np.sqrt(2)
        qstate_phi_plus = ket_to_qstate(ket_phi_plus)
        basis = Basis(qstates=[qstate_00, qstate_11, qstate_phi_plus])

        # create projector and test if equals expected projector
        actual_projector = basis.to_projector()
        expected_projector = np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 1]])
        self.assertTrue(np.allclose(actual_projector, expected_projector))

    def test_qstates_living_in_subspace(self):

        s = 1.0 / np.sqrt(2)

        # create basis
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_00 = ket_to_qstate(ket_00)
        ket_11 = np.array([[0], [0], [0], [1]])
        qstate_11 = ket_to_qstate(ket_11)
        basis = Basis(qstates=[qstate_00, qstate_11])

        # test scores

        # case: |00>
        ket_00_copy = np.array([[1], [0], [0], [0]])
        qstate_00_copy = ket_to_qstate(ket_00_copy)
        self.assertTrue(basis.does_qstate_live_in_subspace(qstate=qstate_00_copy))

        # case: |11>
        ket_11_copy = np.array([[1], [0], [0], [0]])
        qstate_11_copy = ket_to_qstate(ket_11_copy)
        self.assertTrue(basis.does_qstate_live_in_subspace(qstate=qstate_11_copy))

        # case: (|00> + |11>) / sqrt(2)
        ket_phi_plus = np.array([[s], [0], [0], [s]])
        qstate_phi_plus = ket_to_qstate(ket_phi_plus)
        self.assertTrue(basis.does_qstate_live_in_subspace(qstate=qstate_phi_plus))

        # case: (|00> - |11>) / sqrt(2)
        ket_phi_minus = np.array([[s], [0], [0], [s]])
        qstate_phi_minus = ket_to_qstate(ket_phi_minus)
        self.assertTrue(basis.does_qstate_live_in_subspace(qstate=qstate_phi_minus))

    def test_score(self):

        s = 1.0 / np.sqrt(2)

        # create basis
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_00 = ket_to_qstate(ket_00)
        ket_11 = np.array([[0], [0], [0], [1]])
        qstate_11 = ket_to_qstate(ket_11)
        basis = Basis(qstates=[qstate_00, qstate_11])

        # case: (|01> + |10>) / sqrt(2)
        ket_psi_plus = np.array([[0], [s], [s], [0]])
        qstate_psi_plus = ket_to_qstate(ket_psi_plus)
        self.assertTrue(np.isclose(basis.score(qstate=qstate_psi_plus), 0.0))

        # case: (|00> + |01>) / sqrt(2)
        ket = np.array([[s], [s], [0], [0]])
        qstate = ket_to_qstate(ket)
        self.assertTrue(np.isclose(basis.score(qstate=qstate), 1 / np.sqrt(2)))

    def test_deterministically_modify(self):

        s = 1.0 / np.sqrt(2)

        # create basis
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_00 = ket_to_qstate(ket_00)
        ket_11 = np.array([[0], [0], [0], [1]])
        qstate_11 = ket_to_qstate(ket_11)
        basis = Basis(qstates=[qstate_00, qstate_11])

        # case that did not succeed, i.e. (1 + P)|state> = 0
        succeeded = basis.deterministically_modify(
            qstate_index=0,
            pauli=qiskit.quantum_info.Pauli('-ZZ'))
        self.assertFalse(succeeded)
        expected_first_ket = np.array([[1], [0], [0], [0]])
        expected_second_ket = np.array([[0], [0], [0], [1]])
        self.assertTrue(basis.qstates[0].compare(ket_to_qstate(expected_first_ket)))
        self.assertTrue(basis.qstates[1].compare(ket_to_qstate(expected_second_ket)))

        # case that succeeded
        succeeded = basis.deterministically_modify(
            qstate_index=0,
            pauli=qiskit.quantum_info.Pauli('XZ'))
        self.assertTrue(succeeded)
        expected_first_ket = np.array([[s], [0], [s], [0]])  # |00> + |10>
        expected_second_ket = np.array([[0], [0], [0], [1]])
        self.assertTrue(basis.qstates[0].compare(ket_to_qstate(expected_first_ket)))
        self.assertTrue(basis.qstates[1].compare(ket_to_qstate(expected_second_ket)))

    def test_undo_last_modification(self):

        # create and store qstates
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_00 = ket_to_qstate(ket_00)
        ket_11 = np.array([[0], [0], [0], [1]])
        qstate_11 = ket_to_qstate(ket_11)
        qstates = [qstate_00, qstate_11]

        # create basis
        basis = Basis(qstates=qstates)

        # check that nothing has been done, so undoing yields error
        with self.assertRaises(Exception):
            basis.undo_last_modification()

        # modify and undo
        succeeded = basis.deterministically_modify(
            qstate_index=0,
            pauli=qiskit.quantum_info.Pauli('XZ'))
        self.assertTrue(succeeded)
        basis.undo_last_modification()
        self.assertTrue(basis.size, 2)
        self.assertEqual(basis.qstates[0], qstates[0])
        self.assertEqual(basis.qstates[1], qstates[1])

        # check that undoing twice yields error
        with self.assertRaises(Exception):
            basis.undo_last_modification()


class TestGetBasisCopy(unittest.TestCase):

    def test_get_basis_copy(self):

        # create basis

        # |00>
        ket_00 = np.array([[1], [0], [0], [0]])
        qstate_0 = ket_to_qstate(ket_00)

        # ( |00> + |11> ) / sqrt(2)
        z_minus = StabRepr(check_matrix=[[1, 1, 0, 0], [0, 0, 1, 1]],
                           phases=[1, 1])
        qubits = qapi.create_qubits(num_qubits=2)
        qapi.assign_qstate(qubits, qrepr=z_minus)
        qstate_1 = qubits[0].qstate

        basis_a = Basis(qstates=[qstate_0, qstate_1])

        basis_b = get_basis_copy(basis_a)

        self.assertNotEqual(basis_a.qstates[0], basis_b.qstates[0])
        self.assertNotEqual(basis_a.qstates[1], basis_b.qstates[1])
        self.assertTrue(basis_a.qstates[0].compare(basis_b.qstates[0]))
        self.assertTrue(basis_a.qstates[1].compare(basis_b.qstates[1]))


if __name__ == "__main__":
    unittest.main()
