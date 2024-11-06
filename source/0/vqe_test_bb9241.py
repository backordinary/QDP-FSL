# https://github.com/eduardsmetanin/vqe/blob/102426228b8a6caaadb9c148431631122d5dcb1f/vqe_test.py
import unittest
import numpy as np
import qiskit.aqua
import decomposer
import vqe

class TestVQE(unittest.TestCase):
    def test_vqe(self):
        test_cases = [
            np.array([[0, 0, 0, 0], [0, -1, 1, 0], [0, 1, -1, 0], [0, 0, 0, 0]], dtype=np.complex128),
            np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]], dtype=np.complex128),
            np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex128),
        ]
        for hamiltonian in test_cases:
            actual_ground_state_energy = vqe.run(hamiltonian)
            decomposed_hamiltonian = decomposer.run(hamiltonian)
            expectedResult = qiskit.aqua.algorithms.ExactEigensolver(weighted_pauli_operator(decomposed_hamiltonian)).run()
            expected_ground_state_energy = expectedResult['energy']
            print(f"Estimated ground state energy: {actual_ground_state_energy}")
            self.assertAlmostEqual(actual_ground_state_energy, expected_ground_state_energy)

def weighted_pauli_operator(decomposed):
    """ Converts given decomposed matrix to QISKit format.
    :param decomposed: decomposed matrix in decomposer.py format.
    :returns: converted to QISKit format decomposed matrix.
    """
    coefficients = []
    for key, value in decomposed.items():
        coefficients.append({"coeff": {"imag": value.imag, "real": value.real}, "label": key})
    return qiskit.aqua.operators.WeightedPauliOperator.from_dict({"paulis": coefficients})

if __name__ == '__main__':
    unittest.main()
