# https://github.com/petra-b/QTicTacToe/blob/5b9251ae4ba27df21ab816c575b2b1d078d04d07/qttt_tests.py
import unittest
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.circuit.measure import Measure
from qiskit.quantum_info import Statevector
from main import make_grid, o_superposition, x_superposition, measure_grid, run_simulator, check_state, check_full, \
    check_format, check_index, check_occupied, check_not_repeated, check_move, update_cells, check_winner


class QTicTacToeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = Aer.get_backend('aer_simulator')

    def test_make_grid_empty(self):
        grid = make_grid("000000000000000000")
        cells = []
        cells.append(QuantumRegister(2, name='cell_' + str(0)))
        cells.append(QuantumRegister(2, name='cell_' + str(1)))
        cells.append(QuantumRegister(2, name='cell_' + str(2)))
        cells.append(QuantumRegister(2, name='cell_' + str(3)))
        cells.append(QuantumRegister(2, name='cell_' + str(4)))
        cells.append(QuantumRegister(2, name='cell_' + str(5)))
        cells.append(QuantumRegister(2, name='cell_' + str(6)))
        cells.append(QuantumRegister(2, name='cell_' + str(7)))
        cells.append(QuantumRegister(2, name='cell_' + str(8)))
        grid2 = QuantumCircuit(*cells, ClassicalRegister(18))
        grid.save_statevector()
        result1 = self.sim.run(grid).result()
        grid2.save_statevector()
        result2 = self.sim.run(grid2).result()
        self.assertEqual(result1.get_statevector(), result2.get_statevector())

    def test_make_grid_case1(self):
        grid = make_grid("010011000100000011")
        grid.save_statevector()
        result1 = self.sim.run(grid).result().get_statevector()
        sv = Statevector.from_label("010011000100000011"[::-1])
        self.assertEqual(result1, sv)

    def test_o_superposition_2qutrits(self):
        grid1 = QuantumCircuit(4)
        o_superposition(grid1, 0, 1)
        grid2 = QuantumCircuit(4)
        grid2.initialize([0. + 0.j, 0. + 0.j, 0.5 ** 0.5 + 0.j, 0. + 0.j,
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                          0.5 ** 0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j], [0, 1, 2, 3])
        grid1.save_statevector()
        result1 = self.sim.run(grid1).result().get_statevector()
        grid2.save_statevector()
        result2 = self.sim.run(grid2).result().get_statevector()
        self.assertEqual(result1, result2)

    def test_x_superposition_2qutrits(self):
        grid1 = QuantumCircuit(4)
        x_superposition(grid1, 0, 1)
        grid2 = QuantumCircuit(4)
        grid2.initialize([0. + 0.j, 0. + 0.j, 0. + 0.j, 0.5 ** 0.5 + 0.j,
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                          0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j,
                          0.5 ** 0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j], [0, 1, 2, 3])
        grid1.save_statevector()
        result1 = self.sim.run(grid1).result().get_statevector()
        grid2.save_statevector()
        result2 = self.sim.run(grid2).result().get_statevector()
        self.assertEqual(result1, result2)

    def test_measure_grid(self):
        grid = QuantumCircuit(18, 18)
        measure_grid(grid)
        for gate in grid:
            self.assertIsInstance(gate[0], Measure)

    def test_run_simulator(self):
        grid = QuantumCircuit(4, 4)
        grid.x([1, 2, 3])
        grid.measure(0, 0)
        grid.measure(1, 1)
        grid.measure(2, 2)
        grid.measure(3, 3)
        result = run_simulator(grid, self.sim)
        self.assertEqual(result, "0111")

    def test_check_state(self):
        self.assertEqual(check_state("00"), " .")
        self.assertEqual(check_state("01"), " o")
        self.assertEqual(check_state("11"), " x")

    def test_check_full_case1(self):
        full = check_full("110111011111110101")
        self.assertEqual(full, True)

    def test_check_full_case2(self):
        full = check_full("110011011111110101")
        self.assertEqual(full, False)

    def test_check_format_case1(self):
        self.assertEqual(False, check_format("1 3"))

    def test_check_format_case2(self):
        self.assertEqual(True, check_format("3,5"))

    def test_check_format_case3(self):
        self.assertEqual(True, check_format("7"))

    def test_check_index_wrong_input1(self):
        self.assertEqual(False, check_index("9"))

    def test_check_index_wrong_input2(self):
        self.assertEqual(False, check_index("3,9"))

    def test_check_index_correct_input1(self):
        self.assertEqual(True, check_index("3"))

    def test_check_occupied_wrong_input(self):
        self.assertEqual(False, check_occupied("2,3", [' .'] * 3 + [' x'] + [' .'] * 5))

    def test_check_occupied_wrong_input2(self):
        self.assertEqual(False, check_occupied("1", [' o'] * 9))

    def test_check_occupied_correct_input(self):
        self.assertEqual(True, check_occupied("5,3", [' .'] * 9))

    def test_check_repeated_one_cell_input(self):
        self.assertEqual(True, check_not_repeated("1"))

    def test_check_move_repeated_cell(self):
        self.assertEqual(False, check_move("3,3", [' .'] * 9))

    def test_check_move_occupied_cell(self):
        self.assertEqual(False, check_move("3,7", [' .'] * 7 + [' o'] * 2))

    def test_check_move_valid_input1(self):
        self.assertEqual(True, check_move("3,1", [' .'] * 9))

    def test_check_move_valid_input2(self):
        self.assertEqual(True, check_move("1", [' .'] * 9))

    def test_update_cells(self):
        cells = update_cells("011111111111111111")
        self.assertEqual(cells, [' o'] + [' x'] * 8)

    def test_check_winner_empty_board(self):
        self.assertEqual(False, check_winner([' .'] * 9))

    def test_check_winner_full_diagonal(self):
        self.assertEqual(True, check_winner([' .', ' o', ' x',
                                             ' o', ' x', ' o',
                                             ' x', ' o', ' .']))


if __name__ == '__main__':
    unittest.main()
