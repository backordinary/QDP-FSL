# https://github.com/petra-b/QTicTacToe/blob/5b9251ae4ba27df21ab816c575b2b1d078d04d07/main.py
from qiskit import *

sim = Aer.get_backend('aer_simulator')  # choose the simulator to execute our circuit on


# Use qutrits to describe the possible states
# |00> corresponds to an empty cell
# |01> the cell contains o
# |11> the cell contains x

def make_grid(result):
    """Prepares the quantum circuit representing the grid and initializes it to the previous result."""
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
    grid = QuantumCircuit(*cells, ClassicalRegister(18))
    if result[0 * 2:0 * 2 + 2] == '01':
        grid.x(0 * 2 + 1)
    elif result[0 * 2:0 * 2 + 2] == '11':
        grid.x(0 * 2)
        grid.x(0 * 2 + 1)
    if result[1 * 2:1 * 2 + 2] == '01':
        grid.x(1 * 2 + 1)
    elif result[1 * 2:1 * 2 + 2] == '11':
        grid.x(1 * 2)
        grid.x(1 * 2 + 1)
    if result[2 * 2:2 * 2 + 2] == '01':
        grid.x(2 * 2 + 1)
    elif result[2 * 2:2 * 2 + 2] == '11':
        grid.x(2 * 2)
        grid.x(2 * 2 + 1)
    if result[3 * 2:3 * 2 + 2] == '01':
        grid.x(3 * 2 + 1)
    elif result[3 * 2:3 * 2 + 2] == '11':
        grid.x(3 * 2)
        grid.x(3 * 2 + 1)
    if result[4 * 2:4 * 2 + 2] == '01':
        grid.x(4 * 2 + 1)
    elif result[4 * 2:4 * 2 + 2] == '11':
        grid.x(4 * 2)
        grid.x(4 * 2 + 1)
    if result[5 * 2:5 * 2 + 2] == '01':
        grid.x(5 * 2 + 1)
    elif result[5 * 2:5 * 2 + 2] == '11':
        grid.x(5 * 2)
        grid.x(5 * 2 + 1)
    if result[6 * 2:6 * 2 + 2] == '01':
        grid.x(6 * 2 + 1)
    elif result[6 * 2:6 * 2 + 2] == '11':
        grid.x(6 * 2)
        grid.x(6 * 2 + 1)
    if result[7 * 2:7 * 2 + 2] == '01':
        grid.x(7 * 2 + 1)
    elif result[7 * 2:7 * 2 + 2] == '11':
        grid.x(7 * 2)
        grid.x(7 * 2 + 1)
    if result[8 * 2:8 * 2 + 2] == '01':
        grid.x(8 * 2 + 1)
    elif result[8 * 2:8 * 2 + 2] == '11':
        grid.x(8 * 2)
        grid.x(8 * 2 + 1)
    return grid


def o_superposition(grid, cell1, cell2):
    """Puts the two qutrits in an equal superposition of |0100> and |0001> states."""
    grid.h(cell1 * 2 + 1)
    grid.cx(cell1 * 2 + 1, cell2 * 2 + 1)
    grid.x(cell1 * 2 + 1)


def x_superposition(grid, cell1, cell2):
    """Puts the two qutrits in an equal superposition of |1100> and |0011> states."""
    grid.h(cell1 * 2)
    grid.cx(cell1 * 2, cell2 * 2 + 1)
    grid.x(cell1 * 2)
    grid.cx(cell1 * 2, cell1 * 2 + 1)
    grid.cx(cell2 * 2 + 1, cell2 * 2)


def measure_grid(grid):
    """Measure each qutrit."""
    for i in range(18):
        grid.measure(i, i)


def run_simulator(grid, sim):
    """Run the circuit on the simulator."""
    counts = sim.run(grid, shots=1).result().get_counts()
    result = next(iter(counts))
    return result[::-1]


def check_state(state):
    """Check the qutrit states."""
    if state == '00':
        return ' .'
    elif state == '01':
        return ' o'
    else:
        return ' x'


def print_grid(result):
    """Prints the grid in the conventional format."""
    print('|' + check_state(result[0 * 2:0 * 2 + 2]) + ' ', end='')
    if (0 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[1 * 2:1 * 2 + 2]) + ' ', end='')
    if (1 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[2 * 2:2 * 2 + 2]) + ' ', end='')
    if (2 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[3 * 2:3 * 2 + 2]) + ' ', end='')
    if (3 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[4 * 2:4 * 2 + 2]) + ' ', end='')
    if (4 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[5 * 2:5 * 2 + 2]) + ' ', end='')
    if (5 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[6 * 2:6 * 2 + 2]) + ' ', end='')
    if (6 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[7 * 2:7 * 2 + 2]) + ' ', end='')
    if (7 + 1) % 3 == 0:
        print('|')
    print('|' + check_state(result[8 * 2:8 * 2 + 2]) + ' ', end='')
    if (8 + 1) % 3 == 0:
        print('|')


def check_full(result):
    """Check if the grid is full."""
    full = True
    if not (result[0 * 2:0 * 2 + 2] == '01' or result[0 * 2:0 * 2 + 2] == '11'):
        full = False
    if not (result[1 * 2:1 * 2 + 2] == '01' or result[1 * 2:1 * 2 + 2] == '11'):
        full = False
    if not (result[2 * 2:2 * 2 + 2] == '01' or result[2 * 2:2 * 2 + 2] == '11'):
        full = False
    if not (result[3 * 2:3 * 2 + 2] == '01' or result[3 * 2:3 * 2 + 2] == '11'):
        full = False
    if not (result[4 * 2:4 * 2 + 2] == '01' or result[4 * 2:4 * 2 + 2] == '11'):
        full = False
    if not (result[5 * 2:5 * 2 + 2] == '01' or result[5 * 2:5 * 2 + 2] == '11'):
        full = False
    if not (result[6 * 2:6 * 2 + 2] == '01' or result[6 * 2:6 * 2 + 2] == '11'):
        full = False
    if not (result[7 * 2:7 * 2 + 2] == '01' or result[7 * 2:7 * 2 + 2] == '11'):
        full = False
    if not (result[8 * 2:8 * 2 + 2] == '01' or result[8 * 2:8 * 2 + 2] == '11'):
        full = False
    return full


def player_o():
    """Player o input."""
    move = input("Player o: ")
    return move


def player_x():
    """Player x input."""
    move = input("Player x: ")
    return move


def check_format(last_move):
    """Check if the input is in the correct format."""
    if len(last_move) == 1:
        return True
    if len(last_move) == 3 and last_move[1] == ',':
        return True
    return False


def check_index(last_move):
    """Check if the cell index is within the range."""
    if len(last_move) == 1:
        if 0 <= int(last_move) <= 8:
            return True
        else:
            return False
    if 0 <= int(last_move[2]) <= 8:
        return True
    return False


def check_not_repeated(last_move):
    if len(last_move) == 3:
        return last_move[0] != last_move[2]
    return True


def check_occupied(last_move, cells):
    """Check if the entered cell is already occupied"""
    if cells[int(last_move[0])] != ' .':
        return False
    if len(last_move) == 3:
        if cells[int(last_move[2])] != ' .':
            return False
    return True


def check_move(last_move, cells):
    """Check if the player has entered a valid move."""
    if not check_format(last_move) or not check_index(last_move) or not check_not_repeated(last_move):
        print('Please enter the move in the correct format \'a\' or \'a,b\', where 0 <= a,b <= 8, and a!=b.')
        return False
    if not check_occupied(last_move, cells):
        print('Please enter only the unoccupied cells.')
        return False
    return True


def print_intermediate(cells, last_move, n, player):
    """Print the grid inbetween measurements."""
    if str(0) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[0] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[0] = ' ' + player
    else:
        print('|' + cells[0] + ' ', end='')
    if (0 + 1) % 3 == 0:
        print('|')
    if str(1) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[1] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[1] = ' ' + player
    else:
        print('|' + cells[1] + ' ', end='')
    if (1 + 1) % 3 == 0:
        print('|')
    if str(2) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[2] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[2] = ' ' + player
    else:
        print('|' + cells[2] + ' ', end='')
    if (2 + 1) % 3 == 0:
        print('|')
    if str(3) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[3] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[3] = ' ' + player
    else:
        print('|' + cells[3] + ' ', end='')
    if (3 + 1) % 3 == 0:
        print('|')
    if str(4) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[4] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[4] = ' ' + player
    else:
        print('|' + cells[4] + ' ', end='')
    if (4 + 1) % 3 == 0:
        print('|')
    if str(5) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[5] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[5] = ' ' + player
    else:
        print('|' + cells[5] + ' ', end='')
    if (5 + 1) % 3 == 0:
        print('|')
    if str(6) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[6] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[6] = ' ' + player
    else:
        print('|' + cells[6] + ' ', end='')
    if (6 + 1) % 3 == 0:
        print('|')
    if str(7) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[7] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[7] = ' ' + player
    else:
        print('|' + cells[7] + ' ', end='')
    if (7 + 1) % 3 == 0:
        print('|')
    if str(8) in last_move:
        if len(last_move) > 1:
            print('|' + player + str(n) + ' ', end='')
            cells[8] = player + str(n)
        else:
            print('| ' + player + ' ', end='')
            cells[8] = ' ' + player
    else:
        print('|' + cells[8] + ' ', end='')
    if (8 + 1) % 3 == 0:
        print('|')


def update_cells(result):
    """Update the cells after measurement."""
    cells = [' .'] * 9
    cells[0] = check_state(result[0 * 2:0 * 2 + 2])
    cells[1] = check_state(result[1 * 2:1 * 2 + 2])
    cells[2] = check_state(result[2 * 2:2 * 2 + 2])
    cells[3] = check_state(result[3 * 2:3 * 2 + 2])
    cells[4] = check_state(result[4 * 2:4 * 2 + 2])
    cells[5] = check_state(result[5 * 2:5 * 2 + 2])
    cells[6] = check_state(result[6 * 2:6 * 2 + 2])
    cells[7] = check_state(result[7 * 2:7 * 2 + 2])
    cells[8] = check_state(result[8 * 2:8 * 2 + 2])
    return cells


def check_winner(cells):
    """Check if there is a winner."""
    win_states = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for (x, y, z) in win_states:
        if cells[x] == cells[y] == cells[z] != ' .':
            print('The winner is Player ' + cells[x] + '!')
            return True
    return False


def play_game(sim):
    result = '0' * 18
    cells = [' .'] * 9
    k = 0
    while not check_full(result):
        winner = False
        n = 1
        grid = make_grid(result)
        while ' .' in cells:
            if k % 2 == 0:
                last_move = player_x()
                while not check_move(last_move, cells):
                    last_move = player_x()
                print_intermediate(cells, last_move, n, 'x')
                if len(last_move) > 1:
                    x_superposition(grid, int(last_move[0]), int(last_move[2]))
                else:
                    grid.x(int(last_move) * 2)
                    grid.x(int(last_move) * 2 + 1)
                    winner = check_winner(cells)
                    if winner:
                        break
            else:
                last_move = player_o()
                while not check_move(last_move, cells):
                    last_move = player_o()
                print_intermediate(cells, last_move, n, 'o')
                if len(last_move) > 1:
                    o_superposition(grid, int(last_move[0]), int(last_move[2]))
                else:
                    grid.x(int(last_move) * 2 + 1)
                    winner = check_winner(cells)
                    if winner:
                        break
            n = n + 1
            k = k + 1
        if winner:
            break
        measure_grid(grid)
        result = run_simulator(grid, sim)
        print('Collapsed:')
        print_grid(result)
        cells = update_cells(result)
        winner = check_winner(cells)
        if winner:
            break
    if not winner:
        print('Draw.')


if __name__ == "__main__":
    play_game(sim)
