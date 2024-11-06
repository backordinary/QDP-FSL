# https://github.com/ArtemPervushow/QuantumSea/blob/b272da800f39cf62a74e049f5c9a8787c02ef7dc/QuantumSea/main.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer
from qiskit import IBMQ
import getpass
import random
import numpy
import math


def choose_device():
    d = input("Do you want to play on the real device? (y/n)\n")

    if d == "y":
        IBMQ.enable_account("49e2792855331abd8ce00ec43ae24a1e203af9856e8adc64fb82189acb1df11c1fab330fd9eee6d655e4c9e2fe899436987d83a316ab88dd5258c49b42b434f7")

        provider = IBMQ.get_provider(hub='ibm-q')
        return provider.get_backend('ibmq_armonk')
    else:
        return BasicAer.get_backend('qasm_simulator')


def setup_ships():
    ship_positions = [[-1] * 3 for _ in range(2)]

    for player in [0, 1]:
        print("\nPlayer " + str(player+1) + " is choosing\n")
        for ship in [0, 1, 2]:
            print("\n Setup ship number " + str(ship+1))
            in_choose = True
            while in_choose:
                chosen_position = getpass.getpass("Player " + str(player+1) + ", choose a position for ship " + str(ship+1) + " (0, 1, 2, 3 or 4)\n")
                if chosen_position.isdigit():
                    chosen_position = int(chosen_position)
                    if (chosen_position in [0, 1, 2, 3, 4]) and (chosen_position not in ship_positions[player]):
                        ship_positions[player][ship] = chosen_position
                        in_choose = False
                    elif chosen_position in ship_positions[player]:
                        print("\nPosition is not empty\n")
                    else:
                        print("\nInvalid position\n")
                else:
                    print("\nPosition must be a digit from 0 to 4\n")

    return ship_positions


def make_shot(shot):
    for player in [0, 1]:
        print("\nPlayer " + str(player+1) + " shoot!")
        shooting = True

        while shooting:
            shoot_coord = input("Choose a position to shoot (0, 1, 2, 3 or 4)\n")
            if shoot_coord.isdigit():
                shoot_coord = int(shoot_coord)
                if shoot_coord in range(5):
                    shot[player][shoot_coord] = shot[player][shoot_coord] + 1
                    shooting = False
                    print("\n")
                else:
                    print("\nThat's not a valid position. Try again.\n")
            else:
                print("\nThat's not a valid position. Try again.\n")
    return shot


def run_game():
    proceed = True
    shot = [[0] * 5 for _ in range(2)]
    grid = [{}, {}]
    device = choose_device()
    ship_positions = setup_ships()

    while proceed:
        shot = make_shot(shot)
        qc = []

        for player in [0, 1]:
            q = QuantumRegister(5)
            c = ClassicalRegister(5)
            qc.append(QuantumCircuit(q, c))

            for coord in range(5):
                for _ in range(shot[(player + 1) % 2][coord]):
                    for ship in [0, 1, 2]:
                        if coord == ship_positions[player][ship]:
                            fraction = 1 / (ship + 1)
                            qc[player].u3(fraction * math.pi, 0.0, 0.0, q[coord])
        qc[0].measure_all()
        qc[1].measure_all()

        job = execute(qc, device, shots=1024)
        grid[0] = job.result().get_counts(qc[0])
        grid[1] = job.result().get_counts(qc[1])
        print(grid)

        proceed = display_grid(grid, ship_positions, 1024)


def display_grid(grid, shipPos, shots):
    game = True

    damage = [[0] * 5 for _ in range(2)]
    for bitString in grid[0].keys():
        for position in range(5):
            if bitString[4 - position] == "1":
                damage[0][position] += grid[0][bitString] / shots
    for bitString in grid[1].keys():
        for position in range(5):
            if bitString[4 - position] == "1":
                damage[1][position] += grid[1][bitString] / shots

    for player in [0, 1]:

        input("\nPress Enter to see the results for Player " + str(player + 1) + "'s ships...\n")

        display = [" ?  "] * 5
        for position in shipPos[player]:
            if damage[player][position] > 0.1:
                if damage[player][position] > 0.9:
                    display[position] = "100%"
                else:
                    display[position] = str(int(100 * damage[player][position])) + "% "

        print("Here is the percentage damage for ships that have been bombed.\n")
        print(display[4] + "    " + display[0])
        print(r' |\     /|')
        print(r' | \   / |')
        print(r' |  \ /  |')
        print(" |  " + display[2] + " |")
        print(r' |  / \  |')
        print(r' | /   \ |')
        print(r' |/     \|')
        print(display[3] + "    " + display[1])
        print("\n")
        print("Ships with 95% damage or more have been destroyed\n")

        print("\n")

        if (damage[player][shipPos[player][0]] > .9) and (damage[player][shipPos[player][1]] > .9) and (
                damage[player][shipPos[player][2]] > .9):
            print("***All Player " + str(player + 1) + "'s ships have been destroyed!***\n\n")
            game = False

        if game is False:
            print("")
            print("=====================================GAME OVER=====================================")
            print("")

    return game


if __name__ == '__main__':
    run_game()
