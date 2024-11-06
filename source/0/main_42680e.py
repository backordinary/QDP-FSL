# https://github.com/RavenPillmann/quantum-battleship/blob/0cd9bf7d3d954dc0f32b94bdf4f49853334d232d/main.py
import sys
import getpass
import math
sys.path.append("../qiskit-sdk-py/")
from qiskit import QuantumProgram
import Qconfig

SHOTS = 1024

# configuration for python to write our QASM file

def setShipPosition(player, ship, shipPos):
    positionPicked = False

    while (not positionPicked):
        position = getpass.getpass("Player " + str(player + 1) + ", choose a position for ship " + str(ship + 1) + ". (0, 1, 2, 3, or 4)\n")

        try:
            position = int(position)

            if (position >= 0 and position <= 4 and not (position in shipPos[player])):
                shipPos[player].append(position)
                positionPicked = True
            else:
                print("Not a valid position. Try again.\n")
        except Exception:
            print("Not an integer, please try again.\n")


def bombShip(player, bombPos):
    positionPicked = False

    while (not positionPicked):
        position = input("Player " + str(player + 1) + ", choose a postion to bomb. (0, 1, 2, 3, or 4)\n")

        try:
            position = int(position)

            if (position >= 0 and position <= 4):
                bombPos[player][position] = bombPos[player][position] + 1
                positionPicked = True
            else:
                print("Not a valid position. Try again.\n")
        except Exception:
            print("Not an integer, please try again.\n")


def calculateDamageToShips(grid):
    damage = [[0.0] * 5 for i in range(2)]
    # check all bits
    for key in grid[0].keys():
        for bit in range(5):
            if (key[bit] == "1"):
                damage[0][4 - bit] = damage[0][4 - bit] + grid[0][key]/SHOTS
    # check all bits
    for key in grid[1].keys():
        for bit in range(5):
            if (key[bit] == "1"):
                damage[1][4 - bit] = damage[1][4 - bit] + grid[1][key]/SHOTS

    return damage


def displayBoards(damage):
    print("Player " + str(0) + "'s board:\n")
    print("Position     Damage\n")
    print(str(0) + "            " + (str(math.floor(damage[0][0]*100)) + "%\n" if damage[0][0] else "?\n"))
    print(str(1) + "            " + (str(math.floor(damage[0][1]*100)) + "%\n" if damage[0][1] else "?\n"))
    print(str(2) + "            " + (str(math.floor(damage[0][2]*100)) + "%\n" if damage[0][2] else "?\n"))
    print(str(3) + "            " + (str(math.floor(damage[0][3]*100)) + "%\n" if damage[0][3] else "?\n"))
    print(str(4) + "            " + (str(math.floor(damage[0][4]*100)) + "%\n" if damage[0][4] else "?\n"))
    print("-----------------------\n")
    print("Player " + str(1) + "'s board:\n")
    print("Position     Damage\n")
    print(str(0) + "            " + (str(math.floor(damage[1][0]*100)) + "%\n" if damage[1][0] else "?\n"))
    print(str(1) + "            " + (str(math.floor(damage[1][1]*100)) + "%\n" if damage[1][1] else "?\n"))
    print(str(2) + "            " + (str(math.floor(damage[1][2]*100)) + "%\n" if damage[1][2] else "?\n"))
    print(str(3) + "            " + (str(math.floor(damage[1][3]*100)) + "%\n" if damage[1][3] else "?\n"))
    print(str(4) + "            " + (str(math.floor(damage[1][4]*100)) + "%\n" if damage[1][4] else "?\n"))
    print("-----------------------\n")


def playGame(device, shipPos):
    gameOver = False
    bombPos = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    grid = [0, 0]

    gameOver = False

    while (not gameOver):
        bombShip(0, bombPos)
        bombShip(1, bombPos)
        print("Measuring damage to Player " + str(0 + 1) + "...")

        Q_SPECS = {
            "circuits": [{
                "name": "gridScript",
                "quantum_registers": [{
                    "name": "q",
                    "size": 5
                }],
                "classical_registers": [{
                    "name": "c",
                    "size": 5
            }]}],
        }

        Q_program = QuantumProgram(specs=Q_SPECS)
        gridScript = Q_program.get_circuit("gridScript")
        q = Q_program.get_quantum_registers("q")
        c = Q_program.get_classical_registers("c")
        for hit in range(bombPos[(0 + 1) % 2][0]):
            for ship in range(3):
                if (0 == shipPos[0][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[0])
        for hit in range(bombPos[(0 + 1) % 2][1]):
            for ship in range(3):
                if (1 == shipPos[0][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[1])
        for hit in range(bombPos[(0 + 1) % 2][2]):
            for ship in range(3):
                if (2 == shipPos[0][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[2])
        for hit in range(bombPos[(0 + 1) % 2][3]):
            for ship in range(3):
                if (3 == shipPos[0][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[3])
        for hit in range(bombPos[(0 + 1) % 2][4]):
            for ship in range(3):
                if (4 == shipPos[0][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[4])
        gridScript.measure(q[0], c[0])
        gridScript.measure(q[1], c[1])
        gridScript.measure(q[2], c[2])
        gridScript.measure(q[3], c[3])
        gridScript.measure(q[4], c[4])

        Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
        Q_program.execute(["gridScript"], device, SHOTS, wait=2, timeout=60)

        grid[0] = Q_program.get_counts("gridScript")
        print("Measuring damage to Player " + str(1 + 1) + "...")

        Q_SPECS = {
            "circuits": [{
                "name": "gridScript",
                "quantum_registers": [{
                    "name": "q",
                    "size": 5
                }],
                "classical_registers": [{
                    "name": "c",
                    "size": 5
            }]}],
        }

        Q_program = QuantumProgram(specs=Q_SPECS)
        gridScript = Q_program.get_circuit("gridScript")
        q = Q_program.get_quantum_registers("q")
        c = Q_program.get_classical_registers("c")
        for hit in range(bombPos[(1 + 1) % 2][0]):
            for ship in range(3):
                if (0 == shipPos[1][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[0])
        for hit in range(bombPos[(1 + 1) % 2][1]):
            for ship in range(3):
                if (1 == shipPos[1][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[1])
        for hit in range(bombPos[(1 + 1) % 2][2]):
            for ship in range(3):
                if (2 == shipPos[1][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[2])
        for hit in range(bombPos[(1 + 1) % 2][3]):
            for ship in range(3):
                if (3 == shipPos[1][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[3])
        for hit in range(bombPos[(1 + 1) % 2][4]):
            for ship in range(3):
                if (4 == shipPos[1][ship]):
                    frac = 1/(ship + 1)
                    gridScript.u3(frac * math.pi, 0.0, 0.0, q[4])
        gridScript.measure(q[0], c[0])
        gridScript.measure(q[1], c[1])
        gridScript.measure(q[2], c[2])
        gridScript.measure(q[3], c[3])
        gridScript.measure(q[4], c[4])

        Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
        Q_program.execute(["gridScript"], device, SHOTS, wait=2, timeout=60)

        grid[1] = Q_program.get_counts("gridScript")

        if (('Error' in grid[0].values()) or ('Error' in grid[1].values())):
            print("\nThe process timed out. Try this round again.\n")
        else:
            damage = calculateDamageToShips(grid)
            displayBoards(damage)
            if (
                damage[0][shipPos[0][0]] > 0.95 and
                damage[0][shipPos[0][1]] > 0.95 and
                damage[0][shipPos[0][2]] > .95
            ):
                print("All ships on Player " + str(0) + "'s board are destroyed! \n")
                gameOver = True
            if (
                damage[1][shipPos[1][0]] > 0.95 and
                damage[1][shipPos[1][1]] > 0.95 and
                damage[1][shipPos[1][2]] > .95
            ):
                print("All ships on Player " + str(1) + "'s board are destroyed! \n")
                gameOver = True

            if (gameOver):
                print("Game Over")


def main():
    d = input("Do you want to play on a real device? (y/n)\n")

    if (d == "y"):
        device = 'ibmqx2'
    else:
        device = 'local_qasm_simulator'

    # Read this as shipPos[player][ship] = position of player's ship
    shipPos = [
        [],
        []
    ]
    setShipPosition(0, 0, shipPos)
    setShipPosition(0, 1, shipPos)
    setShipPosition(0, 2, shipPos)
    setShipPosition(1, 0, shipPos)
    setShipPosition(1, 1, shipPos)
    setShipPosition(1, 2, shipPos)

    playGame(device, shipPos)


if __name__ == "__main__":
    main()
