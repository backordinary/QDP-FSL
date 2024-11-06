# https://github.com/yigithankardas/Q2048/blob/24b0837aadba134a0641ae9b58a872426356b801/game2.py
import random
from qiskit import QuantumCircuit, execute, Aer
import time
from tkinter import *


probability_array = []
for i in range(100):
    if i < 70:
        probability_array.append(2)
    elif i < 90 and i > 70:
        probability_array.append(1)    # H Gate
    else:
        probability_array.append(3)    # Z gate
random.shuffle(probability_array)


class Temp:
    current_move = "s"


temp = Temp()


def determineCircuit(cell):
    if cell.value > 0:
        initial_state = [1, 0]
        cell.qc.initialize(initial_state, 0)
    else:
        initial_state = [0, 1]
        cell.qc.initialize(initial_state, 0)


def getSign(qc):
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=1024)
    counts = job.result().get_counts(qc)
    if counts.get('1') > counts.get('0'):
        return "-"
    else:
        return "+"


def passThroughHGate(tile):
    tile.qc.h(0)
    tile.isInSuperposition = not tile.isInSuperposition
    tile.qc.h(0)
    sign = getSign(tile.qc)
    if sign == "+":
        tile.value = abs(tile.value)
    else:
        tile.value = abs(tile.value) * -1


def passThroughZGate(tile):
    tile.qc.z(0)
    sign = getSign(tile.qc)
    if sign == "+":
        tile.value = abs(tile.value)
    else:
        tile.value = abs(tile.value) * -1


class Tile:
    def __init__(self, value):
        self.value = value
        self.qc = QuantumCircuit(1)
        initial_state = [1, 0]   # Define initial_state as |1>
        self.qc.initialize(initial_state, 0)
        self.qc.h(0)
        self.qc.measure_all()
        if self.value != 0:
            sign = getSign(self.qc)
            if sign == "+":
                self.value = 2
            else:
                self.value = -2

    def toString(self):
        if self.value != 0:
            if self.isInSuperposition:
                return "+/-" + str(abs(self.value))
            return str(self.value)
        return ""

    isInSuperposition = False
    isMergable = True
    canMove = True


class HGate:
    def toString(self):
        return self.value
    value = "H"
    isMergable = True


class ZGate:
    def toString(self):
        return self.value
    value = "Z"
    isMergable = True


cells = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
cells[0][0] = Tile(0)
cells[0][1] = Tile(0)
cells[0][2] = Tile(0)
cells[0][3] = Tile(0)
cells[1][0] = Tile(0)
cells[1][1] = Tile(0)
cells[1][2] = Tile(0)
cells[1][3] = Tile(0)
cells[2][0] = Tile(0)
cells[2][1] = Tile(0)
cells[2][2] = Tile(0)
cells[2][3] = Tile(0)
cells[3][0] = Tile(0)
cells[3][1] = Tile(0)
cells[3][2] = Tile(0)
cells[3][3] = Tile(0)


def spawn():
    rand = probability_array[random.randint(0, 99)]
    while True:
        x = random.randint(0, 3)
        y = random.randint(0, 3)
        if cells[x][y].value == 0:
            if rand == 1:
                cells[x][y] = HGate()
            elif rand == 3:
                cells[x][y] = ZGate()
            else:
                cells[x][y] = Tile(-1)
            break


spawn()
spawn()


def canReach(cell, i, j):
    if i < 0 or j < 0 or i > 3 or j > 3:
        return False
    if cells[i][j].value != 0:
        if cells[i][j].value == "H":
            return True
        if cells[i][j].value == "Z":
            return not cell.isInSuperposition
        if cells[i][j].value == cell.value or cells[i][j].value == -cell.value:
            if cells[i][j].isMergable == True and cell.isMergable == True:
                return True
        return False
    return True


def moveUp():
    if cells[0][0].value != 0 and cells[0][0].value != "H" and cells[0][0].value != "Z":
        k = 0
        while canReach(cells[k][0], k-1, 0) and cells[k][0].canMove:
            if cells[k-1][0].value == "H":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughHGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == "Z":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughZGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == cells[k][0].value or cells[k-1][0].value == -cells[k][0].value:
                value = cells[k-1][0].value
                cells[k-1][0] = cells[k][0]
                cells[k-1][0].value = cells[k-1][0].value + value
                cells[k-1][0].isInSuperposition = False
                cells[k-1][0].isMergable = False
                cells[k][0] = Tile(0)
                determineCircuit(cells[k-1][0])
            else:
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
            k -= 1
    if cells[0][1].value != 0 and cells[0][1].value != "H" and cells[0][1].value != "Z":
        k = 0
        while canReach(cells[k][1], k-1, 1) and cells[k][1].canMove:
            if cells[k-1][1].value == "H":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughHGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == "Z":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughZGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == cells[k][1].value or cells[k-1][1].value == -cells[k][1].value:
                value = cells[k-1][1].value
                cells[k-1][1] = cells[k][1]
                cells[k-1][1].value = cells[k-1][1].value + value
                cells[k-1][1].isInSuperposition = False
                cells[k-1][1].isMergable = False
                cells[k][1] = Tile(0)
                determineCircuit(cells[k-1][1])
            else:
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
            k -= 1
    if cells[0][2].value != 0 and cells[0][2].value != "H" and cells[0][2].value != "Z":
        k = 0
        while canReach(cells[k][2], k-1, 2) and cells[k][2].canMove:
            if cells[k-1][2].value == "H":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughHGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == "Z":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughZGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == cells[k][2].value or cells[k-1][2].value == -cells[k][2].value:
                value = cells[k-1][2].value
                cells[k-1][2] = cells[k][2]
                cells[k-1][2].value = cells[k-1][2].value + value
                cells[k-1][2].isInSuperposition = False
                cells[k-1][2].isMergable = False
                cells[k][2] = Tile(0)
                determineCircuit(cells[k-1][2])
            else:
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
            k -= 1
    if cells[0][3].value != 0 and cells[0][3].value != "H" and cells[0][3].value != "Z":
        k = 0
        while canReach(cells[k][3], k-1, 3) and cells[k][3].canMove:
            if cells[k-1][3].value == "H":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughHGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == "Z":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughZGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == cells[k][3].value or cells[k-1][3].value == -cells[k][3].value:
                value = cells[k-1][3].value
                cells[k-1][3] = cells[k][3]
                cells[k-1][3].value = cells[k-1][3].value + value
                cells[k-1][3].isInSuperposition = False
                cells[k-1][3].isMergable = False
                cells[k][3] = Tile(0)
                determineCircuit(cells[k-1][3])
            else:
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
            k -= 1
    if cells[1][0].value != 0 and cells[1][0].value != "H" and cells[1][0].value != "Z":
        k = 1
        while canReach(cells[k][0], k-1, 0) and cells[k][0].canMove:
            if cells[k-1][0].value == "H":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughHGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == "Z":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughZGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == cells[k][0].value or cells[k-1][0].value == -cells[k][0].value:
                value = cells[k-1][0].value
                cells[k-1][0] = cells[k][0]
                cells[k-1][0].value = cells[k-1][0].value + value
                cells[k-1][0].isInSuperposition = False
                cells[k-1][0].isMergable = False
                cells[k][0] = Tile(0)
                determineCircuit(cells[k-1][0])
            else:
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
            k -= 1
    if cells[1][1].value != 0 and cells[1][1].value != "H" and cells[1][1].value != "Z":
        k = 1
        while canReach(cells[k][1], k-1, 1) and cells[k][1].canMove:
            if cells[k-1][1].value == "H":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughHGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == "Z":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughZGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == cells[k][1].value or cells[k-1][1].value == -cells[k][1].value:
                value = cells[k-1][1].value
                cells[k-1][1] = cells[k][1]
                cells[k-1][1].value = cells[k-1][1].value + value
                cells[k-1][1].isInSuperposition = False
                cells[k-1][1].isMergable = False
                cells[k][1] = Tile(0)
                determineCircuit(cells[k-1][1])
            else:
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
            k -= 1
    if cells[1][2].value != 0 and cells[1][2].value != "H" and cells[1][2].value != "Z":
        k = 1
        while canReach(cells[k][2], k-1, 2) and cells[k][2].canMove:
            if cells[k-1][2].value == "H":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughHGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == "Z":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughZGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == cells[k][2].value or cells[k-1][2].value == -cells[k][2].value:
                value = cells[k-1][2].value
                cells[k-1][2] = cells[k][2]
                cells[k-1][2].value = cells[k-1][2].value + value
                cells[k-1][2].isInSuperposition = False
                cells[k-1][2].isMergable = False
                cells[k][2] = Tile(0)
                determineCircuit(cells[k-1][2])
            else:
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
            k -= 1
    if cells[1][3].value != 0 and cells[1][3].value != "H" and cells[1][3].value != "Z":
        k = 1
        while canReach(cells[k][3], k-1, 3) and cells[k][3].canMove:
            if cells[k-1][3].value == "H":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughHGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == "Z":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughZGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == cells[k][3].value or cells[k-1][3].value == -cells[k][3].value:
                value = cells[k-1][3].value
                cells[k-1][3] = cells[k][3]
                cells[k-1][3].value = cells[k-1][3].value + value
                cells[k-1][3].isInSuperposition = False
                cells[k-1][3].isMergable = False
                cells[k][3] = Tile(0)
                determineCircuit(cells[k-1][3])
            else:
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
            k -= 1
    if cells[2][0].value != 0 and cells[2][0].value != "H" and cells[2][0].value != "Z":
        k = 2
        while canReach(cells[k][0], k-1, 0) and cells[k][0].canMove:
            if cells[k-1][0].value == "H":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughHGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == "Z":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughZGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == cells[k][0].value or cells[k-1][0].value == -cells[k][0].value:
                value = cells[k-1][0].value
                cells[k-1][0] = cells[k][0]
                cells[k-1][0].value = cells[k-1][0].value + value
                cells[k-1][0].isInSuperposition = False
                cells[k-1][0].isMergable = False
                cells[k][0] = Tile(0)
                determineCircuit(cells[k-1][0])
            else:
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
            k -= 1
    if cells[2][1].value != 0 and cells[2][1].value != "H" and cells[2][1].value != "Z":
        k = 2
        while canReach(cells[k][1], k-1, 1) and cells[k][1].canMove:
            if cells[k-1][1].value == "H":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughHGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == "Z":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughZGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == cells[k][1].value or cells[k-1][1].value == -cells[k][1].value:
                value = cells[k-1][1].value
                cells[k-1][1] = cells[k][1]
                cells[k-1][1].value = cells[k-1][1].value + value
                cells[k-1][1].isInSuperposition = False
                cells[k-1][1].isMergable = False
                cells[k][1] = Tile(0)
                determineCircuit(cells[k-1][1])
            else:
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
            k -= 1
    if cells[2][2].value != 0 and cells[2][2].value != "H" and cells[2][2].value != "Z":
        k = 2
        while canReach(cells[k][2], k-1, 2) and cells[k][2].canMove:
            if cells[k-1][2].value == "H":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughHGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == "Z":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughZGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == cells[k][2].value or cells[k-1][2].value == -cells[k][2].value:
                value = cells[k-1][2].value
                cells[k-1][2] = cells[k][2]
                cells[k-1][2].value = cells[k-1][2].value + value
                cells[k-1][2].isInSuperposition = False
                cells[k-1][2].isMergable = False
                cells[k][2] = Tile(0)
                determineCircuit(cells[k-1][2])
            else:
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
            k -= 1
    if cells[2][3].value != 0 and cells[2][3].value != "H" and cells[2][3].value != "Z":
        k = 2
        while canReach(cells[k][3], k-1, 3) and cells[k][3].canMove:
            if cells[k-1][3].value == "H":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughHGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == "Z":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughZGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == cells[k][3].value or cells[k-1][3].value == -cells[k][3].value:
                value = cells[k-1][3].value
                cells[k-1][3] = cells[k][3]
                cells[k-1][3].value = cells[k-1][3].value + value
                cells[k-1][3].isInSuperposition = False
                cells[k-1][3].isMergable = False
                cells[k][3] = Tile(0)
                determineCircuit(cells[k-1][3])
            else:
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
            k -= 1
    if cells[3][0].value != 0 and cells[3][0].value != "H" and cells[3][0].value != "Z":
        k = 3
        while canReach(cells[k][0], k-1, 0) and cells[k][0].canMove:
            if cells[k-1][0].value == "H":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughHGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == "Z":
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
                passThroughZGate(cells[k-1][0])
                cells[k-1][0].isMergable = False
                cells[k-1][0].canMove = False
            elif cells[k-1][0].value == cells[k][0].value or cells[k-1][0].value == -cells[k][0].value:
                value = cells[k-1][0].value
                cells[k-1][0] = cells[k][0]
                cells[k-1][0].value = cells[k-1][0].value + value
                cells[k-1][0].isInSuperposition = False
                cells[k-1][0].isMergable = False
                cells[k][0] = Tile(0)
                determineCircuit(cells[k-1][0])
            else:
                cells[k-1][0] = cells[k][0]
                cells[k][0] = Tile(0)
            k -= 1
    if cells[3][1].value != 0 and cells[3][1].value != "H" and cells[3][1].value != "Z":
        k = 3
        while canReach(cells[k][1], k-1, 1) and cells[k][1].canMove:
            if cells[k-1][1].value == "H":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughHGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == "Z":
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
                passThroughZGate(cells[k-1][1])
                cells[k-1][1].isMergable = False
                cells[k-1][1].canMove = False
            elif cells[k-1][1].value == cells[k][1].value or cells[k-1][1].value == -cells[k][1].value:
                value = cells[k-1][1].value
                cells[k-1][1] = cells[k][1]
                cells[k-1][1].value = cells[k-1][1].value + value
                cells[k-1][1].isInSuperposition = False
                cells[k-1][1].isMergable = False
                cells[k][1] = Tile(0)
                determineCircuit(cells[k-1][1])
            else:
                cells[k-1][1] = cells[k][1]
                cells[k][1] = Tile(0)
            k -= 1
    if cells[3][2].value != 0 and cells[3][2].value != "H" and cells[3][2].value != "Z":
        k = 3
        while canReach(cells[k][2], k-1, 2) and cells[k][2].canMove:
            if cells[k-1][2].value == "H":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughHGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == "Z":
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
                passThroughZGate(cells[k-1][2])
                cells[k-1][2].isMergable = False
                cells[k-1][2].canMove = False
            elif cells[k-1][2].value == cells[k][2].value or cells[k-1][2].value == -cells[k][2].value:
                value = cells[k-1][2].value
                cells[k-1][2] = cells[k][2]
                cells[k-1][2].value = cells[k-1][2].value + value
                cells[k-1][2].isInSuperposition = False
                cells[k-1][2].isMergable = False
                cells[k][2] = Tile(0)
                determineCircuit(cells[k-1][2])
            else:
                cells[k-1][2] = cells[k][2]
                cells[k][2] = Tile(0)
            k -= 1
    if cells[3][3].value != 0 and cells[3][3].value != "H" and cells[3][3].value != "Z":
        k = 3
        while canReach(cells[k][3], k-1, 3) and cells[k][3].canMove:
            if cells[k-1][3].value == "H":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughHGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == "Z":
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
                passThroughZGate(cells[k-1][3])
                cells[k-1][3].isMergable = False
                cells[k-1][3].canMove = False
            elif cells[k-1][3].value == cells[k][3].value or cells[k-1][3].value == -cells[k][3].value:
                value = cells[k-1][3].value
                cells[k-1][3] = cells[k][3]
                cells[k-1][3].value = cells[k-1][3].value + value
                cells[k-1][3].isInSuperposition = False
                cells[k-1][3].isMergable = False
                cells[k][3] = Tile(0)
                determineCircuit(cells[k-1][3])
            else:
                cells[k-1][3] = cells[k][3]
                cells[k][3] = Tile(0)
            k -= 1


def moveDown():
    for i in range(3, -1, -1):
        for j in range(3, -1, -1):
            if cells[i][j].value != 0 and cells[i][j].value != "H" and cells[i][j].value != "Z":
                k = i
                while canReach(cells[k][j], k+1, j) and cells[k][j].canMove:
                    if cells[k+1][j].value == "H":
                        cells[k+1][j] = cells[k][j]
                        cells[k][j] = Tile(0)
                        passThroughHGate(cells[k+1][j])
                        cells[k+1][j].isMergable = False
                        cells[k+1][j].canMove = False
                    elif cells[k+1][j].value == "Z":
                        cells[k+1][j] = cells[k][j]
                        cells[k][j] = Tile(0)
                        passThroughZGate(cells[k+1][j])
                        cells[k+1][j].isMergable = False
                        cells[k+1][j].canMove = False
                    elif cells[k+1][j].value == cells[k][j].value or cells[k+1][j].value == -cells[k][j].value:
                        value = cells[k+1][j].value
                        cells[k+1][j] = cells[k][j]
                        cells[k+1][j].value = cells[k+1][j].value + value
                        cells[k+1][j].isInSuperposition = False
                        cells[k+1][j].isMergable = False
                        cells[k][j] = Tile(0)
                        determineCircuit(cells[k+1][j])
                    else:
                        cells[k+1][j] = cells[k][j]
                        cells[k][j] = Tile(0)
                    k += 1


def moveLeft():
    if cells[0][0].value != 0 and cells[0][0].value != "H" and cells[0][0].value != "Z":
        k = 0
        while canReach(cells[0][k], 0, k-1) and cells[0][k].canMove:
            if cells[0][k-1].value == "H":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughHGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == "Z":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughZGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == cells[0][k].value or cells[0][k-1].value == -cells[0][k].value:
                value = cells[0][k-1].value
                cells[0][k-1] = cells[0][k]
                cells[0][k-1].value = cells[0][k-1].value + value
                cells[0][k-1].isInSuperposition = False
                cells[0][k-1].isMergable = False
                cells[0][k] = Tile(0)
                determineCircuit(cells[0][k-1])
            else:
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
            k -= 1
    if cells[1][0].value != 0 and cells[1][0].value != "H" and cells[1][0].value != "Z":
        k = 0
        while canReach(cells[1][k], 1, k-1) and cells[1][k].canMove:
            if cells[1][k-1].value == "H":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughHGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == "Z":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughZGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == cells[1][k].value or cells[1][k-1].value == -cells[1][k].value:
                value = cells[1][k-1].value
                cells[1][k-1] = cells[1][k]
                cells[1][k-1].value = cells[1][k-1].value + value
                cells[1][k-1].isInSuperposition = False
                cells[1][k-1].isMergable = False
                cells[1][k] = Tile(0)
                determineCircuit(cells[1][k-1])
            else:
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
            k -= 1
    if cells[2][0].value != 0 and cells[2][0].value != "H" and cells[2][0].value != "Z":
        k = 0
        while canReach(cells[2][k], 2, k-1) and cells[2][k].canMove:
            if cells[2][k-1].value == "H":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughHGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == "Z":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughZGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == cells[2][k].value or cells[2][k-1].value == -cells[2][k].value:
                value = cells[2][k-1].value
                cells[2][k-1] = cells[2][k]
                cells[2][k-1].value = cells[2][k-1].value + value
                cells[2][k-1].isInSuperposition = False
                cells[2][k-1].isMergable = False
                cells[2][k] = Tile(0)
                determineCircuit(cells[2][k-1])
            else:
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
            k -= 1
    if cells[3][0].value != 0 and cells[3][0].value != "H" and cells[3][0].value != "Z":
        k = 0
        while canReach(cells[3][k], 3, k-1) and cells[3][k].canMove:
            if cells[3][k-1].value == "H":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughHGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == "Z":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughZGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == cells[3][k].value or cells[3][k-1].value == -cells[3][k].value:
                value = cells[3][k-1].value
                cells[3][k-1] = cells[3][k]
                cells[3][k-1].value = cells[3][k-1].value + value
                cells[3][k-1].isInSuperposition = False
                cells[3][k-1].isMergable = False
                cells[3][k] = Tile(0)
                determineCircuit(cells[3][k-1])
            else:
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
            k -= 1
    if cells[0][1].value != 0 and cells[0][1].value != "H" and cells[0][1].value != "Z":
        k = 1
        while canReach(cells[0][k], 0, k-1) and cells[0][k].canMove:
            if cells[0][k-1].value == "H":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughHGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == "Z":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughZGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == cells[0][k].value or cells[0][k-1].value == -cells[0][k].value:
                value = cells[0][k-1].value
                cells[0][k-1] = cells[0][k]
                cells[0][k-1].value = cells[0][k-1].value + value
                cells[0][k-1].isInSuperposition = False
                cells[0][k-1].isMergable = False
                cells[0][k] = Tile(0)
                determineCircuit(cells[0][k-1])
            else:
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
            k -= 1
    if cells[1][1].value != 0 and cells[1][1].value != "H" and cells[1][1].value != "Z":
        k = 1
        while canReach(cells[1][k], 1, k-1) and cells[1][k].canMove:
            if cells[1][k-1].value == "H":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughHGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == "Z":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughZGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == cells[1][k].value or cells[1][k-1].value == -cells[1][k].value:
                value = cells[1][k-1].value
                cells[1][k-1] = cells[1][k]
                cells[1][k-1].value = cells[1][k-1].value + value
                cells[1][k-1].isInSuperposition = False
                cells[1][k-1].isMergable = False
                cells[1][k] = Tile(0)
                determineCircuit(cells[1][k-1])
            else:
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
            k -= 1
    if cells[2][1].value != 0 and cells[2][1].value != "H" and cells[2][1].value != "Z":
        k = 1
        while canReach(cells[2][k], 2, k-1) and cells[2][k].canMove:
            if cells[2][k-1].value == "H":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughHGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == "Z":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughZGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == cells[2][k].value or cells[2][k-1].value == -cells[2][k].value:
                value = cells[2][k-1].value
                cells[2][k-1] = cells[2][k]
                cells[2][k-1].value = cells[2][k-1].value + value
                cells[2][k-1].isInSuperposition = False
                cells[2][k-1].isMergable = False
                cells[2][k] = Tile(0)
                determineCircuit(cells[2][k-1])
            else:
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
            k -= 1
    if cells[3][1].value != 0 and cells[3][1].value != "H" and cells[3][1].value != "Z":
        k = 1
        while canReach(cells[3][k], 3, k-1) and cells[3][k].canMove:
            if cells[3][k-1].value == "H":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughHGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == "Z":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughZGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == cells[3][k].value or cells[3][k-1].value == -cells[3][k].value:
                value = cells[3][k-1].value
                cells[3][k-1] = cells[3][k]
                cells[3][k-1].value = cells[3][k-1].value + value
                cells[3][k-1].isInSuperposition = False
                cells[3][k-1].isMergable = False
                cells[3][k] = Tile(0)
                determineCircuit(cells[3][k-1])
            else:
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
            k -= 1
    if cells[0][2].value != 0 and cells[0][2].value != "H" and cells[0][2].value != "Z":
        k = 2
        while canReach(cells[0][k], 0, k-1) and cells[0][k].canMove:
            if cells[0][k-1].value == "H":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughHGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == "Z":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughZGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == cells[0][k].value or cells[0][k-1].value == -cells[0][k].value:
                value = cells[0][k-1].value
                cells[0][k-1] = cells[0][k]
                cells[0][k-1].value = cells[0][k-1].value + value
                cells[0][k-1].isInSuperposition = False
                cells[0][k-1].isMergable = False
                cells[0][k] = Tile(0)
                determineCircuit(cells[0][k-1])
            else:
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
            k -= 1
    if cells[1][2].value != 0 and cells[1][2].value != "H" and cells[1][2].value != "Z":
        k = 2
        while canReach(cells[1][k], 1, k-1) and cells[1][k].canMove:
            if cells[1][k-1].value == "H":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughHGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == "Z":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughZGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == cells[1][k].value or cells[1][k-1].value == -cells[1][k].value:
                value = cells[1][k-1].value
                cells[1][k-1] = cells[1][k]
                cells[1][k-1].value = cells[1][k-1].value + value
                cells[1][k-1].isInSuperposition = False
                cells[1][k-1].isMergable = False
                cells[1][k] = Tile(0)
                determineCircuit(cells[1][k-1])
            else:
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
            k -= 1
    if cells[2][2].value != 0 and cells[2][2].value != "H" and cells[2][2].value != "Z":
        k = 2
        while canReach(cells[2][k], 2, k-1) and cells[2][k].canMove:
            if cells[2][k-1].value == "H":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughHGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == "Z":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughZGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == cells[2][k].value or cells[2][k-1].value == -cells[2][k].value:
                value = cells[2][k-1].value
                cells[2][k-1] = cells[2][k]
                cells[2][k-1].value = cells[2][k-1].value + value
                cells[2][k-1].isInSuperposition = False
                cells[2][k-1].isMergable = False
                cells[2][k] = Tile(0)
                determineCircuit(cells[2][k-1])
            else:
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
            k -= 1
    if cells[3][2].value != 0 and cells[3][2].value != "H" and cells[3][2].value != "Z":
        k = 2
        while canReach(cells[3][k], 3, k-1) and cells[3][k].canMove:
            if cells[3][k-1].value == "H":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughHGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == "Z":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughZGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == cells[3][k].value or cells[3][k-1].value == -cells[3][k].value:
                value = cells[3][k-1].value
                cells[3][k-1] = cells[3][k]
                cells[3][k-1].value = cells[3][k-1].value + value
                cells[3][k-1].isInSuperposition = False
                cells[3][k-1].isMergable = False
                cells[3][k] = Tile(0)
                determineCircuit(cells[3][k-1])
            else:
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
            k -= 1
    if cells[0][3].value != 0 and cells[0][3].value != "H" and cells[0][3].value != "Z":
        k = 3
        while canReach(cells[0][k], 0, k-1) and cells[0][k].canMove:
            if cells[0][k-1].value == "H":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughHGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == "Z":
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
                passThroughZGate(cells[0][k-1])
                cells[0][k-1].isMergable = False
                cells[0][k-1].canMove = False
            elif cells[0][k-1].value == cells[0][k].value or cells[0][k-1].value == -cells[0][k].value:
                value = cells[0][k-1].value
                cells[0][k-1] = cells[0][k]
                cells[0][k-1].value = cells[0][k-1].value + value
                cells[0][k-1].isInSuperposition = False
                cells[0][k-1].isMergable = False
                cells[0][k] = Tile(0)
                determineCircuit(cells[0][k-1])
            else:
                cells[0][k-1] = cells[0][k]
                cells[0][k] = Tile(0)
            k -= 1
    if cells[1][3].value != 0 and cells[1][3].value != "H" and cells[1][3].value != "Z":
        k = 3
        while canReach(cells[1][k], 1, k-1) and cells[1][k].canMove:
            if cells[1][k-1].value == "H":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughHGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == "Z":
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
                passThroughZGate(cells[1][k-1])
                cells[1][k-1].isMergable = False
                cells[1][k-1].canMove = False
            elif cells[1][k-1].value == cells[1][k].value or cells[1][k-1].value == -cells[1][k].value:
                value = cells[1][k-1].value
                cells[1][k-1] = cells[1][k]
                cells[1][k-1].value = cells[1][k-1].value + value
                cells[1][k-1].isInSuperposition = False
                cells[1][k-1].isMergable = False
                cells[1][k] = Tile(0)
                determineCircuit(cells[1][k-1])
            else:
                cells[1][k-1] = cells[1][k]
                cells[1][k] = Tile(0)
            k -= 1
    if cells[2][3].value != 0 and cells[2][3].value != "H" and cells[2][3].value != "Z":
        k = 3
        while canReach(cells[2][k], 2, k-1) and cells[2][k].canMove:
            if cells[2][k-1].value == "H":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughHGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == "Z":
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
                passThroughZGate(cells[2][k-1])
                cells[2][k-1].isMergable = False
                cells[2][k-1].canMove = False
            elif cells[2][k-1].value == cells[2][k].value or cells[2][k-1].value == -cells[2][k].value:
                value = cells[2][k-1].value
                cells[2][k-1] = cells[2][k]
                cells[2][k-1].value = cells[2][k-1].value + value
                cells[2][k-1].isInSuperposition = False
                cells[2][k-1].isMergable = False
                cells[2][k] = Tile(0)
                determineCircuit(cells[2][k-1])
            else:
                cells[2][k-1] = cells[2][k]
                cells[2][k] = Tile(0)
            k -= 1
    if cells[3][3].value != 0 and cells[3][3].value != "H" and cells[3][3].value != "Z":
        k = 3
        while canReach(cells[3][k], 3, k-1) and cells[3][k].canMove:
            if cells[3][k-1].value == "H":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughHGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == "Z":
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
                passThroughZGate(cells[3][k-1])
                cells[3][k-1].isMergable = False
                cells[3][k-1].canMove = False
            elif cells[3][k-1].value == cells[3][k].value or cells[3][k-1].value == -cells[3][k].value:
                value = cells[3][k-1].value
                cells[3][k-1] = cells[3][k]
                cells[3][k-1].value = cells[3][k-1].value + value
                cells[3][k-1].isInSuperposition = False
                cells[3][k-1].isMergable = False
                cells[3][k] = Tile(0)
                determineCircuit(cells[3][k-1])
            else:
                cells[3][k-1] = cells[3][k]
                cells[3][k] = Tile(0)
            k -= 1


def moveRight():
    for j in range(3, -1, -1):
        for i in range(3, -1, -1):
            if cells[i][j].value != 0 and cells[i][j].value != "H" and cells[i][j].value != "Z":
                k = j
                while canReach(cells[i][k], i, k+1) and cells[i][k].canMove:
                    if cells[i][k+1].value == "H":
                        cells[i][k+1] = cells[i][k]
                        cells[i][k] = Tile(0)
                        passThroughHGate(cells[i][k+1])
                        cells[i][k+1].isMergable = False
                        cells[i][k+1].canMove = False
                    elif cells[i][k+1].value == "Z":
                        cells[i][k+1] = cells[i][k]
                        cells[i][k] = Tile(0)
                        passThroughZGate(cells[i][k+1])
                        cells[i][k+1].isMergable = False
                        cells[i][k+1].canMove = False
                    elif cells[i][k+1].value == cells[i][k].value or cells[i][k+1].value == -cells[i][k].value:
                        value = cells[i][k+1].value
                        cells[i][k+1] = cells[i][k]
                        cells[i][k+1].value = cells[i][k+1].value + value
                        cells[i][k+1].isInSuperposition = False
                        cells[i][k+1].isMergable = False
                        cells[i][k] = Tile(0)
                        determineCircuit(cells[i][k+1])
                    else:
                        cells[i][k+1] = cells[i][k]
                        cells[i][k] = Tile(0)
                    k += 1


def resetMergables():
    cells[0][0].isMergable = True
    cells[0][1].isMergable = True
    cells[0][2].isMergable = True
    cells[0][3].isMergable = True
    cells[1][0].isMergable = True
    cells[1][1].isMergable = True
    cells[1][2].isMergable = True
    cells[1][3].isMergable = True
    cells[2][0].isMergable = True
    cells[2][1].isMergable = True
    cells[2][2].isMergable = True
    cells[2][3].isMergable = True
    cells[3][0].isMergable = True
    cells[3][1].isMergable = True
    cells[3][2].isMergable = True
    cells[3][3].isMergable = True


def resetMoveRestricts():
    cells[0][0].canMove = True
    cells[0][1].canMove = True
    cells[0][2].canMove = True
    cells[0][3].canMove = True
    cells[1][0].canMove = True
    cells[1][1].canMove = True
    cells[1][2].canMove = True
    cells[1][3].canMove = True
    cells[2][0].canMove = True
    cells[2][1].canMove = True
    cells[2][2].canMove = True
    cells[2][3].canMove = True
    cells[3][0].canMove = True
    cells[3][1].canMove = True
    cells[3][2].canMove = True
    cells[3][3].canMove = True


def listener(event):
    temp.current_move = event.keysym
    window.quit()


def checkWin():
    if cells[0][0].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[0][1].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[0][2].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[0][3].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[1][0].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[1][1].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[1][2].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[1][3].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[2][0].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[2][1].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[2][2].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[2][3].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[3][0].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[3][1].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[3][2].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    if cells[3][3].value == 256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Win')
        return True
    return False


def checkLose():
    if cells[0][0].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[0][1].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[0][2].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[0][3].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[1][0].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[1][1].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[1][2].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[1][3].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[2][0].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[2][1].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[2][2].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[2][3].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[3][0].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[3][1].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[3][2].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    if cells[3][3].value == -256:
        elements = gameArea.winfo_children()
        elements[5].configure(text='You')
        elements[6].configure(text='Lose')
        return True
    return False


def updateWidgets():
    elements = gameArea.winfo_children()
    cell_array = []
    cell_array.append(cells[0][0])
    cell_array.append(cells[0][1])
    cell_array.append(cells[0][2])
    cell_array.append(cells[0][3])
    cell_array.append(cells[1][0])
    cell_array.append(cells[1][1])
    cell_array.append(cells[1][2])
    cell_array.append(cells[1][3])
    cell_array.append(cells[2][0])
    cell_array.append(cells[2][1])
    cell_array.append(cells[2][2])
    cell_array.append(cells[2][3])
    cell_array.append(cells[3][0])
    cell_array.append(cells[3][1])
    cell_array.append(cells[3][2])
    cell_array.append(cells[3][3])
    for i in range(len(elements)):
        if cell_array[i].value != 0:
            elements[i].configure(text=cell_array[i].toString())
            if cell_array[i].value == 2 or cell_array[i].value == -2:
                elements[i].configure(bg='#eee4da')
            elif cell_array[i].value == 4 or cell_array[i].value == -4:
                elements[i].configure(bg='#ede0c8')
            elif cell_array[i].value == 8 or cell_array[i].value == -8:
                elements[i].configure(bg='#f2b179')
            elif cell_array[i].value == 16 or cell_array[i].value == -16:
                elements[i].configure(bg='#f59563')
            elif cell_array[i].value == 32 or cell_array[i].value == -32:
                elements[i].configure(bg='#f67c5f')
            elif cell_array[i].value == 64 or cell_array[i].value == -64:
                elements[i].configure(bg='#edc850')
            elif cell_array[i].value == 128 or cell_array[i].value == -128:
                elements[i].configure(bg='#edc22e')
            elif cell_array[i].value == 256 or cell_array[i].value == -256:
                elements[i].configure(bg='#f65e3b')
            elif cell_array[i].value == "Z":
                elements[i].configure(bg='#8A2BE2')
            elif cell_array[i].value == "H":
                elements[i].configure(bg='#7FFF00')
        else:
            elements[i].configure(text='', bg='#9e948a')


window = Tk()
window.title("2048 Quantum")
window.geometry("512x485+500+50")
window.bind("<KeyPress>", listener)
window.resizable(False, False)
window.focus_force()
gameArea = Frame(window, bg='#92877d')
l = Label(gameArea, text=cells[0][0].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=0, column=0, padx=7, pady=7)
l = Label(gameArea, text=cells[0][1].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=0, column=1, padx=7, pady=7)
l = Label(gameArea, text=cells[0][2].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=0, column=2, padx=7, pady=7)
l = Label(gameArea, text=cells[0][3].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=0, column=3, padx=7, pady=7)
l = Label(gameArea, text=cells[1][0].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=1, column=0, padx=7, pady=7)
l = Label(gameArea, text=cells[1][1].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=1, column=1, padx=7, pady=7)
l = Label(gameArea, text=cells[1][2].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=1, column=2, padx=7, pady=7)
l = Label(gameArea, text=cells[1][3].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=1, column=3, padx=7, pady=7)
l = Label(gameArea, text=cells[2][0].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=2, column=0, padx=7, pady=7)
l = Label(gameArea, text=cells[2][1].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=2, column=1, padx=7, pady=7)
l = Label(gameArea, text=cells[2][2].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=2, column=2, padx=7, pady=7)
l = Label(gameArea, text=cells[2][3].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=2, column=3, padx=7, pady=7)
l = Label(gameArea, text=cells[3][0].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=3, column=0, padx=7, pady=7)
l = Label(gameArea, text=cells[3][1].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=3, column=1, padx=7, pady=7)
l = Label(gameArea, text=cells[3][2].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=3, column=2, padx=7, pady=7)
l = Label(gameArea, text=cells[3][3].toString(), bg='#9e948a',
          font=('arial', 22, 'bold'), width=6, height=3)
l.grid(row=3, column=3, padx=7, pady=7)
gameArea.grid()
updateWidgets()
window.mainloop()


def useless(event):
    a = 4


while True:
    current_move = temp.current_move

    if current_move == "Up":
        moveUp()
        spawn()
        time.sleep(0.1)
        current_move = ""
    elif current_move == "Down":
        moveDown()
        spawn()
        time.sleep(0.1)
        current_move = ""
    elif current_move == "Left":
        moveLeft()
        spawn()
        time.sleep(0.1)
        current_move = ""
    elif current_move == "Right":
        moveRight()
        spawn()
        time.sleep(0.1)
        current_move = ""
    resetMergables()
    resetMoveRestricts()
    updateWidgets()
    if checkWin() or checkLose():
        window.bind("<KeyPress>", useless)
    window.mainloop()
