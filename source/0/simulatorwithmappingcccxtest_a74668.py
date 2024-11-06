# https://github.com/Tomev/QEL/blob/d715293d6d2924ef095de8f0cfb1957985b9ae2f/Simulations/simulatorWithMappingCCCXTest.py
import os
import sys

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

import consts

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from methods import test_locally


def ccnot(control1, control2, target):
    quantum_circuit = QuantumCircuit(qr, cr)
    quantum_circuit.ccx(qr[control1], qr[control2], qr[target])
    return quantum_circuit


def rtof3(control1, control2, target):
    rtof = QuantumCircuit(qr, cr)
    rtof.h(qr[target])
    rtof.t(qr[target])
    rtof.cx(qr[control2], qr[target])
    rtof.tdg(qr[target])
    rtof.cx(qr[control1], qr[target])
    rtof.t(qr[target])
    rtof.cx(qr[control2], qr[target])
    rtof.tdg(qr[target])
    rtof.h(qr[target])
    return rtof


def rtof4(c1, c2, c3, t):
    a = 2  # ancilla

    rtof = QuantumCircuit(qr, cr)
    rtof += rtof3(c1, c2, a)
    rtof += ccnot(a, c3, t)
    rtof += rtof3(c1, c2, a)

    return rtof


qr = QuantumRegister(5)
cr = ClassicalRegister(5)

print("Simulator with mapping - Reduced Toffoli 4 Test")
print("Selected backend: " + consts.CONSIDERED_REMOTE_BACKENDS[0])

# Create all pairs of cccnots
circuits = []

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 0, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 0, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 0, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 0, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 0, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 0, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 0, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 0, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 0, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 0, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 0, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 0, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 0, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 0, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 0, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 0, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 0, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 0, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 0, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 0, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 0, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 0, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 0, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 0, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(0)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 0, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 1, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 1, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 1, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 1, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 1, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 1, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 1, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 1, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 1, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 1, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 1, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 1, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 1, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 1, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 1, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 1, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 1, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 1, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 1, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 1, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 1, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 1, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 1, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 1, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(1)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 1, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 2, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 2, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 2, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 2, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 2, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 2, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 2, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 2, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 2, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 2, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 2, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 2, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 2, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 2, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 2, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 2, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 2, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 2, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 2, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 2, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 2, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 2, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 2, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 2, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(2)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 2, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 3, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 3, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 3, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 3, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 3, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 3, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 3, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 3, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 3, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 3, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 3, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 3, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 3, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 3, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 3, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 3, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 3, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 3, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 3, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 3, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 3, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 3, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 3, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 3, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(3)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 3, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 4, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 4, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 4, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 4, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 4, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 4, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 4, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 4, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 4, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 4, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 4, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 4, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 4, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 4, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 4, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 4, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 4, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 4, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 4, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 4, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(0, 4, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(0, 4, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(0, 4, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(0, 4, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(0)
indices.add(4)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(0) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(0, 4, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 0, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 0, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 0, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 0, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 0, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 0, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 0, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 0, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 0, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 0, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 0, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 0, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 0, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 0, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 0, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 0, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 0, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 0, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 0, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 0, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 0, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 0, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 0, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 0, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(0)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 0, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 1, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 1, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 1, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 1, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 1, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 1, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 1, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 1, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 1, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 1, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 1, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 1, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 1, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 1, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 1, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 1, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 1, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 1, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 1, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 1, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 1, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 1, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 1, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 1, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(1)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 1, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 2, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 2, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 2, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 2, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 2, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 2, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 2, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 2, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 2, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 2, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 2, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 2, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 2, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 2, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 2, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 2, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 2, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 2, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 2, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 2, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 2, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 2, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 2, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 2, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(2)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 2, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 3, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 3, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 3, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 3, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 3, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 3, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 3, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 3, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 3, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 3, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 3, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 3, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 3, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 3, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 3, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 3, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 3, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 3, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 3, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 3, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 3, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 3, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 3, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 3, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(3)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 3, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 4, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 4, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 4, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 4, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 4, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 4, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 4, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 4, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 4, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 4, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 4, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 4, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 4, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 4, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 4, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 4, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 4, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 4, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 4, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 4, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(1, 4, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(1, 4, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(1, 4, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(1, 4, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(1)
indices.add(4)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(1) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(1, 4, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 0, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 0, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 0, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 0, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 0, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 0, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 0, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 0, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 0, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 0, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 0, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 0, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 0, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 0, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 0, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 0, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 0, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 0, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 0, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 0, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 0, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 0, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 0, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 0, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(0)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 0, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 1, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 1, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 1, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 1, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 1, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 1, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 1, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 1, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 1, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 1, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 1, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 1, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 1, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 1, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 1, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 1, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 1, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 1, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 1, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 1, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 1, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 1, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 1, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 1, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(1)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 1, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 2, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 2, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 2, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 2, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 2, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 2, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 2, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 2, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 2, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 2, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 2, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 2, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 2, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 2, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 2, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 2, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 2, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 2, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 2, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 2, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 2, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 2, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 2, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 2, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(2)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 2, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 3, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 3, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 3, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 3, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 3, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 3, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 3, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 3, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 3, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 3, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 3, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 3, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 3, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 3, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 3, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 3, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 3, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 3, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 3, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 3, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 3, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 3, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 3, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 3, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(3)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 3, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 4, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 4, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 4, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 4, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 4, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 4, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 4, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 4, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 4, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 4, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 4, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 4, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 4, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 4, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 4, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 4, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 4, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 4, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 4, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 4, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(2, 4, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(2, 4, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(2, 4, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(2, 4, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(2)
indices.add(4)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(2) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(2, 4, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 0, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 0, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 0, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 0, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 0, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 0, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 0, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 0, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 0, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 0, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 0, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 0, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 0, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 0, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 0, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 0, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 0, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 0, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 0, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 0, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 0, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 0, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 0, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 0, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(0)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 0, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 1, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 1, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 1, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 1, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 1, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 1, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 1, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 1, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 1, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 1, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 1, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 1, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 1, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 1, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 1, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 1, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 1, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 1, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 1, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 1, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 1, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 1, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 1, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 1, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(1)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 1, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 2, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 2, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 2, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 2, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 2, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 2, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 2, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 2, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 2, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 2, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 2, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 2, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 2, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 2, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 2, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 2, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 2, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 2, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 2, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 2, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 2, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 2, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 2, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 2, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(2)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 2, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 3, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 3, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 3, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 3, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 3, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 3, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 3, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 3, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 3, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 3, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 3, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 3, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 3, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 3, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 3, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 3, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 3, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 3, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 3, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 3, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 3, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 3, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 3, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 3, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(3)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 3, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 4, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 4, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 4, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 4, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 4, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 4, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 4, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 4, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 4, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 4, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 4, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 4, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 4, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 4, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 4, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 4, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 4, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 4, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 4, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 4, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(3, 4, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(3, 4, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(3, 4, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(3, 4, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(3)
indices.add(4)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(3) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(3, 4, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 0, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 0, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 0, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 0, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 0, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 0, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 0, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 0, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 0, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 0, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 0, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 0, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 0, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 0, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 0, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 0, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 0, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 0, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 0, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 0, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 0, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 0, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 0, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 0, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(0)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(0) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 0, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 1, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 1, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 1, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 1, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 1, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 1, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 1, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 1, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 1, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 1, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 1, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 1, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 1, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 1, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 1, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 1, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 1, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 1, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 1, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 1, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 1, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 1, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 1, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 1, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(1)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(1) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 1, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 2, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 2, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 2, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 2, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 2, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 2, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 2, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 2, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 2, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 2, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 2, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 2, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 2, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 2, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 2, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 2, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 2, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 2, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 2, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 2, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 2, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 2, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 2, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 2, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(2)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(2) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 2, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 3, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 3, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 3, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 3, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 3, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 3, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 3, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 3, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 3, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 3, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 3, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 3, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 3, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 3, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 3, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 3, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 3, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 3, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 3, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 3, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 3, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 3, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 3, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 3, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(3)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(3) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 3, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(0)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 4, 0, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(0)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 4, 0, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(0)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 4, 0, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(0)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 4, 0, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(0)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    0) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 4, 0, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(1)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 4, 1, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(1)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 4, 1, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(1)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 4, 1, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(1)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 4, 1, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(1)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    1) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 4, 1, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(2)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 4, 2, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(2)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 4, 2, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(2)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 4, 2, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(2)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 4, 2, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(2)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    2) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 4, 2, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(3)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 4, 3, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(3)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 4, 3, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(3)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 4, 3, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(3)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 4, 3, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(3)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    3) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 4, 3, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(4)
indices.add(0)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(0)
# print(circuit_name)
circuit += rtof4(4, 4, 4, 0)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(4)
indices.add(1)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(1)
# print(circuit_name)
circuit += rtof4(4, 4, 4, 1)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(4)
indices.add(2)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(2)
# print(circuit_name)
circuit += rtof4(4, 4, 4, 2)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(4)
indices.add(3)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(3)
# print(circuit_name)
circuit += rtof4(4, 4, 4, 3)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if there aren't 4 different indices.
indices = set()
indices.add(4)
indices.add(4)
indices.add(4)
indices.add(4)

if len(indices) != 4:
    continue

if indices.__contains__(2):
    continue  # contains ancilla

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
# circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control1: " + str(4) + ", Control2: " + str(4) + ", Control3: " + str(
    4) + ", Target: " + str(4)
# print(circuit_name)
circuit += rtof4(4, 4, 4, 4)
circuit.name = circuit_name

circuit.measure(qr, cr)
circuits.append(circuit)

print("Created " + str(len(circuits)) + " circuits.")

test_locally(circuits, True)
