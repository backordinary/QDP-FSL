# https://github.com/Tomev/QEL/blob/d715293d6d2924ef095de8f0cfb1957985b9ae2f/Simulations/simulatorWithMappingCNOTTest.py
import os
import sys

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import consts
from methods import test_locally

qr = QuantumRegister(5)
cr = ClassicalRegister(5)

print("Simulator with mapping - CNOT Test")
print("Selected backend: " + consts.CONSIDERED_REMOTE_BACKENDS[0])

# Create all pairs of cnots
circuits = []

# Continue if operation would be on one qubit.
if 0 == 0:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(0) + ", Target: " + str(0)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[0], qr[0])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 0 == 1:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(0) + ", Target: " + str(1)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[0], qr[1])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 0 == 2:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(0) + ", Target: " + str(2)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[0], qr[2])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 0 == 3:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(0) + ", Target: " + str(3)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[0], qr[3])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 0 == 4:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(0) + ", Target: " + str(4)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[0], qr[4])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 1 == 0:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(1) + ", Target: " + str(0)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[1], qr[0])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 1 == 1:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(1) + ", Target: " + str(1)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[1], qr[1])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 1 == 2:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(1) + ", Target: " + str(2)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[1], qr[2])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 1 == 3:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(1) + ", Target: " + str(3)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[1], qr[3])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 1 == 4:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(1) + ", Target: " + str(4)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[1], qr[4])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 2 == 0:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(2) + ", Target: " + str(0)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[2], qr[0])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 2 == 1:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(2) + ", Target: " + str(1)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[2], qr[1])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 2 == 2:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(2) + ", Target: " + str(2)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[2], qr[2])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 2 == 3:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(2) + ", Target: " + str(3)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[2], qr[3])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 2 == 4:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(2) + ", Target: " + str(4)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[2], qr[4])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 3 == 0:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(3) + ", Target: " + str(0)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[3], qr[0])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 3 == 1:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(3) + ", Target: " + str(1)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[3], qr[1])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 3 == 2:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(3) + ", Target: " + str(2)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[3], qr[2])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 3 == 3:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(3) + ", Target: " + str(3)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[3], qr[3])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 3 == 4:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(3) + ", Target: " + str(4)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[3], qr[4])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 4 == 0:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(4) + ", Target: " + str(0)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[4], qr[0])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 4 == 1:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(4) + ", Target: " + str(1)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[4], qr[1])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 4 == 2:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(4) + ", Target: " + str(2)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[4], qr[2])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 4 == 3:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(4) + ", Target: " + str(3)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[4], qr[3])
circuit.measure(qr, cr)
circuits.append(circuit)

# Continue if operation would be on one qubit.
if 4 == 4:
    continue

circuit = QuantumCircuit(qr, cr)

# Create state 11111 so that effect is visible.
circuit.x(qr[0])
circuit.x(qr[1])
circuit.x(qr[2])
circuit.x(qr[3])
circuit.x(qr[4])

circuit_name = "Control: " + str(4) + ", Target: " + str(4)
# print(circuit_name)
circuit.name = circuit_name
circuit.cx(qr[4], qr[4])
circuit.measure(qr, cr)
circuits.append(circuit)

print("Created " + str(len(circuits)) + " circuits.")

test_locally(circuits)
# test_locally_with_noise(circuits, True)
# run_main_loop_with_chsh_test(circuits)
