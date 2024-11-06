# https://github.com/Alvaradom08/reporte-final-/blob/6c013c8a25d408bf001b10c293608070fa56419c/ejerciciosMatrices.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def Matriz(m):
    for i in m:
        print(" ".join(list(map(str, i))))


def decimalabinario(num):
    cont = 0
    res = 0
    while cont < len(num):
        res += int(num[-1 - cont]) * (2 ** cont)
        cont += 1
    return res


simulator = Aer.get_backend('qasm_simulator')

# Constante

print("Funcion 1")
mtrx = [[0 for k in range(2 ** (5))] for l in range(2 ** (5))]

cont = 0
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx[decimalabinario(z)][cont] = 1
cont += 1
Matriz(mtrx)

# Balanciadas
print("Funcion 2")
mtrx1 = [[0 for k in range(2 ** (5))] for l in range(2 ** (5))]

cont = 0
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(0, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx1[decimalabinario(z)][cont] = 1
cont += 1
Matriz(mtrx1)

# Balanciadas
print("Funcion 3")
mtrx2 = [[0 for k in range(2 ** (5))] for l in range(2 ** (5))]

cont = 0
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(1, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx2[decimalabinario(z)][cont] = 1
cont += 1
Matriz(mtrx2)

# Balanciadas
print("Funcion 4")
mtrx3 = [[0 for k in range(2 ** (5))] for l in range(2 ** (5))]

cont = 0
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 0 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 0 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 0 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 0 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 0 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
circuit = QuantumCircuit(5, 5)
if 1 == 1:
    circuit.x(0)
if 1 == 1:
    circuit.x(1)
if 1 == 1:
    circuit.x(2)
if 1 == 1:
    circuit.x(3)
if 1 == 1:
    circuit.x(4)
circuit.barrier()
circuit.cnot(2, 4)
circuit.barrier()
circuit.measure([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
circuit.barrier()

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(circuit)

for z in counts:
    mtrx3[decimalabinario(z)][cont] = 1
cont += 1
Matriz(mtrx3)
