# https://github.com/vm6502q/simulator-benchmarks/blob/d4f765ebe6d5528f3ab247d7ecde06b2862d39e7/qiskit_qrack/qiskit_qrack_random_circuit.py
#Adapted from https://github.com/libtangle/qcgpu/blob/master/benchmark/benchmark.py by Adam Kelly

import click
import time
import random
import csv
import os.path
import math

from qiskit import QuantumCircuit
from qiskit import execute
from qiskit.providers.qrack import QasmSimulator

# Implementation of random universal circuit
def rand_circuit(num_qubits, depth, circ):
    single_bit_gates = circ.h, circ.x, circ.y, circ.z, circ.t
    multi_bit_gates = circ.swap, circ.cx, circ.cz, circ.ccx

    for i in range(depth):
        # Single bit gates
        for j in range(num_qubits):
            gate = random.choice(single_bit_gates)
            gate(j)

        # Multi bit gates
        bit_set = [i for i in range(num_qubits)]
        while len(bit_set) > 1:
            b1 = random.choice(bit_set)
            bit_set.remove(b1)
            b2 = random.choice(bit_set)
            bit_set.remove(b2)
            gate = random.choice(multi_bit_gates)
            while len(bit_set) == 0 and gate == circ.ccx:
                gate = random.choice(multi_bit_gates)
            if gate == circ.ccx:
                b3 = random.choice(bit_set)
                bit_set.remove(b3)
                gate(b1, b2, b3)
            else:
                gate(b1, b2)

    for j in range(num_qubits):
        circ.measure(j, j)

    return circ

sim_backend = QasmSimulator(shots=1)

def bench(num_qubits, depth):
    circ = QuantumCircuit(num_qubits, num_qubits)
    rand_circuit(num_qubits, depth, circ)
    start = time.time()
    job = execute([circ], sim_backend, timeout=600)
    result = job.result()
    return time.time() - start

# Reporting
def create_csv(filename):
    file_exists = os.path.isfile(filename)
    csvfile = open(filename, 'a')
   
    headers = ['name', 'num_qubits', 'depth', 'time']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

    if not file_exists:
        writer.writeheader()  # file doesn't exist yet, write a header

    return writer

def write_csv(writer, data):
    writer.writerow(data)



@click.command()
@click.option('--samples', default=100, help='Number of samples to take for each qubit.')
@click.option('--qubits', default=28, help='How many qubits you want to test for')
@click.option('--depth', default=20, help='How large a circuit depth you want to test for')
@click.option('--out', default='benchmark_data.csv', help='Where to store the CSV output of each test')
@click.option('--single', default=False, help='Only run the benchmark for a single amount of qubits, and print an analysis')
def benchmark(samples, qubits, depth, out, single):
    if single:
        low = qubits - 1
    else:
        low = 3
    high = qubits

    functions = bench,
    writer = create_csv(out)

    for n in range(low, high):
        for d in [4, 9, 14, 19]:
            # Progress counter
            progress = (((n - low) * depth) + d) / ((high - low) * depth)
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50), progress*100), end="", flush=True)

            # Run the benchmarks
            for i in range(samples):
                func = random.choice(functions)
                t = func(n+1, d+1)
                write_csv(writer, {'name': 'qiskit_random', 'num_qubits': n+1, 'depth': d+1, 'time': t})

if __name__ == '__main__':
    benchmark()
