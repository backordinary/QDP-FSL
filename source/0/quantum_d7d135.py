# https://github.com/enatheme/Qube/blob/d794fd2376b0432c95c7975b8b5ca1eb17dfbb83/src/quantum.py
from qiskit import *

from qiskit import QuantumCircuit, QuantumRegister
import math
from qiskit.visualization import plot_histogram

from quantum_gates import *


def generate_floor_v0(number_line):
    """ version 0 of generation
    simple H gate on all qubits, generates 'flying' floor blocks
    """
    qr = QuantumRegister(number_line)
    cr = ClassicalRegister(number_line)

    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr)

    circuit.barrier()
    circuit.measure(qr, cr)

    qasm_sim = Aer.get_backend('qasm_simulator')
    qobj = assemble(circuit, qasm_sim)

    arr = []
    for j in range(20):
        line = []
        results = qasm_sim.run(qobj, shots = 1).result()
        for i in results.get_counts():
            for k in i:
                line.append(int(k))
            arr.append(line)
    return arr

def generate_floor_v1(number_line):
    """ version 1 of generation
    h gate is applied only if the block under is a floor, prevents 'flying' floor blocks
    """
    qr = QuantumRegister(number_line)
    cr = ClassicalRegister(number_line)

    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])

    for i in range(number_line - 1):
        circuit.unitary(C2HGate(), [i, i + 1], label='ch')
    circuit.barrier()
    circuit.measure(qr, cr)

    qasm_sim = Aer.get_backend('qasm_simulator')
    qobj = assemble(circuit, qasm_sim)

    arr = []
    for j in range(20):
        line = []
        results = qasm_sim.run(qobj, shots = 1).result()
        for i in results.get_counts():
            for k in i:
                line.append(int(k))
            arr.append(line)
    return arr

def generate_floor_v2(number_line, initial_column, min_x = 2, max_x = 2):
    """ version 2 of generation
    h gate is applied only if the block under is a floor and if it doesn't create a 'big step'
    with previous column
    """
    assert(number_line == len(initial_column))

    arr = []
    arr.append(initial_column)

    initial_colum = reversed(initial_column)

    for j in range(1, 20):
        initial_qr = QuantumRegister(number_line)
        qr = QuantumRegister(number_line)
        cr = ClassicalRegister(number_line)
        circuit = QuantumCircuit(initial_qr, qr, cr)

        for i, ii in enumerate(initial_column):
            if ii == 1:
                circuit.x(initial_qr[i])

        circuit.barrier()
        first_block = True

        for i in reversed(range(number_line)):
            if first_block:
                first_block = False
                circuit.unitary(C2HGateNot(), [i - max_x, i + number_line], label='C2HGate')
            elif i < min_x:
                circuit.unitary(C3HGateAnd(), [i + number_line + 1, i + min_x, i + number_line], label='C3HGateAnd')
            elif i > number_line - max_x - 1:
                circuit.unitary(C3HGateAndNot(), [i + number_line + 1, i - max_x, i + number_line], label='C3HGateAndNot')
            else:
                circuit.unitary(C4HGate(), [i + number_line + 1, i + min_x, i - max_x, i + number_line], label='C4HGate')
        circuit.measure(qr, cr)

        qasm_sim = Aer.get_backend('qasm_simulator')
        qobj = assemble(circuit, qasm_sim)

        line = []
        results = qasm_sim.run(qobj, shots = 1).result()
        for i in results.get_counts():
            for k in reversed(i):
                line.append(int(k))
            initial_column = line
            arr.append(line)
    return arr

def generate_floor_v3(number_line, initial_column, min_x = 2, max_x = 2):
    """ version 3 of generation
    now let's have an equal probability for possible states for all levels.
    """
    assert(number_line == len(initial_column))
    print_debug = False

    arr = []
    arr.append(initial_column)

    initial_colum = reversed(initial_column)

    for j in range(1, 20):
        initial_qr = QuantumRegister(number_line)
        qr = QuantumRegister(number_line)
        cr = ClassicalRegister(number_line)
        circuit = QuantumCircuit(initial_qr, qr, cr)

        debug_arr = []

        for i, ii in enumerate(initial_column):
            if ii == 1:
                circuit.x(initial_qr[i])

        circuit.barrier()
        first_block = True

        for i in reversed(range(number_line)):
            if first_block:
                first_block = False
                circuit.unitary(C2HGateNot(i, number_line), [i - max_x, i + number_line], label='C2HGate')
                debug_arr.append(f"\tC2HGateNot( {i}, {number_line}) __ [{i - max_x}, {i + number_line}]")
            elif i < min_x:
                circuit.unitary(C3HGateAnd(i, number_line), [i + number_line + 1, i + min_x, i + number_line], label='C3HGateAnd')
                debug_arr.append(f"\tC3HGateAnd( {i}, {number_line}) __ [{i + number_line + 1}, {i + min_x}, {i + number_line}]")
            elif i > number_line - max_x - 1:
                circuit.unitary(C3HGateAndNot(i, number_line), [i + number_line + 1, i - max_x, i + number_line], label='C3HGateAndNot')
                debug_arr.append(f"\tC3HGateAndNot( {i}, {number_line}) __ [{i + number_line + 1}, {i - max_x}, {i + number_line}]")
            else:
                circuit.unitary(C4HGate(i, number_line), [i + number_line + 1, i + min_x, i - max_x, i + number_line], label='C4HGate')
                debug_arr.append(f"\tC4HGate( {i}, {number_line}) __ [{i + number_line + 1}, {i + min_x}, {i - max_x}, {i + number_line}]")
        circuit.measure(qr, cr)

        qasm_sim = Aer.get_backend('qasm_simulator')
        qobj = assemble(circuit, qasm_sim)

        line = []
        rerun = True
        r = None
        while rerun:
            results = qasm_sim.run(qobj, shots = 24).result()
            if print_debug:
                print(results.get_counts())
            try:
                # very ugly way to handle the most frequent len > 1
                r = results.get_counts().most_frequent()
                rerun = False
            except qiskit.exceptions.QiskitError:
                r = None
        for i in reversed(r):
            line.append(int(i))
        initial_column = line
        if print_debug:
            print(f"|> {line}")
            for debug in debug_arr:
                print(debug)
        arr.append(line)
        debug_arr = []
    return arr
