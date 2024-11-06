# https://github.com/emilehreich/grover-3SAT/blob/c0db7f33c661cf61cf21db4e386bdca67160eef7/grover_3SAT.py
'''
    @file: grover_3SAT.py

    @date: 15/05/2022

    @author: Emile Janho Dit Hreich (309994)
             emile.janhodithreich@epfl.ch

             Arion Zimmerman
             arion.zimmermann@epfl.ch (315007)

    @brief: Mini-Projet quantique 2022

'''
# =============================================================================================
# Libraries

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import *
from qiskit.compiler import transpile, assemble
from qiskit.circuit.library import MCMT
import numpy as np
import time

# =============================================================================================
# Account loading
provider = IBMQ.load_account()
provider = IBMQ.get_provider(group='open')

# =============================================================================================
# Problem instance

# Question 1
def naive_enumeration():
    solutions = []
    x = 0 & 0b1
    y = (0 >> 1) & 0b1
    z = (0 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))
    x = 1 & 0b1
    y = (1 >> 1) & 0b1
    z = (1 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))
    x = 2 & 0b1
    y = (2 >> 1) & 0b1
    z = (2 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))
    x = 3 & 0b1
    y = (3 >> 1) & 0b1
    z = (3 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))
    x = 4 & 0b1
    y = (4 >> 1) & 0b1
    z = (4 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))
    x = 5 & 0b1
    y = (5 >> 1) & 0b1
    z = (5 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))
    x = 6 & 0b1
    y = (6 >> 1) & 0b1
    z = (6 >> 2) & 0b1

    if ((~x | ~y | ~z) &
        (~x | ~y |  z) &
        (~x | y  |  z) &
        (x  | ~y |  z) &
        (x  | y  | ~z) &
        (x  | y  |  z)) == 1:
            solutions.append((x, y, z))

    print(solutions)


# Question 2
def oracle_uf(circuit):
    # N = 8
    # M = 2
    # Define \theta as \arcsin(\sqrt \frac{M}{N})
    # We must iterate K = [\frac{\pi}{4 \theta_0}] times

    # ex1() yields the solutions s_1 = (1, 0, 1) and s_2 (0, 1, 1)

    # Warning: the bits have been inversed to comply with the binary
    # conventions on the results
    circuit.ccx(2, 0, 3)
    circuit.ccx(1, 0, 3)


def reflection(circuit):
    '''
        The reflection operator
    '''
    circuit.h(0)
    circuit.x(0)
    circuit.h(1)
    circuit.x(1)
    circuit.h(2)
    circuit.x(2)
    # ---------------
    circuit.h(2)
    # ---------------
    circuit.barrier(np.arange(0,3))
    # ---------------
    circuit.ccx(0, 1, 2)
    # ---------------
    circuit.barrier(np.arange(0,3))
    # ---------------
    circuit.h(2)
    circuit.x(0)
    circuit.x(1)
    circuit.x(2)


# Question 5
def direct_reflection(circuit):
    '''
        The reflection operator
    '''
    circuit.h(0)
    circuit.x(0)
    circuit.h(1)
    circuit.x(1)
    circuit.h(2)
    circuit.x(2)
    # ---------------
    # ---------------
    circuit.barrier(np.arange(0,3))
    # ---------------
    gate = MCMT('cz', 2, 1)
    circuit += gate
    # ---------------
    # ---------------
    circuit.barrier(np.arange(0,3))
    circuit.x(0)
    circuit.x(1)
    circuit.x(2)


# Question 5
def direct_uf(circuit):
    circuit.cz(1, 0)
    circuit.cz(2, 0)

# =============================================================================================
# helper functions

def gen_hadamard(circuit, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        circuit.h(q)
    return circuit

# Generic functions
def simulate(circuit):
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator)
    result = job.result()
    counts = result.get_counts(circuit)

    print(counts)

    plot_histogram(counts)

def run(circuit):
    machine = provider.get_backend('ibmq_armonq')
    transpiled = transpile(circuit, backend=machine)
    job = machine.run(transpiled)
    retrieved_job = machine.retrieve_job(job.job_id())

    status = machine.status()
    print("Operational: " + str(status.operational))
    print("Pending jobs: " + str(status.pending_jobs))


    result = retrieved_job.result()
    counts = result.get_counts(circuit)

    print(counts)

    plot_histogram(counts)



# =============================================================================================
# Main


# ----------------------------------------
# Q1: naive enumeration
naive_enumeration()

# ----------------------------------------
# circuit implementation

direct_mode = True
simulation_only = True

qubits  = 3

if direct_mode:
    nb_bits = 3
else:
    nb_bits = 4


# initialize circuit
circuit = QuantumCircuit(nb_bits, qubits)

# set ancilla qubits to 1
for i in range (qubits, nb_bits):
    circuit.x(i)

# initial state
gen_hadamard(circuit, np.arange(0, nb_bits))
# ---------------
circuit.barrier(np.arange(0, nb_bits))
# ---------------

# add oracle U_f

if direct_mode:
    direct_uf(circuit)
else:
    oracle_uf(circuit)

# ---------------
circuit.barrier(np.arange(0, nb_bits))
# ---------------

# add reflector operator
if direct_mode:
    direct_reflection(circuit)
else:
    reflection(circuit)

# ---------------
circuit.barrier(np.arange(0, nb_bits))
# ---------------

for i in range (0, qubits):
    circuit.h(i)

# measure
circuit.measure(np.arange(0, qubits), np.arange(0, qubits))

simulate(circuit)

if not simulation_only:
    run(circuit)
