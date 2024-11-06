# https://github.com/oxarbitrage/qiskit-scripts/blob/118331495f66067de4f5111a3581afc18949c57f/qft.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:55:20 2021

@author: oxarbitrage
"""

from numpy import pi
from bitarray import bitarray
from bitarray.util import int2ba

from qiskit import QuantumCircuit, assemble, Aer

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)
    
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates

def prepare_computational_basis(nqubits, number):
    """Encode number as binary and create the circuit"""
    # Create empty circuit of requested size
    qc = QuantumCircuit(nqubits)
    
    # encode number into a byte array 
    ba = int2ba(number, nqubits); 
    
    # Populate the circuit
    for q in range(nqubits):
        if ba[q]:
            qc.x(q)

    return qc

def prepare_fourier_basis(nqubits, number):
    """Given a number create the Fourier circuit for it"""
    # Create empty circuit of requested size
    qc = QuantumCircuit(nqubits)
    
    # Add hadamard gate to all qubits
    for qubit in range(nqubits):
        qc.h(qubit)
    
    # Rotate each qubit by an angle
    qc.p(number*pi/4,0)
    qc.p(number*pi/2,1)
    qc.p(number*pi,2)

    return qc

"""Test QFT in 3 qubits"""
nqubits = 3
sim = Aer.get_backend("aer_simulator")

print("========================")
print("Decimal number: ", 0)
binary_number = int2ba(0, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 0)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 0)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 0)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 1)
binary_number = int2ba(1, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 1)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 1)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 1)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 2)
binary_number = int2ba(2, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 2)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 2)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 2)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 3)
binary_number = int2ba(3, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 3)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 3)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 3)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 4)
binary_number = int2ba(4, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 4)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 4)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 4)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 5)
binary_number = int2ba(5, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 5)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 5)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 5)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 6)
binary_number = int2ba(6, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 6)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 6)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 6)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)

print("========================")
print("Decimal number: ", 7)
binary_number = int2ba(7, nqubits)
print("Binary number: ", binary_number)

# prepare the state given a number        
qc = prepare_computational_basis(nqubits, 7)
# apply quantum fourier transformation to basis state
qft(qc, nqubits)

# simulate the state after QFT is done
qc.save_statevector()
qobj = assemble(qc)
results = sim.run(qobj).result().get_counts()
print("State after applying QFT: ", results)

# encode the fourier state manually
qc = prepare_fourier_basis(nqubits, 7)

qc.save_statevector()
qobj = assemble(qc)
results2 = sim.run(qobj).result().get_counts()
print("State after manually encoding: ", results2)

# make sure manual and qft function produce the
# same results
assert(results == results2)

# prepare manually again
qc = prepare_fourier_basis(nqubits, 7)

# apply inverse qft
qc = inverse_qft(qc, nqubits)

# simulate and get results
qc.save_statevector()
qobj = assemble(qc)
results_inv = sim.run(qobj).result().get_counts()
print("State after inverse QFT: ", results_inv)
response = bitarray(list(results_inv.keys())[0])

# the initial number should be the same as 
# the last result 
assert(binary_number == response)
