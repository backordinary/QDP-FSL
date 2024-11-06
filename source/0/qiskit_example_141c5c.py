# https://github.com/lazkmet/Senior-Project-1/blob/bc81a7b085039430b6196958b2082cf7000c0585/Qiskit/Example_Circuits/Qiskit_Example.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import QasmSimulator
from qiskit.visualization import circuit_drawer

#Use qiskit_aer for simulation
simulator = QasmSimulator()

#Create a quantum circuit using 2 qubits and 2 classical bits
circuit = QuantumCircuit(QuantumRegister(2, 'q'), ClassicalRegister(2, 'c'))

#Apply a Hadamard gate to qubit 0
circuit.h(0)

#Apply a NOT gate to qubit 1
circuit.x(1)

#Apply a Hadamard gate to qubit 1
circuit.h(1)

#Apply a Conditional-NOT gate to qubit 1, with qubit 0 as the control
circuit.cnot(0, 1)

#Apply a Hadamard gate to qubit 1
circuit.h(1)

#Measure the value of qubit 0, storing it in the classical register at index 0
circuit.measure(0, 0)
#Measure the value of qubit 1, storing it in the classical register at index 1
circuit.measure(1, 1)

#Simulate this circuit 1000 times using QasmSimulator
simulation = simulator.run(circuit, shots=1000)

#Get results from the simulation
results = simulation.result()

#Output results and display a drawing of the circuit
print("The results of the circuit are: ", results.get_counts(circuit))
circuit_text = circuit_drawer(circuit, 1, None, 'text')
print(circuit_text)