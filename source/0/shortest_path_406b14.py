# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/Shortest_path.py
import numpy as np
import math

# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer

# import basic plot tools
from qiskit.visualization import plot_histogram

dreg = QuantumRegister(1, 'd')
nreg = QuantumRegister(2, 'n')
creg = ClassicalRegister(2, 'c')
circ = QuantumCircuit(dreg, nreg, creg)
found_path = False
start_node = "00"
end_node = "11"
vistied_nodes =[]

i = 1
while found_path == False:
    circ.h(0)
    circ.cx(0,1)
    circ.x(0)
    circ.cx(0,2)
    circ.x(0)
    # Execute and see results
    circ.measure(nreg[:2],creg)
    backend = BasicAer.get_backend('qasm_simulator')
    shots = 1
    results = execute(circ, backend=backend, shots=shots).result()
    answer = results.get_counts()
    found_path = False
    for key, value in answer.items():
        vistied_nodes.append(key)
        print("Currently in iteration no. ", i, "We are at node ", key)
        if key == "11":
            found_path=True
    i = i+1
print("we found the path", vistied_nodes)
