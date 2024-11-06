# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/scripts/qiskit/Grover_oracle.py
#!/usr/bin/env python
# coding: utf-8

#initialization
import numpy as np
import csv

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute


with open("/home/biped/catkin_ws/src/jacob/scripts/results/face_ident.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            name = row[0]
# print(name)
# name = "Will_Smith"
#encode name 
input_signal = " "
if name == "Jacob_Viertel":
    input_signal = "00"
elif name == "Marek_Perkowski":
    input_signal = "01"
elif name == "Will_Smith":
    input_signal = "10"

#oracle
n = 4
grover_circuit = QuantumCircuit(n)

if input_signal == "00":
    for qubit in range(n-2):
        grover_circuit.x(qubit)
elif input_signal == "01":
    grover_circuit.x(0)
elif input_signal == "10":
    grover_circuit.x(1)

grover_circuit.toffoli(0,1,2)
grover_circuit.x(1)
grover_circuit.toffoli(0,1,3)
grover_circuit.x(1)

grover_circuit.measure_all()

backend = Aer.get_backend('qasm_simulator')
shots = 1
results = execute(grover_circuit, backend=backend, shots=shots).result()
answer = results.get_counts()
print(answer)

output_state = open("/home/biped/catkin_ws/src/jacob/scripts/results/qvision_output.csv", "w")
writer = csv.writer(output_state)
for key, value in answer.items():
    writer.writerows(answer.keys())
output_state.close()