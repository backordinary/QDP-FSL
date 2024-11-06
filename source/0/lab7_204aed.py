# https://github.com/gypark23/CS228/blob/55efe117834ecb4410f9656d294a2b1433f8bfd6/Qiskit%20(Lab%205-8)/Lab7.py
import qiskit
from qiskit.providers.aer import QasmSimulator  

def hw3_1a_response(circuit, qubit1, qubit2):
    # Put your code here (spaces for indentation)
    circuit.h(qubit1)
    circuit.cx(qubit1, qubit2)
    
    return circuit
      
import qiskit
from qiskit.providers.aer import QasmSimulator  

def hw3_1b_response(num_shots):
    # Put your code here (spaces for indentation)
    qr2 = qiskit.QuantumRegister(2)
    circuit = qiskit.QuantumCircuit(qr2)
    circuit.h(qr2[0])
    circuit.cx(qr2[0], qr2[1])
    circuit.measure_all()
    
    simulator = QasmSimulator()
    executed_job = qiskit.execute(circuit, simulator, shots=num_shots)
    
    r = executed_job.result()
    result_dict = r.get_counts(circuit)
    # End Code
    return circuit, result_dict
      

import qiskit
from qiskit.providers.aer import QasmSimulator  

def hw3_1c_response(num_shots):
    # Put your code here (spaces for indentation)
    qr3 = qiskit.QuantumRegister(2)
    circuit = qiskit.QuantumCircuit(qr3)
    circuit.h(qr3[0])
    circuit.h(qr3[1])
    circuit.measure_all()
    
    simulator = QasmSimulator()
    executed_job = qiskit.execute(circuit, simulator, shots=num_shots)
    
    r = executed_job.result()
    result_dict = r.get_counts(circuit)
    # End Code
    return circuit, result_dict
      

import qiskit
from qiskit.providers.aer import QasmSimulator  

def hw3_2_response(circuit):
    # Put your code to find the entangled qubits here
    circuit.measure_all()
    simulator = QasmSimulator()
    executed_job = qiskit.execute(circuit, simulator, shots=1024)
    r = executed_job.result()
    result_dict = r.get_counts(circuit)

    x = 0
    y = 0
    incorrect = 0
    for j in range(0 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[0] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 0
            y = j
    for j in range(1 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[1] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 1
            y = j
    for j in range(2 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[2] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 2
            y = j
    for j in range(3 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[3] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 3
            y = j
    for j in range(4 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[4] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 4
            y = j

    qubit_1 = 5-y
    qubit_2 = 5-x
    # Put your code here (spaces for indentation)
    # End Code

    return qubit_1, qubit_2
      



import qiskit
from qiskit.providers.aer import QasmSimulator  

def prime_circuit(circuit, qubit_list, bitstring):
    for i in range(len(bitstring)):
        if bitstring[i] == '1':
            circuit.x(qubit_list[i])
    
    return circuit

def hw3_3_response(circuit):
    # Put your code to find the entangled qubits here
    # Put your code here (spaces for indentation)
    circuit.measure_all()
    simulator = QasmSimulator()
    executed_job = qiskit.execute(circuit, simulator, shots=1024)
    r = executed_job.result()
    result_dict = r.get_counts(circuit)

    x = -1
    y = -1
    incorrect = 0
    for j in range(0 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[0] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 0
            y = j
    for j in range(1 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[1] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 1
            y = j
    for j in range(2 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[2] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 2
            y = j
    for j in range(3 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[3] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 3
            y = j
    for j in range(4 + 1, 6):
        for key in result_dict:
            incorrect = 0
            if key[4] == key[j]:
                incorrect = 0
            else:
                incorrect = 1
                break
        if incorrect == 1:
            continue
        else:
            x = 4
            y = j

    if x == -1 or y == -1:
        for j in range(0 + 1, 6):
            for key in result_dict:
                incorrect = 0
                if key[0] != key[j]:
                    incorrect = 0
                else:
                    incorrect = 1
                    break
            if incorrect == 1:
                continue
            else:
                x = 0
                y = j 
        for j in range(1 + 1, 6):
            for key in result_dict:
                incorrect = 0
                if key[1] != key[j]:
                    incorrect = 0
                else:
                    incorrect = 1
                    break
            if incorrect == 1:
                continue
            else:
                x = 1
                y = j 
        for j in range(2 + 1, 6):
            for key in result_dict:
                incorrect = 0
                if key[2] != key[j]:
                    incorrect = 0
                else:
                    incorrect = 1
                    break
            if incorrect == 1:
                continue
            else:
                x = 2
                y = j 
        for j in range(3 + 1, 6):
            for key in result_dict:
                incorrect = 0
                if key[3] != key[j]:
                    incorrect = 0
                else:
                    incorrect = 1
                    break
            if incorrect == 1:
                continue
            else:
                x = 3
                y = j 
        for j in range(4 + 1, 6):
            for key in result_dict:
                incorrect = 0
                if key[4] != key[j]:
                    incorrect = 0
                else:
                    incorrect = 1
                    break
            if incorrect == 1:
                continue
            else:
                x = 4
                y = j 

    qubit_1 = 5-y
    qubit_2 = 5-x



    # End Code
    return qubit_1, qubit_2

import qiskit
from qiskit.providers.aer import QasmSimulator  

def hw3_4_response(n: int):
    # Put your code here
    qr2 = qiskit.QuantumRegister(n)
    circuit = qiskit.QuantumCircuit(qr2)
    circuit.h(qr2[0])
    for i in range(1, n):
        circuit.cx(qr2[i-1],qr2[i])
    # End Code

    return circuit
