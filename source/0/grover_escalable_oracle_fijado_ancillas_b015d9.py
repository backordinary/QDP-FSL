# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/grover_escalable_oracle_fijado_ancillas.py
from qiskit import QuantumCircuit, execute
from qiskit import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.aer import AerSimulator
import random
import operator
import sys
import time

qubits_number = int(sys.argv[1])

def diffuser(nqubits = qubits_number):
    qc = QuantumCircuit(nqubits)
    for qubit in range(nqubits):
        qc.h(qubit)
    for qubit in range(nqubits):
        qc.x(qubit)
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)
    qc.h(nqubits-1)
    for qubit in range(nqubits):
        qc.x(qubit)
    for qubit in range(nqubits):
        qc.h(qubit)
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s


def grover(n = qubits_number):
    var_qubits = QuantumRegister(n, name='v')
    # output_qubit = QuantumRegister(2, name='out')
    cbits = ClassicalRegister(qubits_number-2, name='cbits')
    
    circ = QuantumCircuit(var_qubits, cbits)
    
    L = []
    for i in range(n): L.append(i)
    
    circ.h(range(n-2))
    cont = 1
    # while cont < (2**(n-2))**(1/2) * 3.1415/4:
    while cont <2:
        circ.barrier(range(n))

        circ.mct(L[:n-2],L[n-2],L[n-1], mode= 'recursion')
        circ.z(L[n-2])
        circ.mct(L[:n-2],L[n-2],L[n-1],mode= 'recursion')

        circ.barrier(range(n))
        circ.append(diffuser(n-2), L[:n-2])
        cont+=1
    
    circ.measure(L[:n-2],L[:n-2])
    
    return circ

g = transpile(grover())

backend= AerSimulator(method="statevector")

h = int(sys.argv[2])

start= time.time()
job = execute(g, backend, shots = 20000, blocking_enable=True, blocking_qubits=h)
end = time.time() - start

result = job.result()

count = result.get_counts(g)

max_key = max(count, key = count.get)
# print(str(max_key) + ' : '+ str(count[max_key]) + '\n')
#print(count)
# print("time_taken: " + str(result.time_taken))
# print(result)
# print('\n-----------------------------------------------------------------------\n')
print(str(qubits_number)+';'+str(h)+';'+str(max_key)+';'+str(count[max_key])+';'+str(end)+';'+str(result.time_taken))