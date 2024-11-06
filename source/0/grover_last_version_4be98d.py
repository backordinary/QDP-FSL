# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/grover_last_version.py
from qiskit import QuantumCircuit, execute, ClassicalRegister, QuantumRegister, transpile
from qiskit.providers.aer import AerSimulator
import sys

n=int(sys.argv[1])
Q = []
for i in range(n): Q.append(i)

var_qubits = QuantumRegister(n, name='v')
cbits = ClassicalRegister(n-2, name='cbits')

qc = QuantumCircuit(var_qubits,cbits)

qc.h(range(n-2))

cont = 1
c = 2
while cont < c:
    qc.mct(Q[:n-2],n-2,ancilla_qubits=n-1,mode='recursion')
    qc.z(n-2)
    qc.mct(Q[:n-2],n-2,ancilla_qubits=n-1,mode='recursion')

    qc.barrier()

    qc.h(range(n-2))

    qc.x(range(n-2))

    qc.h(n-3)

    qc.mct(Q[:n-3],n-3,ancilla_qubits=n-1,mode='recursion')

    qc.h(n-3)

    qc.x(range(n-2))

    qc.h(range(n-2))
    
    cont+=1

qc.measure(Q[:n-2],Q[:n-2])

backend= AerSimulator(method="statevector")

bang = 20000
job = execute(transpile(qc), backend, shots = bang, blocking_enable=True, blocking_qubits=int(sys.argv[2]))

result = job.result()

count = result.get_counts(qc)

max_key = max(count, key = count.get)

s = ''
for i in range(n-2): s+='1'

# print(count)
# print(result)

print("\n-----------------------------------------------------------------------")
print('Cycles : ' + str(c-1))
print('Oracle -> |'+ s + '>')
print('Result -> |'+ str(max_key) + '> : '+ str(count[max_key])+' counts')
print('Number of shots -> '+ str(bang))
print('Execution time : ' + str(result.time_taken) +' seconds')