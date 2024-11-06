# https://github.com/PetkaRedka/Kvanti/blob/76050e2025a6e45d66d84054fa1bb438c240a42c/Simon.py
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister

S = '100'
a, clone_a = QuantumRegister(len(S), name = "a"), QuantumRegister(len(S), name = "clone_a")
cr1 = ClassicalRegister(len(S))
Simon_circuit = QuantumCircuit(a, clone_a, cr1)


simulator = Aer.get_backend('qasm_simulator')

#---------- Оракул -----------
for i in range(len(S)):
	Simon_circuit.h(a[i])      
Simon_circuit.barrier()

for i in range(len(S)):
 	Simon_circuit.cx(a[i], clone_a[i]) 


S = S[::-1]
n = 0
for i in range(len(S)):
	if S[i] == '1':
		n = i

for i in range(len(S)):
	if S[i] == '1':
		Simon_circuit.cx(a[n], clone_a[i])

Simon_circuit.barrier()


for i in range(len(S)):
	Simon_circuit.h(a[i]) 

Simon_circuit.barrier()

for i in range(len(S)):
	Simon_circuit.measure(a[i], cr1[i])



job = execute(Simon_circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(Simon_circuit)
print(counts)

print(Simon_circuit.draw())

def Output(S, z):
    accum = 0
    for i in range(len(S)):
        accum += int(S[i]) * int(z[i])
    return (accum % 2)

for z in counts:
    print( '{} * {} ≡ {} (mod 2)'.format(S, z, Output(S,z)) )