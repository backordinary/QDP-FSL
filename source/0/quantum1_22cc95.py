# https://github.com/CollegePaul/python/blob/206668e235c54278d82a8fc4e5115deb97e67745/Quantum/quantum1.py
import qiskit as qk

#make a qubits and normal bits
qb = qk.QuantumRegister(2)      #2 qubits
cb = qk.ClassicalRegister(2)    #2 normal bits

#make a circuit
circuit = qk.QuantumCircuit(qb,cb)  #add the bits to the circuit


#add a Hadamard gate - takes a single qbit, and will output a qbit with 50% proability of 1 or 0
circuit.h(qb[0])

#CNOT on the fisrt and second qbits - controlled NOT, takes 2 qubits, and fips the the second from 0 to 1, if fist is 1
circuit.cx(qb[0], qb[1])


#measure the qubits
circuit.measure(qb, cb)

#print out the result
print(circuit)


#use Qiskit Aer's Qasm Simulator
simulator = qk.BasicAer.get_backend('qasm_simulator')

#Simulate
job = qk.execute(circuit, simulator)
result = job.result()

#results out of 1024
counts = result.get_counts(circuit)
print(counts)  #dictionary of results "00" and "11"
numerator = counts['00'] #how many counts of 00
probability00 = numerator/1024
probability11 = (1024-numerator)/1024

print("Probability of 00: " + str(probability00))
print("Probability of 11: " + str(probability11))

# Import draw_circuit, then use it to draw the circuit
from ibm_quantum_widgets import draw_circuit
draw_circuit(circuit)


#https://quantum-computing.ibm.com/composer/files/0b717e1408ecb7369bae5354ca48e08d


#https://towardsdatascience.com/building-your-own-quantum-circuits-in-python-e9031b548fa7