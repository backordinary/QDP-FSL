# https://github.com/zoraizmohammad/BWSIQuantumSoftware-FinalProject/blob/da9a703e35ba746d2066955a5653205f26c70bdf/Raw%20Code/HHLAttempt8.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np
from qiskit.circuit.library.standard_gates import SGate, SdgGate
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import Aer


#generalized form of allowable: Matrix
#theta = 0 to 2*pi
#matrix = np.array([1+(np.cos(theta))**2, -np.sin(2*theta)/2],[-np.sin(2*theta)/2, 1+(np.sin(theta))**2])

"""#example 1
theta = (3*np.pi)/2

matrix = np.array([1+(np.cos(theta))**2, -np.sin(2*theta)/2],[-np.sin(2*theta)/2, 1+(np.sin(theta))**2])
sol = np.array([1,0])"""

#example 2
matrix = np.array([[1.5, 0.5],[0.5, 1.5]])
sol = np.array([1,-1])

k = np.sqrt((sol[0]**2)+(sol[1]**2)) #scaling factor so solution has mag. 1
sol = sol*(1/k)


qubits = QuantumRegister(4)
measurement = ClassicalRegister(4)
circuit = QuantumCircuit(qubits,measurement)

#t = number of variables + log(2+1/(2*error)) //some magic number
t = np.pi/2 #changed to pi/2

#r = number that increases precision of solution as it gets higher, but requires more computation
r = 4

#number of shots
s=10000

#last index of qubits
last_index = len(qubits)-1

#load sol
circuit.ry(2*np.arctan2(sol[1],sol[0]),qubits[-1])


#phase estimation

#set superposition
circuit.h(qubits[1])
circuit.h(qubits[2])

#trotterization
beta = matrix[0][1]
gamma = (matrix[0][0]-matrix[1][1])/2.0
alpha = (matrix[0][0]+matrix[1][1])/2.0 #added alpha

for i in (range(1,last_index)):
    circuit.rz(t*alpha*2**(i-1),  qubits[last_index-i])
    for x in range(r):
        circuit.crz(((-2*t*gamma*2**(i-1))/r),qubits[last_index-i],qubits[-1])
        circuit.crx(((-2*t*beta*2**(i-1))/r),qubits[last_index-i],qubits[-1])


circuit.swap(qubits[1],qubits[2])
circuit.h(qubits[2])

csdg = SdgGate().control(1) #create a controlled adjoint S gate
circuit.append(csdg,[1,2])


circuit.h(qubits[1])
circuit.swap(qubits[1],qubits[2])

# #controlled rotation
rot_number = 2

circuit.cry((2*np.pi)/(2**rot_number),qubits[1],qubits[0])
circuit.cry((np.pi)/(2**rot_number),qubits[2],qubits[0])


# #inverse phase estimation

circuit.swap(qubits[1],qubits[2])
circuit.h(qubits[1])

cs = SGate().control(1) #create a controlled S gate
circuit.append(cs,[1,2])

circuit.h(qubits[2])
circuit.swap(qubits[1],qubits[2])

# #reverse trotterization
for i in reversed(range(1,last_index)):
    circuit.rz(-t*alpha*2**(i-1),  qubits[last_index-i]) #added by Nikita
    for x in range(r):
        circuit.crx(((2*t*beta*2**(i-1))/r),qubits[last_index-i],qubits[-1])
        circuit.crz(((2*t*gamma*2**(i-1))/r),qubits[last_index-i],qubits[-1])
        
        
# #undo superposition
circuit.h(qubits[1])
circuit.h(qubits[2])

    
circuit.measure(qubits,measurement)

print(circuit)


simulator = Aer.get_backend('aer_simulator')
simulation = execute(circuit, simulator, shots=s)
result = simulation.result()

counts = result.get_counts(circuit)

#Only count instances when last qubit is 1

#classical post-processing
total = 0
x_count = 0
y_count = 0

print("COUNTS: "+str(counts))

for measure in counts:
    if (measure[3]=='1'):
       # print("MEASURE: "+measure)
        total+=counts[measure]

        if (measure[0]=='0'):
            x_count+=counts[measure]
        else:
            y_count+=counts[measure]

x_value = np.sqrt(x_count/total)*k
y_value = np.sqrt(y_count/total)*k
ratio = x_value/y_value

print("RATIO OF SOLUTIONS (X/Y): "+str(ratio))