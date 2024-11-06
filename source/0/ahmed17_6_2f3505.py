# https://github.com/ahmedkfu2020/-/blob/a205805a9dfaef2f8cb2ff0645c597b1b119747c/ahmed17_6.py
from random import randrange
from math import pi
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(0,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(1,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(2,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(3,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(4,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(5,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(6,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(7,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(8,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
# pick a random angle
random_angle = randrange(3600)/10

# specify the angles
rotation_angle1 = random_angle/360*2*pi
rotation_angle2 = rotation_angle1 + pi/2

#
# first qubit
#
q1 =  QuantumRegister(1) 
c1 = ClassicalRegister(1) 
qc1 = QuantumCircuit(q1,c1)

# rotate the qubit
qc1.ry(2 * rotation_angle1,q1[0])

# read the quantum state
job = execute(qc1,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state1=job.result().get_statevector(qc1) 
[x1,y1]=[current_quantum_state1[0].real,current_quantum_state1[1].real]

#
# second qubit 
#
q2 =  QuantumRegister(1) 
c2 = ClassicalRegister(1) 
qc2 = QuantumCircuit(q2,c2)

# rotate the qubit    
qc2.ry(2 * rotation_angle2,q2[0])
    
# read the quantum state
job = execute(qc2,Aer.get_backend('statevector_simulator'),optimization_level=0)
current_quantum_state2=job.result().get_statevector(qc2) 
[x2,y2]=[current_quantum_state2[0].real,current_quantum_state2[1].real]

    
#
# dot product
#
print(9,"- the result of dot product is ",round(x1*x2+y1*y2,5))
print("random angle is",random_angle)
print("x1 , y1 =",round(x1,5),round(y1,5))
print("x2 , y2 =",round(x2,5),round(y2,5))
print()
