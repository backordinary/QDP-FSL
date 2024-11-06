# https://github.com/RohitMidha23/quantum_algorithms/blob/51097ec4a7f87c319f9e1bbb4a0d0c3d91c21cd7/bitwise_operations.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer

def bitwise_and(circuit,a,b,c,N):
    for i in range(N):
        circuit.ccx(a[i],b[i],c[i])

def bitwise_or(circuit,a,b,c,N):
    for i in range(N):
        circuit.ccx(a[i],b[i],c[i])
        circuit.cx(a[i],b[i])
        circuit.cx(b[i],c[i])

def bitwise_not(circuit,a,c,N):
    for i in range(N):
        circuit.cx(a[i],c[i])
        circuit.x(c[i])

def bitwise_xor(circuit,a,b,c,N):
    for i in range(N):
        circuit.cx(a[i],c[i])
        circuit.cx(b[i],c[i])

def bitwise_xnor(circuit,a,b,c,N):
    bitwise_xor(circuit,a,b,c,N)
    for i in range(N):
        circuit.x(c[i])

# Bitwise AND
# Registers and circuit.
a = QuantumRegister(4)
b = QuantumRegister(4)
c = QuantumRegister(4)
ca = ClassicalRegister(4)
cb = ClassicalRegister(4)
cc = ClassicalRegister(4)
circuit = QuantumCircuit(a, b, c, ca, cb, cc)

# Inputs
# a = 1010
# b = 1011
circuit.x(a[1])
circuit.x(a[3])
circuit.x(b[0])
circuit.x(b[1])
circuit.x(b[3])

# Take the bitwise AND.
bitwise_and(circuit, a, b, c, 4)

# Measure.
circuit.measure(a, ca)
circuit.measure(b, cb)
circuit.measure(c, cc)

# Simulate the circuit.
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend_sim)
result_sim = job_sim.result()

# Expected Output : 1010 1011 1010
# NOTE: In qiskit, little endian is followed and hence the output is actually c b a
#       where c in the bitwise AND of a and b
print("Bitwise AND : ")
print(result_sim.get_counts(circuit))

# Bitwise OR
# Registers and circuit.
a = QuantumRegister(4)
b = QuantumRegister(4)
c = QuantumRegister(4)
ca = ClassicalRegister(4)
cb = ClassicalRegister(4)
cc = ClassicalRegister(4)
circuit = QuantumCircuit(a, b, c, ca, cb, cc)

# Inputs
# a = 1010
# b = 1011
circuit.x(a[1])
circuit.x(a[3])
circuit.x(b[0])
circuit.x(b[1])
circuit.x(b[3])

# Take the bitwise OR.
bitwise_or(circuit, a, b, c, 4)

# Measure.
circuit.measure(a, ca)
circuit.measure(b, cb)
circuit.measure(c, cc)

# Simulate the circuit.
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend_sim)
result_sim = job_sim.result()

# Expected Output : 1011 1011 1010
# NOTE: In qiskit, little endian is followed and hence the output is actually c b a
#       where c in the bitwise OR of a and b
print("Bitwise OR : ")
print(result_sim.get_counts(circuit))


# Bitwise NOT
# Registers and circuit.
a = QuantumRegister(4)
c = QuantumRegister(4)
ca = ClassicalRegister(4)
cc = ClassicalRegister(4)
circuit = QuantumCircuit(a, c, ca, cc)

# Inputs
# a = 1010
circuit.x(a[1])
circuit.x(a[3])


# Take the bitwise NOT
bitwise_not(circuit, a, c, 4)

# Measure.
circuit.measure(a, ca)
circuit.measure(c, cc)

# Simulate the circuit.
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend_sim)
result_sim = job_sim.result()

# Expected Output : 0101 1010
# NOTE: In qiskit, little endian is followed and hence the output is actually c a
#       where c in the bitwise NOT of a
print("Bitwise NOT : ")
print(result_sim.get_counts(circuit))


# Bitwise XOR
# Registers and circuit.
a = QuantumRegister(4)
b = QuantumRegister(4)
c = QuantumRegister(4)
ca = ClassicalRegister(4)
cb = ClassicalRegister(4)
cc = ClassicalRegister(4)
circuit = QuantumCircuit(a, b, c, ca, cb, cc)

# Inputs
# a = 1010
# b = 1011
circuit.x(a[1])
circuit.x(a[3])
circuit.x(b[0])
circuit.x(b[1])
circuit.x(b[3])

# Take the bitwise XOR.
bitwise_xor(circuit, a, b, c, 4)

# Measure.
circuit.measure(a, ca)
circuit.measure(b, cb)
circuit.measure(c, cc)

# Simulate the circuit.
backend_sim = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend_sim)
result_sim = job_sim.result()

# Expected Output : 0001 1011 1010
# NOTE: In qiskit, little endian is followed and hence the output is actually c b a
#       where c in the bitwise XOR of a and b
print("Bitwise XOR : ")
print(result_sim.get_counts(circuit))
