# https://github.com/mine260309/qiskit_practice/blob/76c55f733874e2b0865b83104c9e8edea5c45978/one-bit-full-adder.py
#!/usr/bin/env python

from qiskit import QuantumProgram

Circuit = 'oneBitFullAdderCircuit'

# Create the quantum program
qp = QuantumProgram()

# Creating registers
n_qubits = 5
qr = qp.create_quantum_register("qr", n_qubits)
cr = qp.create_classical_register("cr", n_qubits)

# One-bit full adder circuit, where:
# qr[0], qr[1] are the bits to add
# qr[2] is the carry_in
# qr[3] is the result
# qr[4] is the carry_out
obc = qp.create_circuit(Circuit, [qr], [cr])

# Prepare bits to add
obc.h(qr[0])
obc.h(qr[1])
obc.h(qr[2])

# The result in qr[3]
obc.cx(qr[0], qr[3])
obc.cx(qr[1], qr[3])
obc.cx(qr[2], qr[3])

# The carry_out in qr[4]
obc.ccx(qr[0], qr[1], qr[4])
obc.ccx(qr[0], qr[2], qr[4])
obc.ccx(qr[1], qr[2], qr[4])

# Measure
for i in range(0, n_qubits):
    obc.measure(qr[i], cr[i])

# Get qasm source
source = qp.get_qasm(Circuit)
print(source)

# Compile and run
backend = 'local_qasm_simulator'
circuits = [Circuit]  # Group of circuits to execute

qobj = qp.compile(circuits, backend)  # Compile your program

result = qp.run(qobj, wait=2, timeout=240)
print(result)

print(result.get_counts(Circuit))
