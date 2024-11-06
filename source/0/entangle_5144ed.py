# https://github.com/gquarles/quantum-testing/blob/c5a40d5ef0b5a2fa0f3f584298d8e325932bcdc3/entangle.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, assemble, execute, IBMQ
from qiskit.tools.monitor import job_monitor

import config

IBMQ.enable_account(config.api_key)
provider = IBMQ.get_provider(hub='ibm-q')
backend = 'ibmq_belem'
monitor = True


#Init n qbits
q = QuantumRegister(2,'q')
c = ClassicalRegister(2,'c')
circuit = QuantumCircuit(q,c)

#Put qbits into superposition of 1 and 0
circuit.h(0)
circuit.cx(0, 1)

circuit.measure(q,c) # Collapse wave function

backend = provider.get_backend(backend)
job = execute(circuit, backend, shots=20000, memory=True) #64 * 4 = 256

if monitor:
    job_monitor(job)

#Get list of bits as string EX: '1010' 
bits = job.result().get_counts()
print(bits)

bits = job.result().get_memory()

bitString = ''

for bit in bits:
    bitString += bit

b0 = 0
b1 = 0
for bit in bitString:
    if bit == '0':
        b0 += 1
    elif bit == '1':
        b1 += 1
    else:
        print('HOW:? ', end='')
        print(bit)

print()
print(b0)
print(b1)

print()
print(b0 / (b0 + b1))