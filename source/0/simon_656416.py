# https://github.com/lbadams2/csc_qc/blob/c9894fa587a331e44583717341013abd3a5d6875/hw4/simon.py
# Adapted from a qiskit notebook

from qiskit import Aer, QuantumCircuit, execute

def dotprod(a, b):
    temp_sum = 0
    for i in range(len(a)):
        temp_sum += int(a[i])*int(b[i])
    return temp_sum%2

def oracle(circ, b):
    # copy contents of first register to second register
    n = len(b)
    for i, char in enumerate(b):
        circ.cx(i, i+len(b))
        # for bits in b = 1, make corresponding qubits in second register target of CNOT
        # if b = 110, second reg is |q3q4q5>, targets are q4 and q5
        if char == '1':
            target = (2*n - 1) - i
            control = n - 1 # not sure what this should be (most significant qubit in first register)
            circ.cx(control, target)        

# Main function
#b = '110'
b = '1010'
n = len(b)
circ = QuantumCircuit(n*2, n)

circ.h(range(n))
circ.barrier()
oracle(circ, b)
circ.barrier()
circ.h(range(n))

circ.measure(range(n), range(n))
print(circ)

backend = Aer.get_backend('qasm_simulator')
shots = 1024
job = execute(circ, backend=backend, shots=shots)
results = job.result()
counts = results.get_counts()

for code in counts:
    print("(" + str(b) + " dot " + str(code) + ")%2 = " + str(dotprod(b, code)))
