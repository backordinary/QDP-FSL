# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/Exercises/6/2/one_counter.py
# my_first_score.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# Define the Quantum and Classical Registers
a = QuantumRegister(3, "a")
t = QuantumRegister(5, "t")
c = ClassicalRegister(2, "b")

# Build the circuit
cir = QuantumCircuit(a, c)
cir.add_register(t)

#get Input
a_in = input("a> ")
for i in range(0, len(a_in)):
    if a_in[i] == "1":
        cir.x(a[i])
cir.x(t[3])
cir.x(t[4])

#b0
#AND
cir.ccx(a[0], a[2], t[0])
cir.ccx(a[0], a[1], t[1])
cir.ccx(a[1], a[2], t[2])
#OR
cir.cswap(t[0], t[1], t[3])
cir.reset(t[3])
cir.x(t[3])
cir.cswap(t[1], t[2], t[3])


cir.measure(t[2], c[1])

cir.barrier()
#b1
cir.reset(t)
cir.x(a[0:2])
cir.ccx(a[0], a[1], t[0])
cir.ccx(a[2], t[0], t[1])


cir.x(a[1:3])
cir.reset(t[0])
cir.ccx(a[0], a[1], t[0])
cir.ccx(a[2], t[0], t[2])

cir.x(a[0:2])
cir.reset(t[0])
cir.ccx(a[0], a[1], t[0])
cir.ccx(a[2], t[0], t[3])

cir.x(a[1:3])
cir.reset(t[0])
cir.ccx(a[0], a[1], t[4])
cir.ccx(a[2], t[4], t[0])

cir.reset(t[4])
cir.x(t[4:6])

cir.cswap(t[0], t[1], t[4])
cir.reset(t[4])
cir.x(t[4])
cir.cswap(t[1], t[2], t[4])

cir.reset(t[4:6])
cir.x(t[4:6])
cir.cswap(t[2], t[3], t[4])
#cir.cswap(t[3], t[4], t[5])


cir.measure(t[3], c[0])
#cir.measure(t[0:4], c[2:6])
#cir.measure(a, c[5:8])


# measurement operations

print(cir)
# Execute the circuit
job = execute(cir, backend = Aer.get_backend('qasm_simulator'), shots=1)
result = job.result()

# Print the result
print(result.get_counts(cir))
