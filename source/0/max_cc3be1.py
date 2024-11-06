# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/Exercises/6/4/max.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# Copy
def Copy(n):
    a_c = QuantumRegister(n)
    b_c = QuantumRegister(n)

    copy = QuantumCircuit(a_c, b_c, name="copy")

    for i in range(0, n):
        copy.cx(a_c[i], b_c[i])

    return copy

# Compare equal
a_e = QuantumRegister(1)
b_e = QuantumRegister(1)
c_e = QuantumRegister(1)

equal = QuantumCircuit(a_e, b_e, c_e, name="Equal")
equal.cx(a_e, b_e) #Calculate XOR
equal.x(b_e) # X
equal.append(Copy(1), [b_e, c_e])
equal.x(b_e) #reset
equal.cx(a_e, b_e)

# Compare max
a_m = QuantumRegister(1)
b_m = QuantumRegister(1)
c_m = QuantumRegister(1)

max_c = QuantumCircuit(a_m, b_m, c_m, name="a>b")
max_c.append(Copy(1), [a_m, c_m])


#Input
a_i = input("a> ")
b_i = input("b> ")
N = len(a_i)

a = QuantumRegister(N, name="a")
b = QuantumRegister(N, name="b")
t = QuantumRegister(1, name="a>b")
p = QuantumRegister(1, name="equal")

m = ClassicalRegister(N, name="m")
e = ClassicalRegister(1, name="e")

q = QuantumCircuit(a, b, c, t, m, e, name="Max")

for i in range(0, N):
    if (a_i[i] == "1"):
        q.x(a[N - i - 1])
    if (b_i[i] == "1"):
        q.x(b[N - i - 1])

for i in range(0, N):
    q.append(equal, [a[N-i - 1], b[N-i-1], t])

if (e == 1):
    q.append(Copy(N), a[0:N] + c[0:N])
else:
    q.append(Copy(N), b[0:N] + c[0:N])

q.measure(c, m)

print(q)

# Execute the circuit
job = execute(q, backend = Aer.get_backend('qasm_simulator'), shots=1)
result = job.result()

# Print the result
print(result.get_counts(q))