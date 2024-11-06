# https://github.com/Talkal13/Quantum/blob/ccda55776da0a3f5bd212a8566f0a1e367061a6f/Lab/main.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.qasm import pi
from phase import phase_change

q = QuantumRegister(1)
a = QuantumRegister(1, name="a")
m = ClassicalRegister(1)
main = QuantumCircuit(q, a, m, name="main")

q_uf = QuantumRegister(1)
a_uf = QuantumRegister(1)

Uf = QuantumCircuit(q_uf, a_uf, name="Uf")
Uf.cx(q_uf[0], a_uf[0])

main.append(phase_change(pi/4, 1, Uf), q[::] + [a[:]])
#main.h(q)

main.measure(q, m)
print(main.decompose())

# Execute the circuit
job = execute(main, backend = Aer.get_backend('qasm_simulator'), shots=1000)
result = job.result()

# Print the result
print(result.get_counts(main))

