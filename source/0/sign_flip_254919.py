# https://github.com/Ta-SeenJunaid/Quantum-Computing/blob/48df323f9070b3b3266602de2da4594d6f75e809/Quantum%20error%20correction%20(QEC)/Sign_flip/sign_flip.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, IBMQ, BasicAer, execute

q = QuantumRegister(3, "q")
c = ClassicalRegister(3, "c")

circuit = QuantumCircuit(q,c)

circuit.cx(q[0], q[1])
circuit.cx(q[0], q[2])

circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2])

circuit.barrier(q[0], q[1], q[2])
#apply sign flip error
circuit.z(q[0])
#circuit.z(q[1])
#circuit.z(q[2])

circuit.barrier(q[0], q[1], q[2])

circuit.h(q[0])
circuit.h(q[1])
circuit.h(q[2])

circuit.cx(q[0], q[1])
circuit.cx(q[0], q[2])
circuit.ccx(q[1], q[2], q[0])

circuit.measure(q[0], c[0])
circuit.measure(q[1], c[1])
circuit.measure(q[2], c[2])

print(circuit)

backend = BasicAer.get_backend("qasm_simulator")
job = execute(circuit, backend=backend, shots=1024)

result = job.result()
print(result.get_counts(circuit))