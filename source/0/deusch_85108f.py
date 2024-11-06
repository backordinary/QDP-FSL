# https://github.com/Ta-SeenJunaid/Quantum-Computing/blob/6020adc5cb49ec4fa42fed5205f9ff4650d9604d/Deusch%20problem/Deusch.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, BasicAer, execute

q = QuantumRegister(2, "q")
c = ClassicalRegister(2, "c")

circuit = QuantumCircuit(q, c, name="Deusch problem")

circuit.x(q[1])
circuit.barrier()

circuit.h(q[0])
circuit.h(q[1])
circuit.barrier()

# put any of the 4 cases
circuit.cx(q[0], q[1])

circuit.barrier()

circuit.h(q[0])
circuit.h(q[1])

circuit.measure(q[0], c[0])
circuit.measure(q[1], c[1])

print(circuit)

backend = BasicAer.get_backend("qasm_simulator")
job = execute(circuit, backend=backend, shots=1024)

result = job.result()
print(result.get_counts(circuit))
