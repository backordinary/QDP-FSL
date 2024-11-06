# https://github.com/Ta-SeenJunaid/Quantum-Computing/blob/04cf98dcb4a69dfc25275de738cef0b4992bd3bc/Quantum%20Cryptography/Task_1.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, IBMQ, BasicAer, execute

alice_key = ''
bob_key = ''

key_length=100

q = QuantumRegister(3, "q")
c = ClassicalRegister(4, "c")

circuit = QuantumCircuit(q, c)

circuit.h(q[0])
circuit.h(q[1])
circuit.measure(q[0], c[0])
circuit.measure(q[1], c[1])
circuit.ch(q[1], q[0])
circuit.barrier()

circuit.h(q[2])
circuit.measure(q[2], c[2])
circuit.ch(q[2], q[0])

circuit.measure(q[0], c[3])

print(circuit)

backend = BasicAer.get_backend("qasm_simulator")

number_transmissions = 0
number_success = 0
number_failure = 0
unchecked_bits = 0

while len(alice_key) < key_length:
    job = execute(circuit, backend=backend, shots=1)

    result = job.result()
    state = list(result.get_counts(circuit).keys())[0]

    b = int(state[3])
    a = int(state[0])
    ap = int(state[1])
    bp = int(state[2])

    if ap==bp:
        alice_key += str(a)
        bob_key += str(b)


print('Alice\'s key: ', alice_key)
print('Bob\'s   key: ', bob_key)

if alice_key==bob_key:
    print('All keys are confirmed to be the same.')
else:
    print('Not all keys are identical. This should not happen unless somebody is listening to the communication!')
