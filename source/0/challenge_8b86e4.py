# https://github.com/ThijsRay/TU-Delft-CTF-2022/blob/ca915c2419ae8ab890eb22fada9dac101e188e7e/quantum/cloning/src/challenge.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, assemble

import random

import os

debug = False

qc = QuantumCircuit()
secret_register = QuantumRegister(10, "secret")
user_register = QuantumRegister(10, "user")
classical_register = ClassicalRegister(10, "classical")

qc.add_register(secret_register)
qc.add_register(user_register)
qc.add_register(classical_register)

solution = []
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)
# Flip a coin for each secret bit
if random.random() > 0.5:
    solution.append(1)
else:
    solution.append(0)

# leave a qubit on |0> or set it to |1> depending on the secret bit
# do this in reversed order since register[0] is the most significant bit
for i, bit in enumerate(reversed(solution)):
    if bit:
        qc.x(secret_register[i])

# Just for printing in a nicer way
qc.barrier()

# Now allow the user to make changes to the user qubits if they want to
max_operations = 100
for _ in range(max_operations):
    print("Do you want to add any gates? x[user_i], y[user_i], z[user_i], h[user_i], cnot[secret_i,user_i], done")
    user_input = input()

    if user_input == "done":
        break
    elif user_input.startswith("x["):
        user_i = int(user_input[2])
        qc.x(user_register[user_i])
    elif user_input.startswith("y["):
        user_i = int(user_input[2])
        qc.y(user_register[user_i])
    elif user_input.startswith("z["):
        user_i = int(user_input[2])
        qc.z(user_register[user_i])
    elif user_input.startswith("h["):
        user_i = int(user_input[2])
        qc.h(user_register[user_i])
    elif user_input.startswith("cnot["):
        server_i = int(user_input[5])
        user_i = int(user_input[7])
        qc.cnot(secret_register[server_i], user_register[user_i])
    else:
        print("Unknown operation")
        exit()
else:
    print("Added too many operations, don't forget to call done when you are done adding gates")
    exit()
qc.measure(user_register[0], classical_register[0])
qc.measure(user_register[1], classical_register[1])
qc.measure(user_register[2], classical_register[2])
qc.measure(user_register[3], classical_register[3])
qc.measure(user_register[4], classical_register[4])
qc.measure(user_register[5], classical_register[5])
qc.measure(user_register[6], classical_register[6])
qc.measure(user_register[7], classical_register[7])
qc.measure(user_register[8], classical_register[8])
qc.measure(user_register[9], classical_register[9])

# Print the circuit for debugging
print("Running circuit:")

if debug:
    print(qc.draw("text"))

# Simulate the circuit once
simulator = Aer.get_backend('aer_simulator')
qobj = assemble(qc, shots=1)
job = simulator.run(qobj)
results = job.result().get_counts()
measurement = list(results)[0]

# Compare measured result with secret register
# Since the no-cloning theorem exists this should be impossible! ... right?
secret_string = "".join(str(value) for value in solution)
if measurement == secret_string:
    flag = os.getenv("FLAG")
    print(f"Congrats! Here is the flag {flag}")
else:
    print(f"Wrong answer, measurement {measurement} != secret {secret_string}")
