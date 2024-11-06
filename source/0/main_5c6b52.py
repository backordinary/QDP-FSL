# https://github.com/pkruk/qlabs/blob/7a399a41e08ff35e08f3df90423fb1823c6c33ba/teleportation/main.py
# Quantum teleportation example
# this code is based on a publication:
# "Quantum Teleportation is a Universal Computational Primitive" - Daniel Gottesman, Isaac L. Chuang
# https://arxiv.org/abs/quant-ph/9908010


from qiskit import *


q = QuantumRegister(3) # only 3 qubits

l0 = ClassicalRegister(1, "l1")
l1 = ClassicalRegister(1, "l2")
l2 = ClassicalRegister(1, "l3")

# we hade a circuit with 3 lines which will be measured:
#
#  QUBIT0 ----------------- QUANTUM GATES ---- MEASURE ---- OUTPUT0 # line0  l0
#  QUBIT1 ----------------- QUANTUM GATES ---- MEASURE ---- OUTPUT1  # line1 l1
#  QUBIT2 ----------------- QUANTUM GATES ---- MEASURE ---- OUTPUT2  # line2 l2
#
#
# First qubit is a secred that will be passed
# Second and Third qubit is a bell pair  (Alice take second, Bob third one)
# Alice use secret and her part of bell pair to encrypt a message
# Bob use a measurement of a secret and alice qubit with his qubit to decrypt a message
#

circuit = QuantumCircuit(q, l0, l1, l2)

circuit.u3(0.3, 0.2, 0.1, q[0]) # https://developer.ibm.com/code/2017/05/17/developers-guide-to-quantum-qiskit-sdk/

# Bell pairs

#
#  X --- H(Hadamard gate) --- CNOT -------
#  Y ---------- | ------------------------

circuit.h(q[1])
circuit.cx(q[1], q[2])


circuit.barrier(q) # freeze a state, say a stop to any optimalizations

# We had a bell pairs, so lets make a teleportation
#
# X ----- CNOT ------- H ----
# Y ----- |    --------------
#

circuit.cx(q[0], q[1])
circuit.h(q[0])

circuit.measure(q[0], l0[0])
circuit.measure(q[1], l1[0])


# restore a secret with using BEN qubit
circuit.barrier(q)

circuit.z(q[2]).c_if(l0, 1) # if first qubit is one then apply  Z gate
circuit.x(q[2]).c_if(l1, 1) # if second qubit is one then apply X gate

circuit.measure(q[2], l2[0])

# run a experimenet

initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2)}


backend = Aer.get_backend("qasm_simulator")

qobj = compile(circuit, backend=backend, coupling_map=None, shots=1024, initial_layout=initial_layout)
job = backend.run(qobj)
qobj_exp = qobj.experiments[0]

result = job.result()
print(result.get_counts(circuit))
