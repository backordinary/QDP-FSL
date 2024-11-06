# https://github.com/Mobink980/Python_programs/blob/a08fff616577d01a31f2e6d9fe44226f6c925ee2/Deutsch_josza.py
# We now implement the Deutsch-Josza algorithm for the example of a three-bit function, 
# with both constant and balanced oracles. First let's do our imports:

# initialization
import numpy as np
# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute

# import basic plot tools
from qiskit.visualization import plot_histogram

# Next, we set the size of the input register for our oracle
n = 3

# Let's start by creating a constant oracle, 
# in this case the input has no effect on the ouput so we just randomly set the output qubit to be 0 or 1:
const_oracle = QuantumCircuit(n+1)

output = np.random.randint(2)
if output == 1:
    const_oracle.x(n)

const_oracle.draw()

balanced_oracle = QuantumCircuit(n+1)

# Next, we create a balanced oracle. As we saw in section 1b, we can create a 
# balanced oracle by performing CNOTs with each input qubit as a control and 
# the output bit as the target. We can vary the input states that give 0 or 1 
# by wrapping some of the controls in X-gates. Let's first choose a binary string 
# of length n that dictates which controls to wrap:
b_str = "101"

# Now we have this string, we can use it as a key to place our X-gates.
#  For each qubit in our circuit, we place an X-gate if the corresponding 
# digit in b_str is 1, or do nothing if the digit is 0.

balanced_oracle = QuantumCircuit(n+1)
b_str = "101"

# Place X-gates
for qubit in range(len(b_str)):
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)
balanced_oracle.draw()


# Next, we do our controlled-NOT gates, 
# using each input qubit as a control, and the output qubit as a target:
balanced_oracle = QuantumCircuit(n+1)
b_str = "101"

# Place X-gates
for qubit in range(len(b_str)):
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)

# Use barrier as divider
balanced_oracle.barrier()

# Controlled-NOT gates
for qubit in range(n):
    balanced_oracle.cx(qubit, n)

balanced_oracle.barrier()
balanced_oracle.draw()


# Finally, we repeat the code from two cells up to finish wrapping the controls in X-gates:
balanced_oracle = QuantumCircuit(n+1)
b_str = "101"

# Place X-gates
for qubit in range(len(b_str)):
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)

# Use barrier as divider
balanced_oracle.barrier()

# Controlled-NOT gates
for qubit in range(n):
    balanced_oracle.cx(qubit, n)

balanced_oracle.barrier()

# Place X-gates
for qubit in range(len(b_str)):
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)

# Show oracle
balanced_oracle.draw()

"We have just created a balanced oracle! All that's left to do is see if the Deutsch-Joza algorithm can solve it."

# Let's now put everything together. This first step in the algorithm 
# is to initialise the input qubits in the state [Math Processing Error] 
# and the output qubit in the state [Math Processing Error]:
dj_circuit = QuantumCircuit(n+1, n)

# Apply H-gates
for qubit in range(n):
    dj_circuit.h(qubit)

# Put qubit in state |->
dj_circuit.x(n)
dj_circuit.h(n)
dj_circuit.draw()

# Next, let's apply the oracle. Here we apply the balanced_oracle we created above:
dj_circuit = QuantumCircuit(n+1, n)

# Apply H-gates
for qubit in range(n):
    dj_circuit.h(qubit)

# Put qubit in state |->
dj_circuit.x(n)
dj_circuit.h(n)

# Add oracle
dj_circuit += balanced_oracle
dj_circuit.draw()

# Finally, we perform H-gates on the [Math Processing Error]-input qubits, 
# and measure our input register:
dj_circuit = QuantumCircuit(n+1, n)

# Apply H-gates
for qubit in range(n):
    dj_circuit.h(qubit)

# Put qubit in state |->
dj_circuit.x(n)
dj_circuit.h(n)

# Add oracle
dj_circuit += balanced_oracle

# Repeat H-gates
for qubit in range(n):
    dj_circuit.h(qubit)
dj_circuit.barrier()

# Measure
for i in range(n):
    dj_circuit.measure(i, i)

# Display circuit
dj_circuit.draw()


# Let's see the output:
# use local simulator
backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(dj_circuit, backend=backend, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)

# We can see from the results above that we have a 0% chance of measuring 000. 
# This correctly predicts the function is balanced.

