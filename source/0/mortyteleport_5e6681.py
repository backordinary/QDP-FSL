# https://github.com/evilmortyyy/mortyportal/blob/41a566cc97ca5e631d7ae342a94514b0ba7db38d/mortyteleport.py
import argparse
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

# Create the parser
parser = argparse.ArgumentParser()

# Add the hex string option
parser.add_argument("--hex_string", help="The morty hex string to be encoded and measured", type=str,nargs='+')

# Parse the arguments
args = parser.parse_args()

# Convert the hex string into binary
binary_string = ''.join(args.hex_string)
binary_string= bin(int(binary_string, 16))[2:].zfill(len(binary_string)*4)

# Create a quantum register with the same number of qubits as the binary string
num_qubits = len(binary_string)
q = QuantumRegister(num_qubits)
c = ClassicalRegister(num_qubits)

# Create a quantum circuit
circuit = QuantumCircuit(q, c)

# Define the initial state of the qubits
initial_state = [int(i) for i in binary_string]

# Iterate through the input string and initialize the qubits
for i, char in enumerate(binary_string):
    if char == "1":
        circuit.x(q[i])
    elif char == "0":
        circuit.x(q[i])

# Iterate through the qubits and perform the teleportation operation
for i in range(0, num_qubits-2, 3):
    circuit.h(q[i+1])
    circuit.cx(q[i+1],q[i])
    circuit.cx(q[i+1],q[i+2])
    circuit.h(q[i+1])
    circuit.measure(q[i+1], c[i+1])
    circuit.measure(q[i], c[i])
    circuit.barrier()
    circuit.cx(q[i+1],q[i+2])
    circuit.cz(q[i],q[i+2])
    circuit.barrier()

# Choose a backend to run the circuit on
backend = Aer.get_backend('qasm_simulator')

# Execute the circuit and get the result
job = execute(circuit, backend, shots=1)
result = job.result()

# Get the measurement outcome from the result
measurement_outcome = result.get_counts(circuit)

# Extract the measurement results
measurements = [measurement_outcome.get('{0:b}'.format(i), 0) for i in range(num_qubits)]

# Apply the correction based on the measurements
for i in range(num_qubits):
    if measurements[i] == 1:
        circuit.x(q[i])

# Extract the final state of the qubit
final_state = execute(circuit, backend, shots=1).result().get_counts(circuit)
final_state = list(final_state.keys())[0]
final_state = [int(i) for i in final_state]

print("Morty state: ", initial_state)
# Print the final state
print("Rick state: ", final_state)

print("pickle: ", measurement_outcome)

if initial_state == final_state:
    print("Teleportation successful")
else:
    print("Teleportation failed") 
    
    
# Measure the qubits after teleportation
circuit.measure(q, c)

# Run the circuit with the measurement
measurement_result = execute(circuit, backend, shots=1).result().get_counts(circuit)
print("Measurement result after teleportation: ", measurement_result)
