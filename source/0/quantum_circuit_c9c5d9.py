# https://github.com/WGtheLearner/Quantum-Compute-Simulator/blob/a2b62189852de5da21767064cf71ab413d6eab24/out/production/QC_Simulator/com/company/quantum_circuit.py
import numpy as np
import sys
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


# Create a Quantum Circuit acting on the q register
qubit_counts = int(sys.argv[1])
bit_counts = qubit_counts
circuit = QuantumCircuit(qubit_counts, bit_counts)
simulator = Aer.get_backend('qasm_simulator')

file = open("D:\\Java\\QC_Simulator\\Instructions.txt","r")
lines = file.readlines()

if not lines:
    exit()
for line in lines:
    line = line.strip("\n")
    line = line.split(' ')
    if line[0] == 'H':
        # Use Aer's qasm_simulator
        q_index = int(line[1])
# Add a H gate on qubit 0
        circuit.h(q_index)

# Map the quantum measurement to the classical bits
        #circuit.measure([q_index], [q_index])

# Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

# Grab results from the job
        result = job.result()
# Returns counts
#counts = result.get_counts(circuit)
#print("\nTotal count for 00 and 11 are:",counts)

# Plot a histogram
        #plot_histogram(counts)
    elif line[0] == 'X':

        q_index = int(line[1])
# Add a H gate on qubit 0
        circuit.x(q_index)

# Map the quantum measurement to the classical bits
        #circuit.measure([q_index], [q_index])

# Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

# Grab results from the job
        result = job.result()

# Returns counts
#counts = result.get_counts(circuit)
#print("\nTotal count for 00 and 11 are:",counts)
        
    elif line[0] == 'Y':

# Add a Y gate on qubit 0
        q_index = int(line[1])
        
        circuit.y(q_index)
        
# Map the quantum measurement to the classical bits
        #circuit.measure([q_index], [q_index])
        
# Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

# Grab results from the job
        result = job.result()

# Returns counts
#counts = result.get_counts(circuit)
#print("\nTotal count for 00 and 11 are:",counts)
        
    elif line[0] == 'Z':
# Add a Z gate on qubit 0
        q_index = int(line[1])

        circuit.z(q_index)

# Map the quantum measurement to the classical bits
        #circuit.measure([q_index], [q_index])
        
# Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

# Grab results from the job
        result = job.result()
        
    elif line[0] == 'SDG':
# Add a SDG gate on qubit 0
        q_index = int(line[1])

        circuit.sdg(q_index)

# Map the quantum measurement to the classical bits
        #circuit.measure([q_index], [q_index])
        
# Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

# Grab results from the job
        result = job.result()
        
    elif line[0] == 'CX':
# Add a CNOT gate on qubit 0/1

        q_index1 = int(line[1])
        q_index2 = int(line[2])

        circuit.cx(q_index1,q_index2)

        #circuit.measure([q_index1,q_index2], [q_index1,q_index2])

# Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

# Grab results from the job
        result = job.result()

    elif line[0] == 'CZ':
   # Add a ControlledZ gate on qubit 0/1

        q_index1 = int(line[1])
        q_index2 = int(line[2])

        circuit.cz(q_index1,q_index2)

           #circuit.measure([q_index1,q_index2], [q_index1,q_index2])

   # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

   # Grab results from the job
        result = job.result()
        
    elif line[0] == 'SWAP':
   # Add a SWAP gate on qubit 0/1

        q_index1 = int(line[1])
        q_index2 = int(line[2])

        circuit.swap(q_index1,q_index2)

   # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

   # Grab results from the job
        result = job.result()
        
    elif line[0] == 'M':
        q_index = int(line[1])
        
        circuit.measure([q_index], [q_index])

      # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

      # Grab results from the job
        result = job.result()

# Draw the circuit
print(circuit)       
