# https://github.com/JffMv/ImplementacionDeutsch-Jozsa/blob/bc96117b1913d41ca797244906945f74e0e38e50/implementacion_de_algorimos.py
#¡¡¡¡¡¡¡No Ejecutarlo en bloque y de una sola vez, 
#ya que cada bloque es una ejecución diferente.!!!!!!

#Los tres bloques comparten las importaciones.

########################## El siguiente código pertenece a la función 1 de Deuch Joza##########################################

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
# Use Aer's qasm_simulator
# simulator = Aer.get_backend('qasm_simulator')


qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
U = Operator(circuit)

# Show the results
print(U.data)

circuit.measure(qreg_q[1], creg_c[3])
circuit.measure(qreg_q[0], creg_c[4])
circuit.measure(qreg_q[3], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[4], creg_c[0])


# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
#circuit.cx(1, 0)
# Map the quantum measurement to the classical bits
#es el numero de medidas q1 ---> c1 o mas
# circuit.measure([0], [0])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
#2. Install Qiskit
#3. Install the visualization support for Qiskit
#4. If you are using zsh, for example in MacOS, you can write instead:
# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()

########################## El siguiente código pertenece a la función 2 de Deuch Joza##########################################


qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[2], qreg_q[3])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])

# Show the results
print(U.data)


circuit.measure(qreg_q[1], creg_c[3])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[1])
circuit.measure(qreg_q[0], creg_c[4])
circuit.measure(qreg_q[4], creg_c[0])




# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
#circuit.cx(1, 0)
# Map the quantum measurement to the classical bits
#es el numero de medidas q1 ---> c1 o mas
# circuit.measure([0], [0])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
#2. Install Qiskit
#3. Install the visualization support for Qiskit
#4. If you are using zsh, for example in MacOS, you can write instead:
# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()




########################## El siguiente código pertenece a la función 3 de Deuch Joza##########################################



qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.x(qreg_q[3])
circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])

# Show the results
print(U.data)


circuit.measure(qreg_q[1], creg_c[3])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[1])
circuit.measure(qreg_q[0], creg_c[4])
circuit.measure(qreg_q[4], creg_c[0])




# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
#circuit.cx(1, 0)
# Map the quantum measurement to the classical bits
#es el numero de medidas q1 ---> c1 o mas
# circuit.measure([0], [0])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
#2. Install Qiskit
#3. Install the visualization support for Qiskit
#4. If you are using zsh, for example in MacOS, you can write instead:
# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()





########################## El siguiente código pertenece a la función 4 de Deuch Joza##########################################

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])
circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3], qreg_q[4])

# Show the results
print(U.data)


circuit.measure(qreg_q[1], creg_c[3])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[1])
circuit.measure(qreg_q[0], creg_c[4])
circuit.measure(qreg_q[4], creg_c[0])




# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
#circuit.cx(1, 0)
# Map the quantum measurement to the classical bits
#es el numero de medidas q1 ---> c1 o mas
# circuit.measure([0], [0])
# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)
#2. Install Qiskit
#3. Install the visualization support for Qiskit
#4. If you are using zsh, for example in MacOS, you can write instead:
# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)
# Draw the circuit
print(circuit)
# Plot a histogram
plot_histogram(counts)
plt.show()


