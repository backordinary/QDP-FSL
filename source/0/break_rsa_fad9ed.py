# https://github.com/LuckyGitHub777/quantum-code/blob/8ecf394e1ab44eaee75b459c20f5b74c7e0cbf68/break-RSA.py
#!/usr/bin/env python3
import sys
# Breaking RSA Encryption using Qiskit Aqua library to run Shor's algorithm (@amarchenkova tutorial)
import qiskit

# Import Shor
from qiskit.aqua.algorithms import Shor

# Import QuantumInstance
from qiskit.aqua import QuantumInstance

# import Aer; a quantum  simulator 
from qiskit.aqua import QuantumInstance

# Set key as the number we want to factor
key = 21 

# Set a random base value that is not a factor of the key
base = 2

# Get the backend of the quantum simulator using Aer.get
# You can replace the 'qasm_simulator' with an actual quantum chip
backend = Aer.get_backend('qasm_simulator')

# Set up a Quantum Instance with a backend
# Set up the number of shots, or runs, of the algorithm
qi = QuantumInstance(backend=backend, shots=1024)

# Call shor with the key, base, and quantum_instance
shors = Shor(N=key, a=base, quantum_instance = qi)

# Call run on shors to get the results
results = shors.run()

# Print the results
print(results['factors'])
