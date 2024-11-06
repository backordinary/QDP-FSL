# https://github.com/KartikChinda/QuantumComputing/blob/b6d50a244fb41d096cfda09baf4f2c3a8898b5b8/Shors_Algorithm_Quantum.py
from qiskit import QuantumCircuit, Aer, IBMQ, execute
from qiskit.aqua import QuantumInstance, quantum_instance
from qiskit.aqua.algorithms import Shor

# Enter your API token here
IBMQ.enable_account(
    'your key goes in here')
provider = IBMQ.get_provider(hub='ibm-q')

# Specifies the quantum device
backend = Aer.get_backend('qasm_simulator')

print('\n Shors Algorithm')
print('--------------------')
print('\nExecuting...\n')

quantum_instance = QuantumInstance(backend, shots=1000)
# Function to run Shor's algorithm where 21 is the integer to be factored
my_shor = Shor(N=15, a=2, quantum_instance=quantum_instance)


Shor.run(my_shor)
print('\nPress any key to close')
input()
