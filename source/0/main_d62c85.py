# https://github.com/Norbeen/quantumFinal/blob/8b532f8a33066444222bd8862bc14749dd3419d6/quantumSimulator/main.py
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor


# provider = IBMQ.load_account()

IBMQ.enable_account(
    '2c48804aa2f1d7c872f4cb2a37a4ec649f24241766d16c4b81061b00f62c2f0419dd04f260fe59be8a9550a93bea7692fa956cc6637565b593ae98e29c77331c')
provider = IBMQ.get_provider(hub='ibm-q')


# backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device
backend = Aer.get_backend('qasm_simulator')
# backend = provider.get_backend('ibmq_16_melbourne') # using quantum backend

print('\n Shors Algorithm')
print('--------------------')
print('\nExecuting...\n')

factors = Shor(21) #Function to run Shor's algorithm  

result_dict = factors.run(QuantumInstance(backend, shots=10, skip_qobj_validation=False))
result = result_dict['factors'] # Get factors from results

print(result)