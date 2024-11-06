# https://github.com/omaralnamshan/finalproject/blob/ce9a77c94e2cec9abc7c32020e7b9ff39849da18/final.py
from qiskit import IBMQ

from qiskit.aqua import QuantumInstance

from qiskit.aqua.algorithms import Shor

IBMQ.enable_account('ENTER API TOKEN HERE') # Enter your API token here

provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device

factors = Shor(1499)

result_dict = factors.run(QuantumInstance(backend, shots=1, skip_qobj_validation=False))

result = result_dict['factors'] # Get factors from results

print(result)

factors = Shor(1531)

result_dict = factors.run(QuantumInstance(backend, shots=1, skip_qobj_validation=False))

result = result_dict['factors'] # Get factors from results

print(result)