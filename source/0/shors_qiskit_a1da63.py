# https://github.com/AmitNikhade/Q_algo/blob/a63528b28704fabd1eb98444b9671ecc61f7dffb/Shors_qiskit.py

from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor


IBMQ.enable_account('5732cdb162a69b61f9c1bddaea9f32b38a302d7d98ea6ffa3ce50d1c667428b4ea77b1737f62b6f99c0f35c31ce6e068e07c617caa65ad8fbecc174dfe627397') # Enter your API token here
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device



factors = Shor(21) #Function to run Shor's algorithm where 21 is the integer to be factored

result_dict = factors.run(QuantumInstance(backend, shots=1, skip_qobj_validation=False))
Factors = result_dict['factors'] # Get factors from results

print(Factors)

