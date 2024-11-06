# https://github.com/arshpreetsingh/Qiskit-cert/blob/7946e8774dfa262264c5169bd8ef14ccb5e406e0/7_Shors_algorithm(Factorization).py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.algorithms import Shor
from read_config import get_api_key

# Get connected to Backend!
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')

# Now Run Shor's Operations!
factors = Shor(backend)
result_dict = factors.factor(8565796576576573)
#circuit = factors.construct_circuit(21)
#print(circuit)
#result = result_dict['factors'] # Get factors from results
print(result_dict)
