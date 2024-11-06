# https://github.com/reecejrobertson/QuantumComputingEmulator/blob/29dcc72c95f56de513234cdc4cb93f48cf949a21/Experiments/qiskit_shors.py
from qiskit import IBMQ
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor
import time

IBMQ.enable_account('99c0e0cf0828e07052aa037e95f66583b58e649173193b0743e9eeb3458cb4ec5ce2980423e36d2a13ca1a9e6cc1ac42c68799865ec41ea58d89964c662a1273') # Enter your API token here
provider = IBMQ.get_provider(hub='ibm-q')

backend = provider.get_backend('ibmq_qasm_simulator') # Specifies the quantum device

N =  67 * 127   # 8509
a = 38

# N = 179 * 239   # 42781
# a = 10

factors = Shor(QuantumInstance(backend, shots=2, skip_qobj_validation=False))

start_time = time.time()
result_dict = factors.factor(N=13*17, a=2) # Where N is the integer to be factored
execution_time = time.time() - start_time
result = result_dict.factors

print(result)
print("Time:", execution_time)