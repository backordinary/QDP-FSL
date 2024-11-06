# https://github.com/mistershashanksingh/Basic-Practice-Project/blob/80c0924aa62bde9d010a6436379e2dbc3a6924e8/PYTHON/QUANTUM/quntum/qiskit2.py
from qiskit.aqua.algorithms import Shor  # This qiskit quantum package from IBM
from qiskit.aqua import QuantumInstance
from qiskit import Aer
import time  # I import time package to calculate execution time

start_time = time.time()
key = 21   # number tp factor
base = 2

backend = Aer.get_backend('qasm_simulator')

qi = QuantumInstance(backend=backend, shots=1024)

shors = Shor(N=key, a=base, quantum_instance=qi)
results = shors.run()
print(results['factors'])

end_time = time.time()

print("Total time to execute only above quantum code i.e.", start_time+end_time)

