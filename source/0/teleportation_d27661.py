# https://github.com/agustinsilva447/QuantumComputing/blob/c8b592e722aa7bf451b133bb56dd715d5c057642/curso_basic/teleportation.py
from qiskit import *
from qiskit import IBMQ
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
import matplotlib.pyplot as plt

circuit = QuantumCircuit(3,3)
circuit.x(0) 
circuit.barrier() 

circuit.h(1)
circuit.cx(1,2)
circuit.barrier() 

circuit.cx(0,1)
circuit.h(0)
circuit.barrier() 

circuit.cx(1, 2)
circuit.cz(0, 2)
circuit.barrier() 

circuit.measure([0, 1, 2], [0, 1, 2]) 
print(circuit)

backend1 = Aer.get_backend('qasm_simulator')
job1 = execute(circuit, backend=backend1, shots=1000)
result1 = job1.result()
measurement1 = result1.get_counts(circuit)
plot_histogram(measurement1)

IBMQ.save_account('bc0221c2c435408a718744a0e3388325cd4241375e15beefb8909a759a28201db7c6a425e8d07e09e54105e373aceccb175fa8eb46700b4db88ac50dbdc6fa84')
provider = IBMQ.load_account()

backend2 = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 3 and not b.configuration().simulator and b.status().operational==True))
job2 = execute(circuit, backend=backend2, shots=1000)
result2 = job2.result()
measurement2 = result2.get_counts(circuit)
plot_histogram(measurement2)

plt.show()