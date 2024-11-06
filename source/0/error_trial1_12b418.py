# https://github.com/Aarun2/Quantum_Repo/blob/854234af2c4e14774ace90af5a4604507d4b1e50/Qiskit_Tutorials/Error_trial1.py
from qiskit import *
circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure([0, 1, 2], [0,1,2])
circuit.draw()

simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=simulator, shots=102).result()
from qiskit.tools.visualization import plot_histogram
plot_histogram(result.get_counts(circuit))


IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
print(provider.backends())
device = provider.get_backend('ibmq_lima')

job = execute(circuit, backend=device, shots=1024)
print(job.job_id())
from qiskit.tools.monitor import job_monitor

job_monitor(job)


job.status()

