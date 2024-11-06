# https://github.com/Glowman554/quantum-goodnes/blob/c474b958c132ac242221206d66be2e1e5006f171/python/first_try.py
import qiskit as q
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

#https://quantum-computing.ibm.com/

def list_backends():
    provider = IBMQ.get_provider("ibm-q")
    for backend in provider.backends():
        try:
            qubit_count = len(backend.properties().qubits)
        except:
            qubit_count = "simulated"
        
        print(backend.name(), "has", backend.status().pending_jobs, "qued and", qubit_count, "qubits")

def run(b, circuit):
    provider = IBMQ.get_provider("ibm-q")
    backend = provider.get_backend(b)
    job = q.execute(circuit, backend=backend, shots=500)
    job_monitor(job)
    return(job)

IBMQ.load_account()

circuit = q.QuantumCircuit(2,2) # 2 qubits 2 clasical bits
# currently: 0,0
circuit.x(0)
# 1,0
circuit.cx(0, 1) # cnot flips 2nd qubit value IF first qubit is 1
circuit.measure([0,1], [0,1])
circuit.draw()

list_backends()
job = run(input("Backend> "), circuit)

result = job.result()
counts = result.get_counts(circuit)
print(counts)

