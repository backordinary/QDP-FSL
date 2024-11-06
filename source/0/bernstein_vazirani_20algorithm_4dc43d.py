# https://github.com/Glowman554/quantum-goodnes/blob/c474b958c132ac242221206d66be2e1e5006f171/python/Bernstein-Vazirani%20Algorithm.py
import qiskit as q
import math
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
    job = q.execute(circuit, backend=backend, shots=1000)
    job_monitor(job)
    return(job)

IBMQ.load_account()


s = '110'
n = len(s)
circuit = q.QuantumCircuit(n+1,n)
circuit.x(n)
circuit.barrier()
circuit.h(range(n+1))
circuit.barrier()

for ii, yesno in enumerate(reversed(s)):
    if yesno == '1': 
        circuit.cx(ii, n)
    
circuit.barrier()
circuit.h(range(n+1))
circuit.barrier()
circuit.measure(range(n), range(n))

job = run("ibmqx2", circuit)

result = job.result()
counts = result.get_counts(circuit)
print(counts)

biggest = 0
index = 0

for i in counts:
    if(counts[i] > biggest):
        biggest = counts[i]
        index = i

print("The number is", index)
