# https://github.com/duckinator/quantum-experiments/blob/00029dd2492f27b4015f29fc2e57ba4c5a65bfb7/coin-flip.py
#!/usr/bin/env python3

from qiskit import Aer, QuantumCircuit, IBMQ, execute
from qiskit.tools.monitor import job_monitor

def _pending_jobs(backend):
    return backend.status().pending_jobs

def _not_simulator(backend):
    return 'simulator' not in backend.name()

def best_ibmq_backend():
    """Returns the actual-hardware IBMQ backend with the fewest pending jobs."""
    provider = IBMQ.get_provider(hub='ibm-q')
    return sorted(provider.backends(filters=_not_simulator), key=_pending_jobs)[0]


# Actual circuit.
circuit = QuantumCircuit(1, 1)
circuit.h(0)  # H operator results in an equal probability of 0 or 1.
circuit.measure(0, 0)

# Connect to IBM Quantum and get the backend to use
IBMQ.load_account()
backend = best_ibmq_backend()

# Run the job
job = execute(circuit, backend, shots=1)

job_monitor(job)

# Get the results.
result = job.result()
counts = result.get_counts(circuit)

if list(counts.keys())[0] == '0':
    print("heads")
else:
    print("tails")
