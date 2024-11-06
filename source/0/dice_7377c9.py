# https://github.com/duckinator/quantum-experiments/blob/00029dd2492f27b4015f29fc2e57ba4c5a65bfb7/dice.py
#!/usr/bin/env python3

import sys
from qiskit import Aer, QuantumCircuit, IBMQ, execute
from qiskit.tools.monitor import job_monitor

def _pending_jobs(backend):
    return backend.status().pending_jobs

def _not_simulator(backend):
    return 'simulator' not in backend.name()

def _has_enough_qubits(backend, n_qubits):
    return backend.configuration().n_qubits >= n_qubits

def best_ibmq_backend(n_qubits):
    """Returns the actual-hardware IBMQ backend with the fewest pending jobs."""
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    return sorted(provider.backends(
        filters=lambda b: _not_simulator(b) and _has_enough_qubits(b, n_qubits)
    ), key=_pending_jobs)[0]


def build_circuit(bits):
    circuit = QuantumCircuit(bits, bits)
    circuit.h(range(bits))  # H operator results in an equal probability of 0 or 1.
    circuit.measure(range(bits), range(bits))

    return circuit

def run(circuit, backend, num_sides):
    number = 0
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    # Run the job
    job = execute(circuit, backend, shots=1)
    job_monitor(job)

    # Get the results.
    result = job.result()
    counts = result.get_counts(circuit)

    number = int(list(counts.keys())[0], 2)
    if number in range(1, num_sides + 1):
        return number
    raise Exception("couldn't get valid die roll in 10 tries; giving up.")

dice = 3
sides = 6

bits = 3  # TODO: Calculate based on `num_sides`.

if sys.argv[0] == "real":
    backend = best_ibmq_backend(n_qubits=bits)
else:
    backend = Aer.get_backend('qasm_simulator')

circuit = build_circuit(bits)
numbers = [run(circuit, backend, sides) for _ in range(dice)]

print(
    f"{dice}d{sides} =",
    " + ".join(map(str, numbers)),
    "=",
    sum(numbers)
)
