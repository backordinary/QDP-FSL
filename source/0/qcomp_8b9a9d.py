# https://github.com/neocarton/demo-quantum-hello/blob/bc5042830f89b45b4ee7342d3c174967c6a346d4/qcomp.py
# Reference:
# https://github.com/Qiskit/qiskit-community-tutorials/blob/master/hello_world/hello_zero.ipynb
# https://delapuente.github.io/qiskit-textbook

from qiskit import (IBMQ, providers, Aer, execute)
from qiskit.tools.visualization import plot_bloch_multivector
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def fetchBackend() :
    # IBM Q backend
    IBMQ.load_accounts()
    backend = providers.ibmq.least_busy(IBMQ.backends(simulator = False))
    print("Using the least busy device:", backend.name())
    return backend

def qexec(circuit, backend = None, shots = 1000) :
    if backend is None :
        backend = defaultBackend
    # Execute the circuit on the qasm backend
    job = execute(circuit, backend, shots = shots)
    # Grab results from the job
    result = job.result()
    # Print
    print_result(circuit, result)
    # Draw
    draw_result(circuit, result)

def print_result(circuit, result) :
    print(circuit)
    print("Counts :", result.get_counts(circuit))
    # print("State vectors:", result.get_statevector())

def draw_result(circuit, result) :
    circuit.draw(output = "mpl")
    plot_histogram(result.get_counts(circuit))
    # plot_bloch_multivector(result.get_statevector())

# List IBMQ backends
# provider = IBMQ.get_provider(hub='ibm-q')
# provider.backends()

# List simulators
# unitary_simulator
# statevector_simulator
# qasm_simulator
print(Aer.backends())

defaultBackend = Aer.get_backend("qasm_simulator")
print("Default backend:", defaultBackend.name())
