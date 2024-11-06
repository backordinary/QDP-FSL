# https://github.com/dowusu-antwi/qiskit-tutorials/blob/01ac1b346839324b9ce682c7191c660630381d5f/qiskit-start.py
import numpy as np
print("================================================")
print("Importing qiskit objects...")
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_state_city
print("Import completed.")
print("================================================")
#%matplotlib inline

'''
Stages of Qiskit workflow:
 1. Build: make quantum circuits representing the problem
 2. Execute: run quantum circuits on different backend
After jobs have run, data is collected and postprocessed
'''

# Apply gate operations to make GHZ state: Hadamard on qubit 0, CNOT between
#  qubits 0 and 1, CNOT between qubits 0 and 2
def ghz(circuit):
    '''
    Applies gate operations to make GHZ state.
    '''
    circuit.h(0)        # puts qubit in superposition
    circuit.cx(0, 1)    # puts qubits in Bell state (control, target)
    circuit.cx(0, 2)    # puts qubits in GHZ state


## Simulating circuits using Qiskit Aer (package for simulation)
# Backends summary:
#  Statevector   outputs 2^n dimensional complex vector (will quickly eat space)
#  Unitary       calculates 2^n x 2^n matrix representing gates (requires all 
#                operations to be unitary)
#  OpenQASM      simulates circuit with measurement, mapping to classical bits
def statevector_sim(circuit):
    '''
    Runs quantum circuit on statevector simulator backend.

    Inputs:
     circuit (QuantumCircuit)

    Returns output state.
    '''
    backend = Aer.get_backend('statevector_simulator')     # most common backend
    job = execute(circuit, backend)
    result = job.result()
    status = job.status()
    output_state = result.get_statevector(circuit, decimals=3)
    return output_state


def unitary_sim(circuit):
    '''
    Runs quantum circuit on unitary simulator backend.

    Inputs:
     circuit (QuantumCircuit)

    Returns output state.
    '''
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circuit, backend)
    result = job.result()
    status = job.status()
    output_state = result.get_unitary(circuit, decimals=3)
    return output_state


def qasm_sim(nbits, circuit):
    '''
    Runs quantum circuit with a measurment (i.e., maps to classical bits).

    Inputs:
     nbits (number of quantum circuit bits)
     circuit (QuantumCircuit, of SAME NUMBER (?) bits)

    Returns dictionary mapping bitstrings to measurement probabilities.
    '''

    TRIALS = 1024

    # Generates circuit and maps quantum measurement to classical bits.
    measurement = QuantumCircuit(nbits, nbits)
    measurement.barrier(range(nbits))
    measurement.measure(range(nbits), range(nbits))
    measured_circuit = circuit + measurement

    # Executes circuit and simulates measurement.
    backend = Aer.get_backend('qasm_simulator')
    job = execute(measured_circuit, backend, shots=TRIALS)
    result = job.result()
    counts = result.get_counts(measured_circuit)
    stats = {bitstring: counts[bitstring]/TRIALS for bitstring in counts}
    return stats


## Circuit Basics
circuit = QuantumCircuit(3)    # i.e., acts on quantum register of 3 qubits
ghz(circuit)

# Create quantum program for execution
stats = qasm_sim(3, circuit)
print("Final statistics: ", stats)
#output_state = statevector_sim(circuit)
#print("Final output state: ", output_state)

## Visualization
#circuit.draw('mpl')            # visualize circuit (note, qiskit notation
#                               #  reverses tensor order to fit standard
#                               #  notation for binary bitstrings)
#
#plot_state_city(output_state)  # plots real, imaginary components of state
#                               #  density matrix

if __name__ == "__main__":
    pass
