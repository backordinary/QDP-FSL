# https://github.com/austinjhunt/qiskit/blob/b1686a5a10f24d30e55b4666ba36bbedc8acd757/tutorials/basic-circuits/c1.py
""" 
Here, we provide an overview of working with Qiskit. Qiskit provides the basic building blocks necessary to program quantum computers. The fundamental unit of Qiskit is the quantum circuit. A basic workflow using Qiskit consists of two stages: Build and Run. Build allows you to make different quantum circuits that represent the problem you are solving, and Run that allows you to run them on different backends. After the jobs have been run, the data is collected and postprocessed depending on the desired output.
"""
import numpy as np 
import os
from qiskit import QuantumCircuit 
from pnglatex import pnglatex
from qiskit.quantum_info import Statevector
from qiskit.visualization import array_to_latex
 
base_dir = os.path.dirname(__file__)

def build_circuit(): 
    # The basic element needed for your first program is the QuantumCircuit. 
    # We begin by creating a QuantumCircuit comprised of three qubits.
    circuit = QuantumCircuit(3)

    # After you create the circuit with its registers, you can add gates (“operations”) to manipulate the registers. 
    # Operations can be added to the circuit one by one, as shown below.
    # Add a H (hadamard) gate on qubit 0, putting this qubit in superposition.
    circuit.h(0)
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    circuit.cx(0, 1)
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
    # the qubits in a GHZ state.
    circuit.cx(0, 2)

    return circuit

def visualize_circuit(circuit): 
    circuit.draw(output='mpl', filename=f'{base_dir}/c1-circuit.png')

def simulate_circuit(circuit, num_qubits):
    """ 
    To simulate a circuit we use the quant_info module in Qiskit. 
    This simulator returns the quantum state, which is a complex 
    vector of dimensions 2^n, where n is the number of qubits (so 
    be careful using this as it will quickly get too large to run on your machine).
    """
    # Set the intial state of the simulator to the ground state using from_int
    state = Statevector.from_int(0, 2**num_qubits)

    # Evolve the state by the quantum circuit
    state = state.evolve(circuit) 
    #Alternative way of representing in latex
    pnglatex(state.draw('latex'), output=f'{base_dir}/c1-state.png')



    


if __name__ == "__main__": 
    circuit = build_circuit()
    visualize_circuit(circuit)
    simulate_circuit(circuit, num_qubits=3)    