# https://github.com/ryangmazur/cmsc22880/blob/e85b708d0cf4b63b2d8e3abfc3be1edfa34c7db6/week8/LabAnswers.py
# %% [markdown]
# # Introduction
# In this lab you will be exploring applications of entanglement, how to create your own custom gates, analyzing a Qiskit circuit, and adapting a circuit to a device specification.
# 
# # Some helpful programming hints:
# Some helpful programming hints:
# 
# - The line circuit.draw(), where circuit is your Qiskit circuit, will draw out the circuit so you can visualize it. This must be the final call in a cell in order for the circuit to be rendered, alternatively, you can call ```print(circuit)``` at any point to see an ascii representation of the circuit
# - op = qiskit.quantum_info.Operator(circuit) will create an operator object, and op.data will let you look at the overall matrix for a circuit.
# - Keep in mind that Qiskit has a different relationship between the drawing and mathematical representation than we have. Specifically, they place the left-most bit at the bottom rather than at the top. You can [**watch this video**](https://youtu.be/Gf7XFFKS9jE) for more information. This has several implications.
# - If you look at a circuit the way we do, then the state vector ends up being stored as \[00, 10, 01, 11\] rather than \[00, 01, 10, 11\] (where the qubit on top is still the left-most qubit).
# - In reality, though, Qiskit also considers the qubit order to be swapped (little endian), where the top qubit is the least significant (right-most) bit. That is for qubits from top to bottom q0, q1, q2, the bitstring is q2, q1, q0. So the state vector is still \[00, 01, 10, 11\] from this perspective. We can see this in the CX gate.
# 
# ```
# q0_0: ──■──  
#       ┌─┴─┐  
# q0_1: ┤ X ├  
#       └───┘  
# ```
#    
# This ordering also affects the matrix, resulting in the following for CX:  
# ```
# [[1, 0, 0, 0],  
#  [0, 0, 0, 1],  
#  [0, 0, 1, 0],  
#  [0, 1, 0, 0]]  
# ```
# 
# Which will take \[00: w, 01: x, 10: y, 11: z\] to \[00: w, 01: z, 10: y, 11: x\] in little endian form, and \[00: w, 01: y, 10: z, 11: x\] in big endian form (most significant bit first).
# 
# **You are allowed to use Numpy and Networkx in addition to the python standard library**
# 
# # Grading:  
# - The output matrix of your circuit will be compared against a baseline circuit, your circuit will be compared against this matrix.
# - If they do not match, we will test the circuit with different inputs and compare against the expected values.
# - You will receive feedback for whether the circuit runs. If it does not, you will receive an error message. If it runs with no message, it means that your circuit runs, but not necessarily that the answer is correct.
# - **Do not change any function names or return types**
# 
# 

# %% [markdown]
# # Exercise 1: Teleportation
# 
# You are given a circuit with two qubits qubit_pair, represented as a tuple of two qubits, in a Bell state. The entangled pair can be in any possible Bell Pair (i.e., starting in |00>, |01>, |10>, or |11> before being entangled). The circuit also has a third qubit, outside_qubit. Write a function that transfers the state from the outside qubit to the second qubit in the Bell pair.
# 
# bell_pair_start is a two-character string of 0s or 1s representing the start of the Bell pair before they are entangled.
# 
# For ease of grading, please do not add measurement gates to your circuit. It is not explicitly necessary to demonstrate the transfer of state.
# 

# %%
import qiskit

def hw4_1_response(circuit, outside_qubit, qubit_pair, bell_pair_start):
    # Put your code here (spaces for indentation)
    circuit.h(qubit_pair[0])
    circuit.cx(qubit_pair[0], qubit_pair[1])
    circuit.cx(outside_qubit, qubit_pair[0])
    circuit.h(outside_qubit)
    circuit.cx(qubit_pair[0], qubit_pair[1])
    circuit.cz(outside_qubit, qubit_pair[1])
    # End Code
    return circuit

# reg = qiskit.QuantumRegister(3)
# circuit = qiskit.QuantumCircuit(reg)

# hw4_1_response(circuit, reg[0], (reg[1], reg[2]), "00").draw()

# %% [markdown]
# # Exercise 2: Making Gates
# Create a function that, given a list of n-bit codes and the length of the code, creates a gate that acts on n+1 qubits, and implements the Archimedes Oracle. Then add it to an n-qubit circuit, and return the circuit from the function.
# 
# Remember that a Qiskit uses a different ordering of states, where the top qubit is the least significant qubit when creating bitstrings.
# 
# In our convention, the top qubit is the most significant bit, and a Qiskit matrix acting on a three qubit state vector will act on the state vector as if it was \[000, 100, 010, 110, 001, 101, 011, 111\].
# 
# There is documentation on creating your own, custom gate [**here**](https://qiskit.org/documentation/apidoc/extensions.html)

# %%
import numpy as np

def to_bin(val):
    binstring = ""

    while val != 0:
        if val % 2 == 1:
            binstring += "1"
        else:
            binstring += "0"

        val //= 2

    return binstring

def hw4_2_response(circuit, n, codes):
    # Put your code to find the entangled qubits here
    #UPDATE
    matrix = np.zeros((n+1, n+1), dtype=np.int8)
    shit_right = True

    for i in range(0, n+1):
        if (to_bin(i) in codes):
            if shift_right:
                matrix[i, i+1] = 1
                shift_right = False
            else:
                matrix[i, i-1] = 1
                shift_right = True
        else:
            matrix[i, i] = 1

    my_gate = qiskit.UnitaryGate(matrix)

    circuit.append(my_gate)

    # Put your code here (spaces for indentation)
    # End Code

    return circuit

# %% [markdown]
# # Exercise 3: Analyzing Circuits
# 
# Write a function that given an n qubit circuit, returns a length n bitstring presenting the code for the Bernstein Vazarani Oracle embedded in the circuit, and the target of the oracle. Your bitstring should use an "x" to represent the location of the target, and 0s and 1s to represent the rest of the code and treat Qubit 0 as the most signficant bit.
# 
# The only CX gates included in this circuit are involved in the oracle. You should do this without simulating the circuit, only analyzing the different gates in the circuit.
# 
# You can examine the different operations in a circuit with a for loop over the circuit: for i in circuit: and i will be a tuple of the form (operation, qubits, classical bits).

# %%
import qiskit

def hw4_3_response(circuit):
    # Put your code to find the entangled qubits here
    bitstring = "0" * circuit.num_qubits
    bitlist = list(bitstring)

    for gate in circuit:
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index

            bitlist[control] = "1"
            bitlist[target] = "x"
    # Put your code here (spaces for indentation)
    # End Code

    return "".join(bitlist)

# %% [markdown]
# # Exercise 4: Routing a Circuit
# You will be given a random 8 qubit circuit and a connection graph, both as a NetworkX and a dictionary of lists, where the key is a qubit, and the value is a list of qubits it is connected to.
# 
# Your goal is to create a function to decompose the circuit to 1 and 2 qubit gates.
# 
# Then create a function, using the circuit you just created to recreate the circuit, and add SWAPs as necessary to make sure that the resulting circuit fits the restraints of the connection graph, while still ultimately performing the same actions.
# 
# This is a very necessary step of compilation for NISQ era quantum computing, they are sparsely connected, so it is not possible for every qubit to interact with every other qubit. On many machines, 3 qubit gates must be rewritten in terms of their 1 and 2 qubit gate decompositions.
# 
# Programming Notes
# Qiskit provides many utility "passes" which are algorithms that act on a circuit to analyze, alter or optimize the circuit. These passes work on a Directed Acyclic Graph or DAG, created from the circuit.
# 
# To convert a circuit to a DAG use the code snippet dag = qiskit.converters.circuit_to_dag(circuit), and convert a DAG to a circuit with c = qiskit.converters.dag_to_circuit(dag).
# 
# To decompose a circuit, you can use the pass found at qiskit.transpiler.passes.Unroll3qOrMore where the documentation can be found here.
# 
# If you would like, you can use the NetworkX graph with qubits as nodes, or dictionary based graph, with qubits as the keys, and a list of neighboring qubits as the values as the basis of your transformations.
# 
# Information for graph algorithms you will need using NetworkX, or other functions can be found here. Specifically, you will likely need:
# - Neighbors [**docs**](https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.neighbors.html?highlight=neighbors#networkx.Graph.neighbors)
# - Shortest Path [**docs**](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html#networkx.algorithms.shortest_paths.generic.shortest_path) Can be used as nx.shortest_path.
# 
# You may include helper functions if needed.

# %%
def decompose_circuit(circuit):
    # return decomposed_circuit
    return circuit

def hw4_4_response(original_circuit, reg, connection_graph_nx, connection_graph_dictionary):
    new_c = qiskit.QuantumCircuit(reg)
    # Put your code here
    # End Code

    return new_c
      

# %% [markdown]
# # Submission
# Congratulations on completing the lab!  
# Make sure you:
# 1. Download your lab as a python script (File-> Save and Export Notebook As...->Executable Script
# 2. Rename the downloaded file to **LabAnswers.py**
# 3. Upload the **LabAnswers.py** file to gradescope
# 4. Ensure the autograder runs successfully 


