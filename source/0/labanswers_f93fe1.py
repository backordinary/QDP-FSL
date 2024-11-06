# https://github.com/ryangmazur/cmsc22880/blob/734e265dbf52c79a0b04fba005d3f74e1211b80d/week7/LabAnswers.py
# %% [markdown]
# # Introduction
# In this lab you will be exploring how to create entangled circuits in Qiskit and simulating those circuits in Qiskit by using one of the built-in simulators.
# 
# You will also be using the properties of entanglement to solve small problems.
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
# # Exercise 1a: Bell Pairs
# 
# Write a function that given a circuit and two qubits in the circuit, creates a maximally-entangled state (called a Bell state) between the two qubits.
# 
# You may assume that the two qubits in the given circuit are initially in the state |00>. Your function should return the completed circuit. Also, note that there are four different Bell states. Your circuit should create the Bell state |Φ+> which is the one that we called "same entangle without phase".
# 
# You may include helper functions if needed.
# 

# %%
import qiskit

def hw3_1a_response(circuit, qubit1, qubit2):
    # Put your code here (spaces for indentation)
    circuit.h(qubit1)
    circuit.cx(qubit1,qubit2)

    return circuit

# reg = qiskit.QuantumRegister(2)
# circuit = qiskit.QuantumCircuit(reg)

# print(hw3_1a_response(circuit, 1, 0))

# %% [markdown]
# # Simulating a Quantum Circuit
# 
# Now that we have a mechanism to create a Bell pair, let's test it. Qiskit includes a quantum circuit simulator, and since most classical computers can easily simulate 4 to 5 error-free qubits, let's see what happens!
# 
# # Setting up the Simulator
# 
# You can simulate a quantum circuit with the following lines of code:
# ```
# from qiskit.providers.aer import QasmSimulator  
#   
# simulator = QasmSimulator()
# executed_job = qiskit.execute(circuit, simulator, shots=1024)
# ```
# The first line tells Qiskit to use one of its simulators called QasmSimulator. The second line runs the simulator on the circuit. The documentation for the execute function can be found here, which can be used to run simulators (or actual devices) on quantum circuits. It's also possible to specify the number of trials (the number of shots) to run. The execute function returns a Job object.
# 
# Extracting Results from a Simulation
# 
# The job object has a result() function that gives you the results of the simulator.
# ```
# executed_job.result()
# ```
# This returns a result object. See the documentation here.
# 
# Next, we want to extract the counts from a given experiment. The circuit needs to record a measurement outcome at the end of each trial, so when you create your circuit, you must add a measurement at the end to record the outcome in a classical register. A simple way of doing this can be done with the QuantumCircuit.measure_all() function which creates the classical register for you. Alternatively, you can add the classical register yourself and use the QuantumCircuit.measure() function. See the Qiskit documentation for details.
# 
# You can get a dictionary of results from the result object using get_counts:
# ```
# r = executed_job.result()
# counts = r.get_counts(circuit)
# ```
# If printed out, the results dictionary will look like: 
# {"bitstring1": count_1, "bitstring2": count_2, ... , "bitstringn": count_n}. 
# The bitstrings in the dictionary represent the possible measurement outcomes of the quantum circuit.
# 
# # Other Notes
# 
# - You can use the plot_histogram(counts) function (see documentation **[here](https://qiskit.org/documentation/stubs/qiskit.visualization.plot_histogram.html)) to visualize the results of a simulation once you have extracted the counts from the result object.
# - When looking at measurement results, Qiskit treats the classical bits as "little endian", that is the first classical bit, is the least significant bit. So, the following classical bits:
# ```
# c0: ----
# c1: ----
# c2: ----
# ```
# Corresponds to a bitstring of "c2,c1,c0". Make sure to take this into account when looking at any results.
# 
# # Exercise 1b: Simulating Bell Pairs
# 
# Now that you have played around with the simulator function, how to control how many trials (or shots) it performs, and how to extract the counts from the result of the simulation, you are ready for the next exercise.
# 
# A Bell state is a maximally-entangled two-qubit state that produces correlations in the measurement outcomes. You can see these correlations in the results dictionary produced by the simulator in Qiskit.
# 
# For this exercise, write a function that takes a number of shots, creates a two-qubit circuit that produces a Bell Pair ("same entangle" state |Φ+>) and then simulates that circuit. Your function must return the circuit and the results dictionary.
# 
# You may include helper functions if needed.

# %%
import qiskit
from qiskit.providers.aer import QasmSimulator

def hw3_1b_response(num_shots):
    # Put your code here (spaces for indentation)
    simulator = QasmSimulator()
    reg = qiskit.QuantumRegister(2)
    circuit = qiskit.QuantumCircuit(reg)

    circuit.h(0)
    circuit.cx(0,1)
    circuit.measure_all()

    executed_job = qiskit.execute(circuit, simulator, shots=num_shots)
    r = executed_job.result()
    result_dict = r.get_counts(circuit)
    # End Code
    return circuit, result_dict

# hw3_1b_response(1024)

# %% [markdown]
# # Exercise 1c: Simulating Superposition
# 
# 0.0/10.0 points (graded)
# A uniform superposition of multiple qubits can be created by using Hadamard gates in a quantum circuit. If all the qubits are initially in the |0> state, then after a Hadamard gate is applied to each qubit, the resulting state is a uniform superposition which is not entangled and does not produce correlations when the qubits are measured. Instead, all possible measurement outcomes will be equally likely to occur.
# 
# Write a function that takes a number of shots, creates a two-qubit circuit with two Hadamard gates (one Hadamard gate applied to each qubit), simulates that circuit, and returns the circuit and the results dictionary.
# 
# You may include helper functions if needed.

# %%
import qiskit
from qiskit.providers.aer import QasmSimulator

def hw3_1c_response(num_shots):
    # Put your code here (spaces for indentation)
    simulator = QasmSimulator()
    reg = qiskit.QuantumRegister(2)
    circuit = qiskit.QuantumCircuit(reg)

    circuit.h(0)
    circuit.h(1)
    circuit.measure_all()

    executed_job = qiskit.execute(circuit, simulator, shots=num_shots)
    r = executed_job.result()
    result_dict = r.get_counts(circuit)
    # End Code
    return circuit, result_dict

# hw3_1b_response(1024)

# %% [markdown]
# # Some Quick Notes on Programming in Python
# 
# This section assumes you have general knowledge of programming concepts. It is merely here to introduce Python syntax you may be unfamiliar with.
# 
# ## Range
# 
# To create a for loop over a sequence of numbers, k to n, use for i in range(k, n+1), with i being the item in the sequence for that iteration of the loop.
# 
# ## Dictionary
# 
# You will likely need to iterate over a dictionary. To do this, you can use a for loop to iterate over the keys to get the value. Use for key in dictionary: to get the keys, and use dictionary\[key\] to get the values.
# 
# ## String
# 
# You will also likely need to work with string literals. To get a specific character in a Python string, use string\[i\]. This gives the ith character of the string. Python indices start from 0. You can iterate over the characters of a string.
# 
# There is one more trick with strings to get a range of characters. For example, to get characters 2 to 4, use the following syntax: string\[2:5\]. We use string\[start:end\] to get the characters starting at start and ending at end-1.
# 
# You can get the length of any item that supports it by using len(datastructure)
# 
# ## Conditional Statements
# 
# In addition to for loops, you will need conditionals. This is a statement that performs the code inside of an indented block only if a certain given condition is met. These are of the form:
# ```
# if condition:
#   action to perform
# ```
# If there is an alternative action you want to perform only if condition was not met, you can use an else statement underneath.
# ```
# if condition:
#   action to perform
# else:
#   action to perform if the condition was not true
# ```
# These are a basic primer of Python syntax that you will likely need. W3 Schools has excellent demos/tutorials for showing off Python's features and how to use them. Additionally, many of these techniques can be found in the linked notebook.
# 
# # Exercise 2: Find the Entangled Qubit
# 
# In this exercise, you will be given a 6-qubit circuit and you will write a function that finds the two qubits that are entangled.
# 
# The circuit creates a pair of entangled qubits, then moves or separates those entangled qubits through a set of swap gates from their original position in a 6-qubit system. We will make the simplifying assumption that when the qubits became entangled, they started in the |0> states, meaning they are guaranteed to be in the "same entangle" state |Φ+>. How could one figure out which of these 6 qubits will be entangled? Using the simulation and analyzing the results! While 4 of the qubits will be in a 50/50 superposition resulting in random results, two of the qubits will be entangled and have correlated outputs to each other.
# 
# You are responsible for creating the function that determines which two qubits are in the "same entangle" state. It takes the circuit, and, using your knowledge of the properties of entanglement, returns the pair of qubit indices (based on their location in the bitstring) that are entangled.
# 
# Your result from hw3_2_response should be in terms of the original quantum bits, not the classical bits. The top qubit should be considered qubit 0 and the bottom qubit should be considered qubit 5.
# 
# You may include helper functions if needed.

# %%
import qiskit
from qiskit.providers.aer import QasmSimulator

def hw3_2_response(circuit):
    # Put your code to find the entangled qubits here
    print("In hw3_2_response")
    simulator = QasmSimulator()

    circuit.measure_all()

    executed_job = qiskit.execute(circuit, simulator, shots=1024)
    r = executed_job.result()
    result_dict = r.get_counts(circuit)

    bitstrings = ["000001", "000010", "000100", "001000", "010000", "100000"]
    qubit_1 = -1
    qubit_2 = -1
    one_or_two = 1
    if result_dict.get(bitstrings[0]) == None:
        if one_or_two == 1:
            qubit_1 = 0
            one_or_two = 2
        else:
            qubit_2 = 0
    if result_dict.get(bitstrings[1]) == None:
        if one_or_two == 1:
            qubit_1 = 1
            one_or_two = 2
        else:
            qubit_2 = 1
    if result_dict.get(bitstrings[2]) == None:
        if one_or_two == 1:
            qubit_1 = 2
            one_or_two = 2
        else:
            qubit_2 = 2
    if result_dict.get(bitstrings[3]) == None:
        if one_or_two == 1:
            qubit_1 = 3
            one_or_two = 2
        else:
            qubit_2 = 3
    if result_dict.get(bitstrings[4]) == None:
        if one_or_two == 1:
            qubit_1 = 4
            one_or_two = 2
        else:
            qubit_2 = 4
    if result_dict.get(bitstrings[5]) == None:
        if one_or_two == 1:
            qubit_1 = 5
            one_or_two = 2
        else:
            qubit_2 = 5
    # Put your code here (spaces for indentation)
    # End Code
    return qubit_1, qubit_2

# reg = qiskit.QuantumRegister(6)
# circuit = qiskit.QuantumCircuit(reg)

# circuit.h(0)
# circuit.cx(0,1)
# circuit.h(2)
# circuit.h(3)
# circuit.h(4)
# circuit.h(5)
# print(hw3_2_response(circuit))

# %% [markdown]
# # Exercise 3: Different EPR Pairs
# This exercise is identical to Exercise 2 except for one major difference: the two entangled qubits are not necessarily in the "same entangle" state. We know that they were not in superposition when they became entangled, but we do not know which of the states |00>, |01>, |10>, or |11> they were in when they became entangled.
# 
# The goal is to find the two qubits that are entangled in a 6-qubit circuit. In order to do this, you are going to write two functions.
# 
# - First, write a function that will convert the original |00> input to the desired |xy> output. The output of this function will then become the input to the previous problem's circuit that entangles two qubits and then moves them.
# 
# Using gates to adjust the starting state of the circuit, fill in the function prime_circuit that "primes" the entangled qubits to the given state based on a bitstring of 0s and 1s. (You can assume that bitstring will not be longer than qubit_list, the quantum register for the circuit.) That is, all of the qubits in the circuit start in the state |0>, and if there is a "1" in the corresponding bitstring, that qubit should be changed to be in the |1> state. The first character in the bitstring corresponds with the first qubit in the circuit, and the rest of the qubits follow.
# 
# The second function you need to adjust is the function that finds the entangled pair. Now you cannot assume the outputs of those two bits will always be the same as you did above, merely that they will always have a relationship corresponding with how entanglement correlates entangled outputs based on one of the above pre-entanglement starting states (|00>, |01>, |10>, or |11>). So think about what the possible relationships are (there aren't many) and search for all of those relationships in the outputs. Just like last time, all of the qubits in the circuit are in a 50/50 superposition state, so you have to look at the relationships between different pairs of outputs, not just the values of the outputs.
# Your result from hw3_3_response should be in terms of the original quantum bits, not the classical bits. The top qubit should be considerd qubit 0 and the bottom qubit should be considered qubit 5.
# 
# You may include helper functions if needed.

# %%
import qiskit
from qiskit.providers.aer import QasmSimulator

def prime_circuit(circuit, qubit_list, bitstring):
    for i in range(0,len(bitstring)):
        if bitstring[i] == '1':
            circuit.x(qubit_list[i])

    return circuit

def hw3_3_response(circuit):
    # Put your code to find the entangled qubits here
    print("in hw3_3_response")
    simulator = QasmSimulator()

    circuit.measure_all()

    executed_job = qiskit.execute(circuit, simulator, shots=10000)
    r = executed_job.result()
    result_dict = r.get_counts(circuit)
    print(result_dict)

    bitstrings = ["000001", "000010", "000100", "001000", "010000", "100000"]
    qubit_1 = -1
    qubit_2 = -1
    one_or_two = 1
    if result_dict.get(bitstrings[0]) == None:
        if one_or_two == 1:
            qubit_1 = 0
            one_or_two = 2
        else:
            qubit_2 = 0
    if result_dict.get(bitstrings[1]) == None:
        if one_or_two == 1:
            qubit_1 = 1
            one_or_two = 2
        else:
            qubit_2 = 1
    if result_dict.get(bitstrings[2]) == None:
        if one_or_two == 1:
            qubit_1 = 2
            one_or_two = 2
        else:
            qubit_2 = 2
    if result_dict.get(bitstrings[3]) == None:
        if one_or_two == 1:
            qubit_1 = 3
            one_or_two = 2
        else:
            qubit_2 = 3
    if result_dict.get(bitstrings[4]) == None:
        if one_or_two == 1:
            qubit_1 = 4
            one_or_two = 2
        else:
            qubit_2 = 4
    if result_dict.get(bitstrings[5]) == None:
        if one_or_two == 1:
            qubit_1 = 5
            one_or_two = 2
        else:
            qubit_2 = 5

    if qubit_1 != -1 and qubit_2 != -1:
        print("returning same entangle")
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[0] == '1'
    bitstring[1] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 1
        qubit_2 = 0
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[0] == '1'
    bitstring[2] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 2
        qubit_2 = 0
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[0] == '1'
    bitstring[3] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 3
        qubit_2 = 0
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[0] == '1'
    bitstring[4] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 4
        qubit_2 = 0
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[0] == '1'
    bitstring[5] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 5
        qubit_2 = 0
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[1] == '1'
    bitstring[1] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 1
        qubit_2 = 1
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[1] == '1'
    bitstring[2] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 2
        qubit_2 = 1
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[1] == '1'
    bitstring[3] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 3
        qubit_2 = 1
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[1] == '1'
    bitstring[4] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 4
        qubit_2 = 1
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[1] == '1'
    bitstring[5] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 5
        qubit_2 = 1
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[2] == '1'
    bitstring[1] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 1
        qubit_2 = 2
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[2] == '1'
    bitstring[2] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 2
        qubit_2 = 2
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[2] == '1'
    bitstring[3] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 3
        qubit_2 = 2
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[2] == '1'
    bitstring[4] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 4
        qubit_2 = 2
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[2] == '1'
    bitstring[5] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 5
        qubit_2 = 2
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[3] == '1'
    bitstring[1] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 1
        qubit_2 = 3
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[3] == '1'
    bitstring[2] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 2
        qubit_2 = 3
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[3] == '1'
    bitstring[3] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 3
        qubit_2 = 3
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[3] == '1'
    bitstring[4] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 4
        qubit_2 = 3
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[3] == '1'
    bitstring[5] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 5
        qubit_2 = 3
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[4] == '1'
    bitstring[1] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 1
        qubit_2 = 4
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[4] == '1'
    bitstring[2] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 2
        qubit_2 = 4
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[4] == '1'
    bitstring[3] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 3
        qubit_2 = 4
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[4] == '1'
    bitstring[4] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 4
        qubit_2 = 4
        return qubit_1, qubit_2
    bitstring = "000000"
    bitstring[4] == '1'
    bitstring[5] == '1'
    print(bitstring)

    if result_dict.get(bitstring) == None:
        qubit_1 = 5
        qubit_2 = 4
        return qubit_1, qubit_2

    # Put your code here (spaces for indentation)
    # End Code

# reg = qiskit.QuantumRegister(6)
# circuit = qiskit.QuantumCircuit(reg)

# circuit.h(0)
# circuit.cx(0,1)
# circuit.h(2)
# circuit.h(3)
# circuit.h(4)
# circuit.h(5)
# print(hw3_3_response(circuit))


# %% [markdown]
# # Exercise 4: Entangling Multiple Qubits
# Extend the idea of creating a Bell pair, and create a function that, given an integer n, returns a circuit of that size in which all qubits at the output are entangled together in a maximally-entangled state. We assume that the input can be in any of the 2^n states |00...0> to |11...1>. The circuit will always output a maximally-entangled state.
# 
# For example, given a three-qubit circuit that is initially in the |000⟩ state, your circuit output would be in the state (|000⟩ + |111⟩)/√(2)
# 
# For ease of grading, your strategy should also ensure that given an input state of |111⟩, the resulting output state is (|010⟩ - |101⟩)/√(2)
# 
# Some more example states:
# ```
# |001⟩ -> (|001⟩ + |110⟩)/√(2)
# 
# |010⟩ -> (|100⟩ + |011⟩)/√(2)
# 
# |100⟩ -> (|000⟩ - |111⟩)/√(2)
# 
# |101⟩ -> (|001⟩ - |110⟩)/√(2)
# ```
# You may include helper functions if needed.

# %%
import qiskit

def hw3_4_response(n: int):
    # Put your code here
    reg = qiskit.QuantumRegister(n)
    circuit = qiskit.QuantumCircuit(reg)

    circuit.h(0)

    for i in range(1, n):
        circuit.cx(i-1, i)
    # End Code

    return circuit

# print(hw3_4_response(2))

# %% [markdown]
# # Submission
# Congratulations on completing the lab!  
# Make sure you:
# 1. Download your lab as a python script (File-> Save and Export Notebook As...->Executable Script
# 2. Rename the downloaded file to **LabAnswers.py**
# 3. Upload the **LabAnswers.py** file to gradescope
# 4. Ensure the autograder runs successfully 


