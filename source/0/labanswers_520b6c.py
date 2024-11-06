# https://github.com/ryangmazur/cmsc22880/blob/1fe1b2880e81aa9b9c1ce32193ed7fa4ed0aefaf/week6/LabAnswers.py
# %% [markdown]
# # Introduction
# This lab continues using the Qiskit Framework, but explores how to create equivalent circuits.
# 
# In these exercises you will be given circuits that you will look for relationships between the gates in order to reorganize, and reduce in gate count, but ultimately still result in the same operation. This is an important step in quantum compilation, as current quantum computers have limited capacity, it is important to use as few gates as possible.
# 
# For the following questions you will create Qiskit circuits to be submitted. There will be no empty circuits (no gates), if the circuit is empty, then you will receive no points.
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
# # Grading:  
# - The output matrix of your circuit will be compared against a baseline circuit, your circuit will be compared against this matrix.
# - If they do not match, we will test the circuit with different inputs and compare against the expected values.
# - You will receive feedback for whether the circuit runs. If it does not, you will receive an error message. If it runs with no message, it means that your circuit runs, but not necessarily that the answer is correct.
# - **Do not change any function names or return types**
# 

# %% [markdown]
# # Exercise 1: Decomposition and Cancellation
# ```
# 
#     ┌───┐          ┌───┐   ┌───┐
# q0: ┤ X ├──────────┤ X ├─X─┤ X ├
#     ├───┤┌───┐┌───┐└─┬─┘ │ └─┬─┘
# q1: ┤ H ├┤ X ├┤ H ├──■───X───■──
#     └───┘└───┘└───┘  
#       
# ```
# Recreate the above circuit, using less gates. Examine how different gates can be broken up, and reorganized, to use less gates overall. It may be helpful to think of how the circuit would transform a given state α|00⟩+β|01⟩+γ|10⟩+𝛿|11⟩), and make sure that both circuits perform the same operation.
# 
# You may include helper functions if needed.
# 

# %%
import qiskit

def hw2_1_response():
    qr1 = qiskit.QuantumRegister(2)
    qc1 = qiskit.QuantumCircuit(qr1)

    # Put your code here (spaces for indentation)
    qc1.x(0)
    qc1.cx(1, 0)
    qc1.swap(0, 1)
    qc1.cx(1, 0)
    qc1.z(0)
    qc1.z(1)
    # End Code
    return qc1
      

# %% [markdown]
# # Exercise 2: Commutative Instructions
# ```
# 
#                    ┌───┐┌───┐┌───┐┌───┐┌───┐     ┌───┐
# q0: ──■─────────■──┤ Z ├┤ X ├┤ X ├┤ X ├┤ Z ├──■──┤ Z ├
#     ┌─┴─┐┌───┐┌─┴─┐├───┤├───┤└─┬─┘├───┤├───┤┌─┴─┐├───┤
# q1: ┤ X ├┤ H ├┤ X ├┤ X ├┤ Z ├──■──┤ Z ├┤ X ├┤ X ├┤ X ├
#     └───┘└───┘└───┘└───┘└───┘     └───┘└───┘└───┘└───┘
#       
# 
# ```
# Recreate the above circuit, using less gates. Examine how different gates can be flipped, to use less gates overall. It may be helpful to think of how the circuit would transform a given state α|00⟩+β|01⟩+γ|10⟩+𝛿|11⟩), and make sure that both circuits perform the same operation.
# 
# You may include helper functions if needed.

# %%
import qiskit

def hw2_2_response():
    qr2 = qiskit.QuantumRegister(2)
    qc2 = qiskit.QuantumCircuit(qr2)

    # Put your code here (spaces for indentation)
    qc2.h(1)
    qc2.swap(0,1)
    qc2.x(0)
    qc2.cz(1,0)
    # End Code
    return qc2

# %% [markdown]
# # Exercise 3: Equivalent Composition
# ```
# 
#     ┌───┐┌───┐     ┌───┐                    ┌───┐
# q0: ┤ H ├┤ Z ├──■──┤ Z ├──■──────────────■──┤ H ├
#     └───┘└───┘┌─┴─┐└───┘┌─┴─┐            │  └───┘
# q1: ──────────┤ X ├──■──┤ X ├──■─────────┼───────
#               └───┘┌─┴─┐└───┘┌─┴─┐┌───┐┌─┴─┐     
# q2: ───────────────┤ X ├─────┤ X ├┤ X ├┤ X ├─────
#                    └───┘     └───┘└───┘└───┘     
#       
# 
# ```
# Recreate the above circuit, using less gates. Look at how you can combine gates to reorganize the circuit and use less gates overall.. It may be helpful to think of how the circuit would transform a given state α|000⟩+β|001⟩+γ|010⟩+𝛿|011⟩+w|100⟩+x|101⟩+y|110⟩+z|111⟩), and make sure that both circuits perform the same operation.
# 
# You may include helper functions if needed.

# %%
import qiskit

def hw2_3_response():
    qr3 = qiskit.QuantumRegister(3)
    qc3 = qiskit.QuantumCircuit(qr3)

    # Put your code here (spaces for indentation)
    qc3.x(2)
    # End Code
    return qc3

# %% [markdown]
# # Submission
# Congratulations on completing the lab!  
# Make sure you:
# 1. Download your lab as a python script (File-> Save and Export Notebook As...->Executable Script
# 2. Rename the downloaded file to **LabAnswers.py**
# 3. Upload the **LabAnswers.py** file to gradescope
# 4. Ensure the autograder runs successfully 


