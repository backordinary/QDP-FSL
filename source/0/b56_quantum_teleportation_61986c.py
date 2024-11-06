# https://github.com/aadamwaj/qbronzesubmissions/blob/51ec20bbe5e39051087c61183b827f4bd76ee926/Day%204/B56_Quantum_Teleportation.py
#!/usr/bin/env python
# coding: utf-8

# <table> <tr>
#         <td  style="background-color:#ffffff;">
#             <a href="http://qworld.lu.lv" target="_blank"><img src="..\images\qworld.jpg" width="25%" align="left"> </a></td>
#         <td style="background-color:#ffffff;vertical-align:bottom;text-align:right;">
#             prepared by <a href="http://abu.lu.lv" target="_blank">Abuzer Yakaryilmaz</a> (<a href="http://qworld.lu.lv/index.php/qlatvia/" target="_blank">QLatvia</a>)
#         </td>        
# </tr></table>

# <table width="100%"><tr><td style="color:#bbbbbb;background-color:#ffffff;font-size:11px;font-style:italic;text-align:right;">This cell contains some macros. If there is a problem with displaying mathematical formulas, please run this cell to load these macros. </td></tr></table>
# $ \newcommand{\bra}[1]{\langle #1|} $
# $ \newcommand{\ket}[1]{|#1\rangle} $
# $ \newcommand{\braket}[2]{\langle #1|#2\rangle} $
# $ \newcommand{\dot}[2]{ #1 \cdot #2} $
# $ \newcommand{\biginner}[2]{\left\langle #1,#2\right\rangle} $
# $ \newcommand{\mymatrix}[2]{\left( \begin{array}{#1} #2\end{array} \right)} $
# $ \newcommand{\myvector}[1]{\mymatrix{c}{#1}} $
# $ \newcommand{\myrvector}[1]{\mymatrix{r}{#1}} $
# $ \newcommand{\mypar}[1]{\left( #1 \right)} $
# $ \newcommand{\mybigpar}[1]{ \Big( #1 \Big)} $
# $ \newcommand{\sqrttwo}{\frac{1}{\sqrt{2}}} $
# $ \newcommand{\dsqrttwo}{\dfrac{1}{\sqrt{2}}} $
# $ \newcommand{\onehalf}{\frac{1}{2}} $
# $ \newcommand{\donehalf}{\dfrac{1}{2}} $
# $ \newcommand{\hadamard}{ \mymatrix{rr}{ \sqrttwo & \sqrttwo \\ \sqrttwo & -\sqrttwo }} $
# $ \newcommand{\vzero}{\myvector{1\\0}} $
# $ \newcommand{\vone}{\myvector{0\\1}} $
# $ \newcommand{\stateplus}{\myvector{ \sqrttwo \\  \sqrttwo } } $
# $ \newcommand{\stateminus}{ \myrvector{ \sqrttwo \\ -\sqrttwo } } $
# $ \newcommand{\myarray}[2]{ \begin{array}{#1}#2\end{array}} $
# $ \newcommand{\X}{ \mymatrix{cc}{0 & 1 \\ 1 & 0}  } $
# $ \newcommand{\I}{ \mymatrix{rr}{1 & 0 \\ 0 & 1}  } $
# $ \newcommand{\Z}{ \mymatrix{rr}{1 & 0 \\ 0 & -1}  } $
# $ \newcommand{\Htwo}{ \mymatrix{rrrr}{ \frac{1}{2} & \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} & \frac{1}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{1}{2} & -\frac{1}{2} & -\frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} & -\frac{1}{2} & \frac{1}{2} } } $
# $ \newcommand{\CNOT}{ \mymatrix{cccc}{1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0} } $
# $ \newcommand{\norm}[1]{ \left\lVert #1 \right\rVert } $
# $ \newcommand{\pstate}[1]{ \lceil \mspace{-1mu} #1 \mspace{-1.5mu} \rfloor } $

# <h2>Quantum Teleportation</h2>
# 
# [Watch Lecture](https://youtu.be/4PYeoqALKHk)
# 
# <hr>
# 
# _**Prepare a few pieces of papers**_
# - _**to draw the circuit of the following protocol step by step and**_
# - _**to solve some of tasks requiring certain calculations.**_
# 
# <hr>

# Asja wants to send a qubit to Balvis by using only classical communication.
# 
# Let $ \ket{v} = \myvector{a\\b} \in \mathbb{R}^2 $ be the quantum state.
# 
# _Discussion:_ If Asja has many copies of this qubit, then she can collect the statistics based on these qubits and obtain an approximation of $ a $ and $ b $, say $ \tilde{a} $ and $\tilde{b}$, respectively. After this, Asja can send $ \tilde{a} $ and $\tilde{b}$ by using many classical bits, the number of which depends on the precision of the amplitudes. 

# On the other hand, If Asja and Balvis share the entangaled qubits in state $ \sqrttwo\ket{00} + \sqrttwo\ket{11} $ in advance, then it is possible for Balvis to create $ \ket{v} $ in his qubit after receiving two bits of information from Asja. 

# <h3> Protocol </h3>
# 
# The protocol uses three qubits as specified below:
# 
# <img src='../images/quantum_teleportation_qubits.png' width="25%" align="left">

# Asja has two qubits and Balvis has one qubit.
# 
# Asja's quantum message (key) is $ \ket{v} = \myvector{a\\b} = a\ket{0} + b\ket{1} $.
# 
# The entanglement between Asja's second qubit and Balvis' qubit is  $ \sqrttwo\ket{00} + \sqrttwo\ket{11} $.
# 
# So, the quantum state of the three qubits is
# 
# $$ \mypar{a\ket{0} + b\ket{1}}\mypar{\sqrttwo\ket{00} + \sqrttwo\ket{11}} 
#     = \sqrttwo \big( a\ket{000} + a \ket{011} + b\ket{100} + b \ket{111} \big).  $$

# <h4> CNOT operator by Asja </h4>
# 
# Asja applies CNOT gate to her qubits where $q[2]$ is the control qubit and $q[1]$ is the target qubit.

# <h3>Task 1</h3>
# 
# Calculate the new quantum state after this CNOT operator.

# <a href="B56_Quantum_Teleportation_Solutions.ipynb#task1">click for our solution</a>

# <h3>Hadamard operator by Asja</h3>
# 
# Asja applies Hadamard gate to $q[2]$.

# <h3>Task 2</h3>
# 
# Calculate the new quantum state after this Hadamard operator.
# 
# Verify that the resulting quantum state can be written as follows:
# 
# $$  
#     \frac{1}{2} \ket{00} \big( a\ket{0}+b\ket{1} \big) +
#     \frac{1}{2} \ket{01} \big( a\ket{1}+b\ket{0} \big) +
#     \frac{1}{2} \ket{10} \big( a\ket{0}-b\ket{1} \big) +
#     \frac{1}{2} \ket{11} \big( a\ket{1}-b\ket{0} \big) .
# $$

# <a href="B56_Quantum_Teleportation_Solutions.ipynb#task2">click for our solution</a>

# <h3> Measurement by Asja </h3>
# 
# Asja measures her qubits. With probability $ \frac{1}{4} $, she can observe one of the basis states.
# 
# Depeding on the measurement outcomes, Balvis' qubit is in the following states:
# <ol>
#     <li> "00": $ \ket{v_{00}} = a\ket{0} + b \ket{1} $ </li>
#     <li> "01": $ \ket{v_{01}} =  a\ket{1} + b \ket{0} $ </li>
#     <li> "10": $ \ket{v_{10}} =  a\ket{0} - b \ket{1} $ </li>
#     <li> "11": $ \ket{v_{11}} =  a\ket{1} - b \ket{0} $ </li>
# </ol>

# As can be observed, the amplitudes $ a $ and $ b $ are "transferred" to Balvis' qubit in any case.
# 
# If Asja sends the measurement outcomes, then Balvis can construct $ \ket{v} $ exactly.

# <h3>Task 3</h3>
# 
# Asja sends the measurement outcomes to Balvis by using two classical bits: $ x $ and $ y $. 
# 
# For each $ (x,y) $ pair, determine the quantum operator(s) that Balvis can apply to obtain $ \ket{v} = a\ket{0}+b\ket{1} $ exactly.

# <a href="B56_Quantum_Teleportation_Solutions.ipynb#task3">click for our solution</a>

# <h3> Task 4 </h3>
# 
# Create a quantum circuit with three qubits as described at the beginning of this notebook and three classical bits.
# 
# Implement the protocol given above until Asja makes the measurements (included).
# - The state of $q[2]$ can be set by the rotation with a randomly picked angle.
# - Remark that Balvis does not make the measurement.
# 
# At this point, read the state vector of the circuit by using "statevector_simulator". 
# 
# _When a circuit having measurement is simulated by "statevector_simulator", the simulator picks one of the outcomes, and so we see one of the states after the measurement._
# 
# Verify that the state of Balvis' qubit is in one of these: $ \ket{v_{00}}$, $ \ket{v_{01}}$, $ \ket{v_{10}}$, and $ \ket{v_{11}}$.
# 
# Guess the measurement outcome obtained by "statevector_simulator".

# In[2]:


#
# your code is here
#
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi, cos, sin
from random import randrange

q =  QuantumRegister(3,"q") 
c = ClassicalRegister(3,"c") 
qc = QuantumCircuit(q,c)

r = randrange(100)
theta = pi*(r/50)
print("selected angle is",r*3.6,"in degrees and",theta,"in radians")
a = cos(theta)
b = sin(theta)
print("a=",round(a,3),"b=",round(b,3))
print("a*a=",round(a**2,3),"b*b=",round(b**2,3))
qc.ry(2*theta,q[2])

qc.h(q[1])
qc.cx(q[1],q[0])
qc.cx(q[2],q[1])
qc.h(q[2])

qc.measure(q[2],c[2])
qc.measure(q[1],c[1])

display(qc.draw(output='mpl',reverse_bits=True))

job = execute(qc,Aer.get_backend('statevector_simulator'),optimization_level=0,shots=1)
current_quantum_state=job.result().get_statevector(qc)
print("state vector is")
for i in range(len(current_quantum_state)):
    print(current_quantum_state[i].real)
print()

classical_outcomes = ['00','01','10','11']
if (current_quantum_state[2*0].real != 0) or (current_quantum_state[2*0+1].real != 0):
    print("the classical outcome is",classical_outcomes[0])
    classical_outcome = classical_outcomes[0]
    balvis_state = [ current_quantum_state[2*0].real,current_quantum_state[2*0+1].real ]
if (current_quantum_state[2*1].real != 0) or (current_quantum_state[2*1+1].real != 0):
    print("the classical outcome is",classical_outcomes[1])
    classical_outcome = classical_outcomes[1]
    balvis_state = [ current_quantum_state[2*1].real,current_quantum_state[2*1+1].real ]
if (current_quantum_state[2*2].real != 0) or (current_quantum_state[2*2+1].real != 0):
    print("the classical outcome is",classical_outcomes[2])
    classical_outcome = classical_outcomes[2]
    balvis_state = [ current_quantum_state[2*2].real,current_quantum_state[2*2+1].real ]
if (current_quantum_state[2*3].real != 0) or (current_quantum_state[2*3+1].real != 0):
    print("the classical outcome is",classical_outcomes[3])
    classical_outcome = classical_outcomes[3]
    balvis_state = [ current_quantum_state[2*3].real,current_quantum_state[2*3+1].real ]
print()
        
readable_quantum_state = "|"+classical_outcome+">"
readable_quantum_state += "("+str(round(balvis_state[0],3))+"|0>+"+str(round(balvis_state[1],3))+"|1>)"
print("the new quantum state is",readable_quantum_state)


all_states = ['000','001','010','011','100','101','110','111']

        
balvis_state_str = "|"+classical_outcome+">("
for i in range(len(current_quantum_state)):
    if abs(current_quantum_state[i].real-a)<0.000001: 
        balvis_state_str += "+a|"+ all_states[i][2]+">"
    elif abs(current_quantum_state[i].real+a)<0.000001:
        balvis_state_str += "-a|"+ all_states[i][2]+">"
    elif abs(current_quantum_state[i].real-b)<0.000001: 
        balvis_state_str += "+b|"+ all_states[i][2]+">"
    elif abs(current_quantum_state[i].real+b)<0.000001: 
        balvis_state_str += "-b|"+ all_states[i][2]+">"
balvis_state_str += ")"        
print("the new quantum state is",balvis_state_str)


# <a href="B56_Quantum_Teleportation_Solutions.ipynb#task4">click for our solution</a>

# <h3> Task 5 </h3>
# 
# Implement the protocol above by including the post-processing part done by Balvis, i.e., the measurement results by Asja are sent to Balvis and then he may apply $ X $ or $ Z $ gates depending on the measurement results.
# 
# We use the classically controlled quantum operators. 
# 
# Since we do not make measurement on $ q[2] $, we define only 2 classical bits, each of which can also be defined separated.
# 
#     q = QuantumRegister(3)
#     c2 = ClassicalRegister(1,'c2')
#     c1 = ClassicalRegister(1,'c1')
#     qc = QuantumCircuit(q,c1,c2)
#     ...
#     qc.measure(q[1],c1)
#     ...
#     qc.x(q[0]).c_if(c1,1) # x-gate is applied to q[0] if the classical bit c1 is equal to 1
# 
# Read the state vector and verify that Balvis' state is $ \myvector{a \\ b} $ after the post-processing.

# In[3]:


#
# your code is here
#
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from math import pi, cos, sin
from random import randrange

q =  QuantumRegister(3,"q") 
c1 = ClassicalRegister(1,"c1") 
c2 = ClassicalRegister(1,"c2") 
qc = QuantumCircuit(q,c1,c2)

r = randrange(100)
theta = pi*(r/50) 
print("the picked angle is",r*3.6,"degrees and",theta,"radians")
a = cos(theta)
b = sin(theta)
print("a=",round(a,4),"b=",round(b,4))
qc.ry(2*theta,q[2])

qc.h(q[1])
qc.cx(q[1],q[0])

qc.cx(q[2],q[1])
qc.h(q[2])
qc.barrier()

qc.measure(q[2],c2)
qc.measure(q[1],c1)

qc.barrier()

qc.x(q[0]).c_if(c1,1)
qc.z(q[0]).c_if(c2,1)

display(qc.draw(output='mpl',reverse_bits=True))

job = execute(qc,Aer.get_backend('statevector_simulator'),optimization_level=0,shots=1)
current_quantum_state=job.result().get_statevector(qc)
print("the state vector is")
for i in range(len(current_quantum_state)):
    print(round(current_quantum_state[i].real,4))
print()

classical_outcomes = ['00','01','10','11']
if (current_quantum_state[2*0].real != 0) or (current_quantum_state[2*0+1].real != 0):
    print("the classical outcome is",classical_outcomes[0])
if (current_quantum_state[2*1].real != 0) or (current_quantum_state[2*1+1].real != 0):
    print("the classical outcome is",classical_outcomes[1])
if (current_quantum_state[2*2].real != 0) or (current_quantum_state[2*2+1].real != 0):
    print("the classical outcome is",classical_outcomes[2])
if (current_quantum_state[2*3].real != 0) or (current_quantum_state[2*3+1].real != 0):
    print("the classical outcome is",classical_outcomes[3])


# <a href="B56_Quantum_Teleportation_Solutions.ipynb#task5">click for our solution</a>

# <!--
# <h3> Task 6 (optional) </h3>
# 
# Observe that Balvis can also t
# 
# Create a quantum circuit with four qubits and four classical bits.
# 
# Assume that Asja has the first two qubits (number 3 and 2) and Balvis has the last two qubits (number 1 and 0).
# 
# Create an entanglement between qubits 2 and 1.
# 
# Implement the protocol (the state of the qubit can be set by a rotation with randomly picked angle):
# - If Asja teleports a qubit, then set the state of qubit 3.
# - If Balvis teleports a qubit, then set the state of qubit 0.
# -->
