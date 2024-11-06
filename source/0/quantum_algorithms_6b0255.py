# https://github.com/olivamadrigal/Jupyter_Notebook/blob/ac846dc42e8c300875166e5951b609b7492311da/quantum_algorithms.py
# initialization
import numpy as np
# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile
# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit_textbook.widgets import dj_widget
from qiskit_textbook.problems import dj_problem_oracle
# Run our circuit on the least busy backend. Monitor the execution of the job in the queue
from qiskit.tools.monitor import job_monitor
# DEUTSCH-JOZA ALGORITHM:
# BOOLEAN ORACLE for which f is unknown, all we know is f is either balance or constant.
# input is arbitrary length n-bit string.
# output is either 1 or 0.
# constant boolean function will always output T or F regardless of input.
# balanced boolean funciton will output as many 0s as 1s over the input.
# CLASSICAL SOLUTION: attempt to test for balanced with queries each consitsing of a different input. If the outputs are different, boom, we are done. It's balanced.
# Otherwise, we must continue testing..
# At least > than half of all possible inputs 2^(n-1) + 1 for 100% confidence or compute a probability that
# f is constant as function of k iputs as 1 - 1/(2^(k-1)). And truncate for a given confidence perfcent.
# QUANTUM SOLUTION: solve 100% confidecne with 1 call to f.
# <--> f is a quantum oracle that maps |x>|y> : |x>|y XOR f(x)>.
# maps the tensor product of x and y to the the tensor product of x with the XOR of y and f(x).
# CREATING QUANTUM ORACLE
dj_widget(size="small", case="balanced")
#https://www.cs.cmu.edu/~odonnell/quantum15/lecture05.pdf

# since f(x) is always 1 or 0 regardless of input, set output to randomly selecte
# 0 or 1.
# set the length of the n-bit input string.
n = 3
# CONSTANT ORACLE
const_oracle = QuantumCircuit(n+1)
output = np.random.randint(2) # random into from [0,2)
if output == 1: # say always 1
    const_oracle.x(n) # set output to |1> else leave as 0.
const_oracle.draw()

# BALANCED ORACLE
balanced_oracle = QuantumCircuit(n+1)

#which controls to wrap:
# Place X-gates
b_str = "101"
for qubit in range(len(b_str)): #0 to n-1
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)
balanced_oracle.draw()

# Use barrier as divider
balanced_oracle.barrier()

# Controlled-NOT gates
for qubit in range(n):
    balanced_oracle.cx(qubit, n)

# finish wrapping the controls in X-gates:
balanced_oracle.barrier()

# Place X-gates
for qubit in range(len(b_str)):
    if b_str[qubit] == '1':
        balanced_oracle.x(qubit)

# Show oracle
balanced_oracle.draw()
# created a balanced oracle! :)


# THE FULL ALGORITHM
# initialize the input qubits in |+>
# and output qubit in |->
dj_circuit = QuantumCircuit(n+1, n)

# Apply H-gates
for qubit in range(n):
    dj_circuit.h(qubit) # H|0> = |+>

# Put output qubit in state |->
dj_circuit.x(n) # \1>
dj_circuit.h(n) # H|1> = |->
dj_circuit.draw()

# let's apply the oracle:
dj_circuit += balanced_oracle
dj_circuit.draw()

# do H on input register since H^-1 = H and measure
# Repeat H-gates
for qubit in range(n):
    dj_circuit.h(qubit)
dj_circuit.barrier()

# Measure
for i in range(n):
    dj_circuit.measure(i, i)

# Display circuit
dj_circuit.draw()
# Let's see the output:
# use local simulator
qasm_sim = Aer.get_backend('qasm_simulator')
run_options = {'shots':2048} #1024 is the default shots.
qobj = assemble(dj_circuit, qasm_sim)
results = qasm_sim.run(qobj,**run_options).result()
answer = results.get_counts()
print(answer)
plot_histogram(answer)
# we can see 0% chanche to measure |000> and hence,
# => f(x) is balanced.

# GENERALIZED CIRCUITS
# generalised function that creates Deutsch-Joza oracles and turns them into quantum gates
# case = 'balanced' or 'constant', and n, the size of the input register:
def dj_oracle(case, n):
    # We need to make a QuantumCircuit object to return
    # This circuit has n+1 qubits: the size of the input,
    # plus one output qubit
    oracle_qc = QuantumCircuit(n+1)
    
    # First, let's deal with the case in which oracle is balanced
    if case == "balanced":
        # First generate a random number that tells us which CNOTs to
        # wrap in X-gates:
        b = np.random.randint(1,2**n)
        # Next, format 'b' as a binary string of length 'n', padded with zeros:
        b_str = format(b, '0'+str(n)+'b')
        # Next, we place the first X-gates. Each digit in our binary string
        # corresponds to a qubit, if the digit is 0, we do nothing, if it's 1
        # we apply an X-gate to that qubit:
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)
        # Do the controlled-NOT gates for each qubit, using the output qubit
        # as the target:
        for qubit in range(n):
            oracle_qc.cx(qubit, n)
        # Next, place the final X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)

    # Case in which oracle is constant
    if case == "constant":
        # First decide what the fixed output of the oracle will be
        # (either always 0 or always 1)
        output = np.random.randint(2)
        if output == 1:
            oracle_qc.x(n)
    
    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle" # To show when we display the circuit
    return oracle_gate

def dj_algorithm(oracle, n):
    dj_circuit = QuantumCircuit(n+1, n)
    # Set up the output qubit:
    dj_circuit.x(n)
    dj_circuit.h(n)
    # And set up the input register:
    for qubit in range(n):
        dj_circuit.h(qubit)
    # Let's append the oracle gate to our circuit:
    dj_circuit.append(oracle, range(n+1))
    # Finally, perform the H-gates again and measure:
    for qubit in range(n):
        dj_circuit.h(qubit)
    
    for i in range(n):
        dj_circuit.measure(i, i)
    
    return dj_circuit

n = 4
oracle_gate = dj_oracle('balanced', n)
dj_circuit = dj_algorithm(oracle_gate, n)
dj_circuit.draw()

# results
transpiled_dj_circuit = transpile(dj_circuit, qasm_sim)
qobj = assemble(transpiled_dj_circuit)
results = qasm_sim.run(qobj).result()
answer = results.get_counts()
plot_histogram(answer)
# we can see 0% chanche to measure |000> and hence,
# => f(x) is balanced.

# Experiment with Real Devices
# Load our saved IBMQ accounts and get the least busy backend device with greater than or equal to (n+1) qubits
API_TOKEN="YOUR_TOKEN"
#https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq
IBMQ.save_account(API_TOKEN, overwrite=True)
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n+1) and not x.configuration().simulator and x.status().operational==True))
print("least busy backend: ", backend)
shots = 1024
transpiled_dj_circuit = transpile(dj_circuit, backend, optimization_level=3)
qobj = assemble(transpiled_dj_circuit, backend)
job = backend.run(qobj)
job_monitor(job, interval=2)
# Get the results of the computation
results = job.result()
answer = results.get_counts()
plot_histogram(answer)


# CREATE A DIFFERENT BALANCED OR CONSTANT ORACLE?
#
#The function dj_problem_oracle (below) returns a Deutsch-Joza oracle for n = 4 in the form of a gate. The gate takes 5 qubits as input where the final qubit (q_4) is the output qubit (as with the example oracles above). You can get different oracles by giving dj_problem_oracle different integers between 1 and 5. Use the Deutsch-Joza algorithm to decide whether each oracle is balanced or constant (Note: It is highly recommended you try this example using the qasm_simulator instead of a real device).
oracle = dj_problem_oracle(1)


#Are you able to create a balanced or constant oracle of a different form?
################################################################################################
################################################################################################
################################################################################################
# GENERALIZED CIRCUITS
# generalised function that creates Deutsch-Joza oracles and turns them into quantum gates
# case = 'balanced' or 'constant', and n, the size of the input register:
def generate_boolean_function(case, n):
    """
    Function creates a BOOLEAN BALANCED OR CONSTANT FUNCTION
    
    :param case: balanced or constant
    :param n: size of input register
    :return: gate from built function f(x).
    """
    fx = QuantumCircuit(n+1)
    # BALANCED BOOLEAN FUNCTION
    if case == "balanced":
        b = np.random.randint(1,2**n)  # generate random number in [1, 2^n)
        b_str = format(b, '0'+str(n)+'b')  # convert random number to binary string
        for qubit in range(len(b_str)):  # parse string bits
            if b_str[qubit] == '1':      # if set, apply x-gate to corresponding input qubit in fx circuit: |0> --> |1>
                fx.x(qubit)
        # all qubits in |1>, will apply x-gate to output :. generating a balanced boolean function
        # this one will output: 1010101... for any input.
        for qubit in range(n):
            fx.cx(qubit, n)
        # apply x-gate to all input qubits in |1>, completes wrap to return them back to |0>.
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                fx.x(qubit)

    # CONSTANT BOOLEAN FUNCTION
    # DECIDED OUR INTERPRESTATION 1ST (ALWAYS 0 OR ALWAYS 1)
    if case == "constant":
        # since output is fixed regardless of input, just set this at random.
        output = np.random.randint(2) # select 1 or 0 at random.
        if output == 1:
            fx.x(n)

    # covert to a gate.
    fx_gate = fx.to_gate()
    fx_gate.name = "fx"
    return fx_gate

def dj_algorithm(fx, n):
    """
    DEUTSCH-JOSZA ALGORITHM - QUANTUM ORACLE THAT TELLS ME IF A GIVEN BOOLEAN FUNCTION
    GURANTEED TO BE CONSTNAT OR BALANCED, IS EITHER IN JUST 1 QUERY.-- O(1) time.
    
    :param fx: boolean function, guranteed to be constant or balanced.
    :param n: size of input register
    :return: resulting circuit after measured.
    """
    dj_circuit = QuantumCircuit(n+1, n) # by default all qubits are initialized to |0>
    dj_circuit.x(n)  # |1>
    
    # Apply Hadamard to all qubits:
    dj_circuit.h(n)  # |->
    for qubit in range(n):
        dj_circuit.h(qubit)  # |+++.....+>
    
    # expand our circuit by connecting fx cirucit
    dj_circuit.append(fx, range(n+1))

    # Apply Hadamard to only input register
    for qubit in range(n):
        dj_circuit.h(qubit)
    
    # Measure the circuit (input register)
    for i in range(n):
        dj_circuit.measure(i, i)
    
    return dj_circuit

n = 4
fx_gate = generate_boolean_function('balanced', n)
dj_circuit = dj_algorithm(fx_gate, n)
dj_circuit.draw()


# use local simulator
qasm_sim = Aer.get_backend('qasm_simulator')
# results
transpiled_dj_circuit = transpile(dj_circuit, qasm_sim)
qobj = assemble(transpiled_dj_circuit)
results = qasm_sim.run(qobj).result()
answer = results.get_counts()
plot_histogram(answer)
# we can see 0% chanche to measure |000> and hence,
# => f(x) is balanced.
