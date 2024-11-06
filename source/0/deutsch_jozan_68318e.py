# https://github.com/MightyGoldenOctopus/QCOMP/blob/5f06fec4bab01be17eade0a7d5455c8ee0cd87bb/Workshop/WP2/givenFile/deutsch-jozaN.py


import numpy as np

# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute

# import basic plot tools
from qiskit.visualization import plot_histogram

#implementation of the circuit for nqbit and an oracle
def dj(n,oracle):
    pass
    circuit.append(oracle,range(n+1))
    pass

#This an implementation of Deutsch-Joza oracle provided by IBM
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


circuit = dj(7,dj_oracle("constant",7))

print(circuit.draw())

circuit.measure_all()
backend = BasicAer.get_backend('qasm_simulator')
shots = 1024
results = execute(circuit, backend=backend, shots=shots).result()
answer = results.get_counts()
if(answer['00000000'] + answer['10000000']== 1024):
   print("Your Deutsch-Joza algorythm handle well constant oracle")

circuit = dj(7,dj_oracle("balanced",7))
circuit.measure_all()
results = execute(circuit, backend=backend, shots=shots).result()
answer = results.get_counts()
if(answer['11111111'] + answer['01111111'] == 1024):
    print("Your Deutsch-Joza algorythm handle well balanced oracle")  
