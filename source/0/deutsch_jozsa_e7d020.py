# https://github.com/reverseredoc/SoC-2022/blob/16a5bc0412142b92407eea0ab5169948003a2d09/week2/deutsch_jozsa.py


#deutsch jozsa algorithm

#importing required tools


# initialization
import numpy as np

# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, assemble, transpile

# import basic plot tools
from qiskit.visualization import plot_histogram


def controlledX(String,Oracle):
    for qubit in range(len(String)):
        if String[qubit]=='1':
            Oracle.x(qubit)
            


def oracle(type,n):
    Oracle=QuantumCircuit(n+1)
    if type=="balanced":
        #creating X gates for creating any balanced function
        b = np.random.randint(1,2**n)
        
        # Next, format 'b' as a binary string of length 'n', padded with zeros:
        b_str = format(b, '0'+str(n)+'b')
        controlledX(b_str,Oracle)
        for qubit in range(n):
            Oracle.cx(qubit,n)
        controlledX(b_str,Oracle)
        
    if type=="constant":
        #deciding the value of the function and accordingly changing the phase of the output qubit
        output=np.random.randint(2)
        if output==1:
           Oracle.x(n)
            
    #Making a gate
    oracle_gate=Oracle.to_gate()
    oracle_gate.name="Oracle"
    return oracle_gate

def Hgates(n,dj):
    for qubit in range(n):
        dj.h(qubit)
        
def dj(type,n):
    dj=QuantumCircuit(n+1,n)
    #output qubit setup
    dj.x(n)
    dj.h(n)
    
    #initial hadamand gates
    Hgates(n,dj)
        
    #appending the oracle
    dj.append(type,range(n+1))
    
    Hgates(n,dj)
    
    for i in range(n):
        dj.measure(i,i)
        
    return dj
    
#user input    
n = 10
oracle_gate = oracle('balanced', n)
dj_circuit = dj(oracle_gate, n)
display(dj_circuit.draw())

# use local simulator to check accurcy of result
aer_sim = Aer.get_backend('aer_simulator')
transpiled_dj_circuit = transpile(dj_circuit, aer_sim)
qobj = assemble(transpiled_dj_circuit)
results = aer_sim.run(qobj).result()
answer = results.get_counts()
plot_histogram(answer)
