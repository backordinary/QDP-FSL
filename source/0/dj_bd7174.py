# https://github.com/kim5284/nT/blob/7b30ede913d55d9f4af9a463546779b64314cba0/dj.py
#%%




import numpy as np
from qiskit import IBMQ, Aer,QuantumCircuit,execute
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit.visualization import plot_histogram


simulator = Aer.get_backend('qasm_simulator')

def dj_oracle(case, n):
    
    oracle_qc = QuantumCircuit(n+1)
    
    if case == "balanced":
        
        b = np.random.randint(1,2**n)
        b_str = format(b, '0'+str(n)+'b')
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)
        for qubit in range(n):
         
           oracle_qc.cx(qubit, n)
        
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)

    
    if case == "constant":
        
        output = np.random.randint(2)
        if output == 1:
            oracle_qc.x(n)
    
    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle" 

    return oracle_gate

def dj_algorithm(oracle, n):
    dj_circuit = QuantumCircuit(n+1, n)
    
    dj_circuit.x(n)
    dj_circuit.h(n)
    
    for qubit in range(n):
        dj_circuit.h(qubit)
    
    dj_circuit.append(oracle, range(n+1))
    
    for qubit in range(n):
        dj_circuit.h(qubit)
    
    for i in range(n):
        dj_circuit.measure(i, i)
    
    
    return dj_circuit



n=input("n")
n=int(n)


oracle_gate = dj_oracle('balanced', n)
dj_circuit = dj_algorithm(oracle_gate, n)
dj_circuit.draw()
job = execute(dj_circuit, simulator, shots=1000)






result = job.result()

counts = result.get_counts(dj_circuit)
print("\n",counts)
print("\n",result)
dj_circuit.draw()






# %%


