# https://github.com/VashuKochar/QiskitProjects/blob/c16ba4e4e0832f4ba5678e3ee7899163b27fd7a1/Basics/entanglement.py
"""
### Source
https://learn.qiskit.org/course/ch-gates/multiple-qubits-and-entangled-states#multi-qubit-gates

### Problem statement
1) Create a quantum circuit that produces the Bell state:(|01>+|10>)/sqrt(2). 
Use the statevector simulator to verify your result.
2) calculate the unitary of this circuit using Qiskit's simulator. Verify this unitary does in fact perform the correct transformation.

### Refrences
CNOT|0+> = (|00>+|11>)/sqrt(2)

### Solution
CNOT|1+> = (|01>+|10>)/sqrt(2)
"""
import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, assemble
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.visualization import *
from ibm_quantum_widgets import *
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector,array_to_latex
# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()

def initCircuit(nq, inp):    
    # Initialise circuit
    qc = QuantumCircuit(nq)
    qc.reset(range(nq))
    for i in range(len(inp)):
        # Encoding the input
        # print(nq-i-1, " ", inp[i])
        if inp[i]=='1':
            qc.x(nq-i-1)
        elif inp[i]=='+':
            qc.h(nq-i-1)
        elif inp[i]=='-':
            qc.x(nq-i-1)
            qc.h(nq-i-1)
        elif inp[i]=='i':
            qc.h(nq-i-1)
            qc.s(nq-i-1)
        elif inp[i]=='-i':
            qc.x(nq-i-1)
            qc.h(nq-i-1)
            qc.s(nq-i-1)
            
    return qc

def getUnitary(qc):
    backend = Aer.get_backend('unitary_simulator')
    
    res = execute(qc, backend).result()
    unitary = res.get_unitary(qc)
    
    return qc, unitary

def getFinalStates(qc):
    backend = Aer.get_backend('aer_simulator')
    
    qc.save_statevector()
    qobj = assemble(qc)
    result = backend.run(qobj).result()
    final_state = result.get_statevector()
    
    return qc, final_state

# Tests

# qc1 = initCircuit(2, '0+') #(|00>+|11>)/sqrt(2)
qc1 = initCircuit(2, '1+') # (|01>+|10>)/sqrt(2)

qc1.barrier()
# perform gate operations on individual qubits
qc1.cx(0,1)

qc1.barrier()

# Draw circuit
print(qc1)

# Plot blochshere
# qc2,final_state = getFinalStates(qc1)
# print("Final State Vector: ",final_state)

qc3,unit = getUnitary(qc1)
print("Unitary: ",unit)