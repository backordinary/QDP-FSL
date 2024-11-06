# https://github.com/tsrobinson/quantum/blob/13115571ccbf50b6b5d5f64954b83832ed3155ed/quantum_addition.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

sim = AerSimulator()

qc = QuantumCircuit(2,2)
qc.x(0) # i.e. qubit 0 is now 1
qc.cx(0,1) # i.e. CNOT (or XOR) where first arg is input and second arg is target, where result is assigned

qc.measure([0,1], [0,1])
display(qc.draw())

job = sim.run(qc)
result = job.result()
result.get_counts()

# Note this returns '11' since x-gate converts qubit 0 to 1. Then target of CNOT is qubit 1, and since CNOT is true (1 XOR 0) qubit 1 also assigned 1
# To correct:

qc = QuantumComputer(3,2)
qc.x(0)
qc.cx(0,2)
qc.cx(1,2)
qc.measure([1,2],[1,2])

# But this won't take into account the second order digit, so the half adder needs another qubit!

qc = QuantumCircuit(4,2)
qc.cx(0,2)
qc.cs(1,2)
qc.ccx(0, 1, 3) # So here, combine two XORs to get first order digit, then use .ccx (AND) to get second order digit

# We only want those last two qubits, so just measure those:

qc.measure(2,0)
qc.measure(3,1)
qc.draw()

job = sim.run(qc)
result = job.result()
result.get_counts()

# Code into a function:

def half_adder(bit1, bit2):

    qc = QuantumComputer(4,2)
    if bit1 == 1:
        qc.x(0)
    if bit2 == 1:
        qc.x(1)
    
    qc.cx(0,2) # XOR gate
    qc.cx(1,2) # XOR gate
    qc.ccx(0,1,3) # AND gate (Tiffoli gate)

    qc.measure(2,0)
    qc.measure(3,1)

    job = sim.run(qc)
    result = job.result()
    return result.get_counts()