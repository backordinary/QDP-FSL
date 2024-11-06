# https://github.com/ajh011/ajh011.github.io/blob/98658c6993c0086579d83ab5024a717431ab407b/products/ipy/quantum/qisims.py
from qiskit import *
from qiskit.visualization import *

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

def runsim(qc, runs=2048):
    backend = BasicAer.get_backend('qasm_simulator')
    tqc = transpile(qc, backend)
    otqc = assemble(tqc, shots = runs)
    result = backend.run(otqc).result()
    plot_histogram(result.get_counts())
    
def checkstate(qct):
    backend2=Aer.get_backend("statevector_simulator")
    result = execute(qct, backend2).result()
    return(result.get_statevector())

