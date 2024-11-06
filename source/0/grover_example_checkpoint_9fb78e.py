# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/.ipynb_checkpoints/grover_example-checkpoint.py
from qiskit import QuantumCircuit, qiskit, Aer
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalfunction import ClassicalFunction

def oracle(n):
    # Define the logical circuit for the oracle.
    prog = '''def oracle_func(x1: Int1, x2: Int1, x3: Int1) -> Int1:
                return (not x1 and not x2 and not x3) or (x1 and x2 and not x3)'''

    # Convert the logic to a quantum circuit.
    formula = ClassicalFunction(prog)
    fc = formula.synth()

    # Convert the quantum circuit to a quantum program.
    qc = QuantumCircuit(n+1)
    qc.compose(fc, inplace=True)

    #print(qc.draw())

    # Convert the oracle to a gate.
    gate = qc.to_gate()
    gate.name = "oracle"

    return gate

def diffuser(n):
    qc = QuantumCircuit(n)

    # Apply transformation |s> -> |00..0> (H-gates)
    qc.h(range(n))

    # Apply transformation |00..0> -> |11..1> (X-gates)
    qc.x(range(n))

    # Do multi-controlled-Z gate
    qc.h(n-1)
    qc.mct(list(range(n-1)), n-1)  # multi-controlled-toffoli
    qc.h(n-1)

    # Apply transformation |11..1> -> |00..0>
    for qubit in range(n):
        qc.x(qubit)

    # Apply transformation |00..0> -> |s>
    qc.h(range(n))

    # We will return the diffuser as a gate
    gate = qc.to_gate()
    gate.name = "diffuser"

    return gate

def grover(n):
    # The circuit is a Grover's search for the all-ones state.
    var = QuantumRegister(n, 'var')
    out = QuantumRegister(1, 'out')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(var, out, cr)

    # Initialize the output qubit to a phase-flip.
    qc.x(n)
    qc.h(n)

    # Apply the Hadamard gate to every qubit.
    qc.h(var)
    qc.barrier()

    # Apply the oracle to every qubit.
    qc.append(oracle(n), range(n+1))
    qc.barrier()

    # Apply the diffuser to every qubit.
    qc.append(diffuser(n), range(n))
    qc.barrier()

    # Undo the output qubit phase-flip.
    qc.h(n)
    qc.x(n)

    qc.measure(var, cr)

    return qc

'''def execute(qc):
    backend = Aer.get_backend('aer_simulator')
    job = qiskit.execute(qc, backend)
    result = job.result()
    return result'''

qc = grover(24)

from qiskit.providers.aer import AerSimulator
from qiskit import transpile
from qiskit import execute

sim = AerSimulator(method='statevector')
circ = transpile(qc)
circ.measure_all()
result = execute(circ, sim, shots=1, blocking_enable=True, blocking_qubits=23).result()

#print(qc.draw())
#result = execute(qc)
#print(result.get_counts())

print(result)
print('----------------------------------------------------- \n')