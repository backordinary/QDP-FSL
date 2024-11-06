# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/.ipynb_checkpoints/grover_qiskit_nuevo-checkpoint.py
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.providers.aer import AerSimulator
from qiskit import transpile
from qiskit import execute

#import pylab
#import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.tools.visualization import plot_histogram
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.circuit.library.phase_oracle import PhaseOracle

expression = '(w ^ x) & ~(y ^ z) & (x & y & z)'
sim = AerSimulator(method='statevector')
oracle = PhaseOracle(expression)
problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)
grover = Grover(quantum_instance=QuantumInstance(sim, shots=1))
result = grover.amplify(problem)

result2 = execute(result, sim, shots=1, blocking_enable=True, blocking_qubits=2).result()
print(result2)
#display(plot_histogram(result.circuit_results[0]))