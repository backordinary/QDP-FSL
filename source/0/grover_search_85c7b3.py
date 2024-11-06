# https://github.com/Aarun2/Quantum_Repo/blob/03febf41fe708260c50e82b615e0bfcc9ed140fe/Qiskit_Tutorials/Grover_Search.py
# Satisfiability Problem

from qiskit import BasicAer
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.tools.visualization import plot_histogram

log_expr = '((Olivia & Abe) | (Jin & Amira)) & ~(Abe & Amira)'
algorithm = Grover(LogicalExpressionOracle(log_expr))

backend = BasicAer.get_backend('qasm_simulator')

result = algorithm.run(backend)

plot_histogram(result['measurement'], title='Possible Party Comb', bar_labels=True)

# takes O(n) time on a classical computer vs 
# takes O(sqrt(n)) on a quantum computer
