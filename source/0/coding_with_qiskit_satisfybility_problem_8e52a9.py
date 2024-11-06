# https://github.com/arshpreetsingh/Qiskit-cert/blob/117a7ebcd8b757432a3ec61ef23bf3634e0c1d62/coding_with_qiskit_Satisfybility_problem.py
'''

* When you have to deal with Multiple Conditions like.
Search for a restaurent Which is
1. Not more than 5 miles away.
2. Which servers Thai Food.
3. Which is not Very Much Expensive.
4. Which you can reserve Tables in Advance.

Conditions could be any of those................
When we have Big-Data-Sets!
'''
from qiskit import BasicAer
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle

from qiskit.tools.visualization import plot_histogram

log_expr = '((Olivia & Abe) |(Jin & Amira)) & ~(Abe & Amira)'
algorithm = Grover(LogicalExpressionOracle(log_expr))

backend = BasicAer.get_backend('qasm_simulator')

result = algorithm.run(backend)
print(result)
print(result['measurement']) # {'1001': 247, '1101': 272, '1110': 245, '0110': 260}

plot_histogram(result['measurement'], title="Possible Party Combinations", bar_labels=True)