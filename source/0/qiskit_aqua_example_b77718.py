# https://github.com/HanlinMiao/CPE_322/blob/632ef34be0d6b87758ff68eeace940427e4a5944/lesson9/qiskit_aqua_example.py
from qiskit import Aer
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.aqua.algorithms import Grover

sat_cnf = """
c Example DIMACS 3-sat
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""

backend = Aer.get_backend('qasm_simulator')
oracle = LogicalExpressionOracle(sat_cnf)
algorithm = Grover(oracle)
assignment = algorithm.run(backend)

print(assignment["assignment"])
