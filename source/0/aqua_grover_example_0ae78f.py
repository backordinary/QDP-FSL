# https://github.com/dlyongemallo/quantum-computation/blob/95a739b94659e6db97d6f9457c0068f645d8eeed/qiskit/aqua-grover-example.py
#!/usr/bin/env python3

"""Grover example from https://github.com/Qiskit/qiskit-aqua/blob/master/README.md
"""

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
result = algorithm.run(backend)
print(result["result"])
