# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/optimization-experiments/grover_optimizer.py
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit import BasicAer
from docplex.mp.model import Model

backend = BasicAer.get_backend('statevector_simulator')
model = Model()
x0 = model.binary_var(name='x0')
x1 = model.binary_var(name='x1')
x2 = model.binary_var(name='x2')
model.minimize(-x0+2*x1-3*x2-2*x0*x2-1*x1*x2)
qp = from_docplex_mp(model)
print(qp.export_as_lp_string())

grover_optimizer = GroverOptimizer(6, num_iterations=10, quantum_instance=backend)
results = grover_optimizer.solve(qp)
print("x={}".format(results.x))
print("fval={}".format(results.fval))

exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
exact_result = exact_solver.solve(qp)
print("x={}".format(exact_result.x))
print("fval={}".format(exact_result.fval))
