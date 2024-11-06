# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/optimization-experiments/cplex-vs-gurobi.py
from qiskit_optimization.problems import QuadraticProgram

# define a problem
qp = QuadraticProgram()
qp.binary_var('x')
qp.integer_var(name='y', lowerbound=-1, upperbound=4)
qp.maximize(quadratic={('x', 'y'):1})
qp.linear_constraint({'x': 1, 'y': -1}, '<=', 0)
print(qp.export_as_lp_string())

from qiskit_optimization.algorithms import CplexOptimizer, GurobiOptimizer
print('cplex')
print(CplexOptimizer().solve(qp))
print()
print('gurobi')
print(GurobiOptimizer().solve(qp))

CplexOptimizer(disp=True, cplex_parameters={'threads': 1, 'timelimit': 0.1}).solve(qp)

from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

qins = QuantumInstance(backend=Aer.get_backend('aer_simulator'), shots=1000)
meo = MinimumEigenOptimizer(QAOA(COBYLA(maxiter=100), quantum_instance=qins))
result = meo.solve(qp)
print(result)
print('\ndisplay the best 5 solution samples')
for sample in result.samples[:5]:
    print(sample)

# docplex model
from docplex.mp.model import Model
docplex_model = Model('docplex')
x = docplex_model.binary_var('x')
y = docplex_model.integer_var(-1, 4, 'y')
docplex_model.maximize(x * y)
docplex_model.add_constraint(x <= y)
docplex_model.prettyprint()

# gurobi model
import gurobipy as gp
gurobipy_model = gp.Model('gurobi')
x = gurobipy_model.addVar(vtype=gp.GRB.BINARY, name="x")
y = gurobipy_model.addVar(vtype=gp.GRB.INTEGER, lb=-1, ub=4, name="y")
gurobipy_model.setObjective(x * y, gp.GRB.MAXIMIZE)
gurobipy_model.addConstr(x - y <= 0)
gurobipy_model.update()
gurobipy_model.display()

from qiskit_optimization.translators import from_docplex_mp, from_gurobipy
qp = from_docplex_mp(docplex_model)
print('QuadraticProgram obtained from docpblex')
print(qp.export_as_lp_string())
print('-------------')
print('QuadraticProgram obtained from gurobipy')
qp2 = from_gurobipy(gurobipy_model)
print(qp2.export_as_lp_string())

from qiskit_optimization.translators import to_gurobipy, to_docplex_mp
gmod = to_gurobipy(from_docplex_mp(docplex_model))
print('convert docplex to gurobipy via QuadraticProgram')
gmod.display()

dmod = to_docplex_mp(from_gurobipy(gurobipy_model))
print('\nconvert gurobipy to docplex via QuadraticProgram')
dmod.prettyprint()

ind_mod = Model('docplex')
x = ind_mod.binary_var('x')
y = ind_mod.integer_var(-1, 2, 'y')
z = ind_mod.integer_var(-1, 2, 'z')
ind_mod.maximize(3 * x + y - z)
ind_mod.add_indicator(x, y >= z, 1)
print(ind_mod.export_as_lp_string())

p = from_docplex_mp(ind_mod)
result = meo.solve(qp)  # apply QAOA to QuadraticProgram
print('QAOA')
print(result)
print('-----\nCPLEX')
print(ind_mod.solve())  # apply CPLEX directly to the Docplex model
