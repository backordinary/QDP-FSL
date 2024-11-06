# https://github.com/SamirFarhat17/quantum-computer-programming-ibm/blob/eeb446026f480cdb48e4dc9c6d23b825300493c9/optimization-experiments/min_eigen_optimizer.py
from qiskit import BasicAer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer, SolutionSample, OptimizationResultStatus
from qiskit_optimization import QuadraticProgram
from qiskit.visualization import plot_histogram
from typing import List, Tuple
import numpy as np


# Converting QUBO to an operator
qubo = QuadraticProgram()
qubo.binary_var('x')
qubo.binary_var('y')
qubo.binary_var('z')
qubo.minimize(linear=[1,-2,3], quadratic={('x', 'y'): 1, ('x', 'z'): -1, ('y', 'z'): 2})
#print(qubo.export_as_lp_string())

op, offset = qubo.to_ising()
#print('offset: {}'.format(offset))
#print('operator:')
#print(op)

qp=QuadraticProgram()
qp.from_ising(op, offset, linear=True)
#print(qp.export_as_lp_string())

# Solving QUBO with MinEiggenOptimizer
algorithm_globals.random_seed = 10598
quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                   seed_simulator=algorithm_globals.random_seed,
                                   seed_transpiler=algorithm_globals.random_seed)
qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=[0., 0.])
exact_mes = NumPyMinimumEigensolver()

qaoa = MinimumEigenOptimizer(qaoa_mes)   # using QAOA
exact = MinimumEigenOptimizer(exact_mes)  # using the exact classical numpy minimum eigen solver
exact_result = exact.solve(qubo)
# print(exact_result)

qaoa_result = qaoa.solve(qubo)
#print(qaoa_result)

print('variable order:', [var.name for var in qaoa_result.variables])
for s in qaoa_result.samples:
    print(s)


def get_filtered_samples(samples: List[SolutionSample],
                         threshold: float = 0,
                         allowed_status: Tuple[OptimizationResultStatus] = (OptimizationResultStatus.SUCCESS,)):
    res = []
    for s in samples:
        if s.status in allowed_status and s.probability > threshold:
            res.append(s)

    return res
filtered_samples = get_filtered_samples(qaoa_result.samples,
                                        threshold=0.005,
                                        allowed_status=(OptimizationResultStatus.SUCCESS,))
for s in filtered_samples:
    print(s)

fvals = [s.fval for s in qaoa_result.samples]
probabilities = [s.probability for s in qaoa_result.samples]
np.mean(fvals)

np.std(fvals)

samples_for_plot = {' '.join(f'{qaoa_result.variables[i].name}={int(v)}'
                             for i, v in enumerate(s.x)): s.probability
                    for s in filtered_samples}
samples_for_plot

p = plot_histogram(samples_for_plot)
p.savefig('histogram.png')

# Recursive Min Eigen Optimizer
rqaoa = RecursiveMinimumEigenOptimizer(qaoa, min_num_vars=1, min_num_vars_optimizer=exact)
rqaoa_result = rqaoa.solve(qubo)
print(rqaoa_result)

iltered_samples = get_filtered_samples(rqaoa_result.samples,
                                        threshold=0.005,
                                        allowed_status=(OptimizationResultStatus.SUCCESS,))
samples_for_plot = {' '.join(f'{rqaoa_result.variables[i].name}={int(v)}'
                             for i, v in enumerate(s.x)): s.probability
                    for s in filtered_samples}
print(samples_for_plot)

po = plot_histogram(samples_for_plot)
po.savefig('rec_histogram')
