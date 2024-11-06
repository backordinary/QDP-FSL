# https://github.com/phikubo/quantum_genetic_scheduler/blob/fb06adef0441c5b9e242e97a5a4dadda707e2389/depleted/quantum_kp.py
#import
import os
if __name__=="__main__":
	#Prototipo:
	pass
else:
	print("Modulo Importado: [", os.path.basename(__file__), "]")

#-----------------------------------------------------------------
from qiskit_optimization.applications import Knapsack             # clase principal para este tipo de problema
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit import Aer, BasicAer

print('Modulos importados')

#---------------------------------------------------------------
# Testeo del problema Knapsack

values  = [6, 10, 12, 5, 8]   # lista de los valores de beneficio de los artículos
weights = [5,  2,  1, 1, 4]   # lista de pesos de artículos
max_weight = 7                # capacidad máxima de peso (capacidad de la mochila)

# definir el problema que queremos resolver
# Usamos la clase Knapsack que es parte del módulo qiskit-optimization
problem = Knapsack(values = values, weights = weights, max_weight = max_weight)

print('problem: ',problem)

qp = problem.to_quadratic_program()

# mostremos los detalles del programa cuadrático
print(qp)

#---------------------------------------------------------------
#Tipos de solucion para qp

# Numpy Eigensolver

meo = MinimumEigenOptimizer(min_eigen_solver=NumPyMinimumEigensolver())
result = meo.solve(qp)
print('resultado:\n', result)
print('\nsolución:\n', problem.interpret(result))

# QAOA

seed = 123
algorithm_globals.random_seed = seed
qinstance = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=1000, 
                            seed_simulator=seed, seed_transpiler=seed)

meo = MinimumEigenOptimizer(min_eigen_solver=QAOA(reps=1, quantum_instance=qinstance))
result = meo.solve(qp)
print('resultado:\n', result)
print('\nsolución:\n', problem.interpret(result))
print('\ntiempo de ejecución:', result.min_eigen_solver_result.optimizer_time)
