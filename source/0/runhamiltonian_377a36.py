# https://github.com/Tihulu/400-Project-Cagil-Benibol/blob/add2ed1b980913eff8ef74ce39177acca4312e91/runhamiltonian.py
from qiskit import Aer
from qiskit.opflow import X, Z, I, Y
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import PauliExpectation
'''
H2_op = (-8.881784197001252e-16  *  Z ^ Z ^ Z ^ Z ) + \
    ( -8  *  Z ^ Z ^ I ^ I )  + \
        ( 8.881784197001252e-16  *  Z  ^  I  ^  Z  ^  Z )  + \
            ( -8.881784197001252e-16  *  Z  ^  I  ^  Z  ^  I )  + \
                ( -7.9999999999999964  *  Z  ^  I  ^  I  ^  I )  + \
                   ( -6.661338147750939e-16  *  I  ^  Z  ^  Z  ^  Z )  + \
                    ( 2.220446049250313e-16  *  I  ^  Z  ^  Z  ^  I )  + \
                        ( 1.1102230246251565e-15  *  I  ^  Z  ^  I  ^  Z )  + \
                            ( -1.5543122344752192e-15  *  I  ^  Z  ^  I  ^  I )  + \
                                ( -7.9999999999999964  *  I  ^  I  ^  Z  ^  Z )  + \
                                    ( -7.9999999999999964  *  I  ^  I  ^  Z  ^  I )  + \
                                        ( -3.552713678800501e-15  *  I  ^  I  ^  I  ^  Z )  + \
                                            ( 48  *  I  ^  I  ^  I  ^  I )
'''                                     
H2_op = ( (-7.999999999999998+0j)  *  Z  ^  Z  ^  Z  ^  Z )  + \
( (7.999999999999995+0j)  *  Z  ^  Z  ^  Z  ^  I )  + \
( 0j  *  Z  ^  Z  ^  I  ^  X )  + \
( 0j  *  Z  ^  Z  ^  I  ^  Y )  + \
( (7.999999999999995+0j)  *  Z  ^  Z  ^  I  ^  Z )  + \
( (-7.999999999999998+0j)  *  Z  ^  Z  ^  I  ^  I )  + \
( (8.000000000000004+0j)  *  Z  ^  I  ^  Z  ^  Z )  + \
( (-8.000000000000004+0j)  *  Z  ^  I  ^  Z  ^  I )  + \
( (-8.000000000000004+0j)  *  Z  ^  I  ^  I  ^  Z )  + \
( (-56+0j)  *  Z  ^  I  ^  I  ^  I )  + \
( (7.999999999999992+0j)  *  I  ^  Z  ^  Z  ^  Z )  + \
( (-7.999999999999993+0j)  *  I  ^  Z  ^  Z  ^  I )  + \
( (-7.999999999999994+0j)  *  I  ^  Z  ^  I  ^  Z )  + \
( (-23.999999999999993+0j)  *  I  ^  Z  ^  I  ^  I )  + \
( (-8.000000000000004+0j)  *  I  ^  I  ^  Z  ^  Z )  + \
( (-8+0j)  *  I  ^  I  ^  Z  ^  I )  + \
( (3.552713678800501e-15+0j)  *  I  ^  I  ^  I  ^  Z )  + \
( (120+0j)  *  I  ^  I  ^  I  ^  I )
                    
seed = 50
algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_transpiler=seed, seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SPSA(maxiter=100)
vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi,
          expectation=PauliExpectation(group_paulis=False))
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
print(result)

'''
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
print(result)
optimizer_evals = result.optimizer_evals

initial_pt = result.optimal_point

algorithm_globals.random_seed = seed
qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = SLSQP(maxiter=1000)
vqe = VQE(ansatz, optimizer=slsqp, initial_point=initial_pt, quantum_instance=qi)
result1 = vqe.compute_minimum_eigenvalue(operator=H2_op)
print(result1)
optimizer_evals1 = result1.optimizer_evals
print()
print(f'optimizer_evals is {optimizer_evals1} with initial point versus {optimizer_evals} without it.')
'''