# https://github.com/benmai22/Quantum-Computing-Project-Linear-Equation-Solver/blob/05c5cdeab5ae371824cccc8472081888ec0a5be4/QuantumLinearEquationSolver.py
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms import HHL, ExactLSsolver
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.qfts import Standard as StandardQFTS
from qiskit.aqua.components.iqfts import Standard as StandardIQFTS
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua.components.initial_states import Custom
import numpy as np
from qiskit.aqua.algorithms import NumPyLSsolver


# Input
matrix = [[1, -1/3], [-1/3, 1]]
vector = [1, 0]

# Liner Solution
linearAlogo = NumPyLSsolver(np.array(matrix),vector)
linearSol = NumPyLSsolver(np.array(matrix),vector)
print(linearSol)


def create_eigs(matrix, num_ancillae, negative_evals):
    ne_qfts = [None, None]
    if negative_evals:
        num_ancillae += 1
        ne_qfts = [StandardQFTS(num_ancillae - 1), StandardIQFTS(num_ancillae - 1)]

    return EigsQPE(MatrixOperator(matrix=matrix),
                   StandardIQFTS(num_ancillae),
                   num_time_slices=50,
                   num_ancillae=num_ancillae,
                   expansion_mode='suzuki',
                   expansion_order=2,
                   evo_time=None,
                   negative_evals=negative_evals,
                   ne_qfts=ne_qfts)

def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("fidelity %f" % fidelity)



orig_size = len(vector)
matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

# Initialize eigenvalue finding module
eigs = create_eigs(matrix, 3, False)
num_q, num_a = eigs.get_register_sizes()

# Initialize initial state module
init_state = Custom(num_q, state_vector=vector)

# Initialize reciprocal rotation module
reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
           init_state, reci, num_q, num_a, orig_size)
result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))

print(np.round(result['solution'], 5))


# IBMQ.save_account('2014e54721e8677688e0c1dffdc92dfe0f39dbae5d75dd5b14fd087c4f93349b6ea92cbbc872d9a838e946527e10144d554b04587662f5567309b75aca0085f0')
# IBMQ.load_account()
