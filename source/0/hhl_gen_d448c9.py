# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/TestScripts/hhl_gen.py
'''
This test script is used to generate HHL algorithm circuits
'''

from qiskit import Aer, transpile, assemble
from qiskit.circuit.library import QFT
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua.components.initial_states import Custom
import numpy as np

def create_eigs(matrix, num_auxiliary, num_time_slices, negative_evals):
    ne_qfts = [None, None]
    if negative_evals:
        num_auxiliary += 1
        ne_qfts = [QFT(num_auxiliary - 1), QFT(num_auxiliary - 1).inverse()]

    return EigsQPE(MatrixOperator(matrix=matrix),
                   QFT(num_auxiliary).inverse(),
                   num_time_slices=num_time_slices,
                   num_ancillae=num_auxiliary,
                   expansion_mode='suzuki',
                   expansion_order=2,
                   evo_time=None,  # This is t, can set to: np.pi*3/4
                   negative_evals=negative_evals,
                   ne_qfts=ne_qfts)

def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("Fidelity:\t\t %f" % fidelity)

def gen_hhl(N, matrix, vector):
    orig_size = len(vector)
    matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

    # Initialize eigenvalue finding module
    eigs = create_eigs(matrix, 3, 50, False)
    num_q, num_a = eigs.get_register_sizes()

    # Initialize initial state module
    init_state = Custom(num_q, state_vector=vector)

    # Initialize reciprocal rotation module
    reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

    algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
            init_state, reci, num_q, num_a, orig_size)

    circ = algo.construct_circuit(False)
    rcirc = transpile(circ, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=1)

    circ_ops = circ.count_ops()
    circ_op_count = 0
    for op in circ_ops:
        circ_op_count += circ_ops[op]

    rcirc_ops = rcirc.count_ops()
    rcirc_op_count = 0
    for op in rcirc_ops:
        rcirc_op_count += rcirc_ops[op]

    print('transpiled_hhl_'+str(N)+'x'+str(N)+'.qasm' + " op count before transpile: " + str(circ_op_count))
    print('transpiled_hhl_'+str(N)+'x'+str(N)+'.qasm' + " op count after transpile: " + str(rcirc_op_count))

    with open('transpiled_hhl_'+str(N)+'x'+str(N)+'.qasm', 'a') as f:
        f.write(rcirc.qasm())

gen_hhl(2,[[1, -1/3], [-1/3, 1]], [1, 0])
gen_hhl(3,[[1, 2, 1], [-3, -1, 2], [0,  5, 3]], [0, 1, -1])
gen_hhl(4,[[5, 6, 7, 1], [0, 2, 3, 4], [0, 0, 4, 5], [0, 0, 0, 3]], [10, 2, 2, 4])
