# https://github.com/jvscursulim/hybrid_quantum_classical_neural_network/blob/10aab36e20dedd1dbe655672cf6a77ce4209e7d3/tools/tools.py
import numpy as np
from neural_network import neural_net
from hamiltonian import hamiltonian_qubit_op
from qiskit import Aer, execute
from qiskit_nature.algorithms import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.results.eigenstate_result import EigenstateResult


def exact_diagonalizer(problem: ElectronicStructureProblem, converter: QubitConverter) -> EigenstateResult:
    """Peforms an exact diagonalization in the Hamiltonian of the problem.

    Args:
        problem (ElectronicStructureProblem): Physical system of interest.
        converter (QubitConverter): A QubitConverter object.

    Returns:
        EigenstateResult: The result of the exact diagonalization.
    """
    
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver = solver)
    result = calc.solve(problem = problem)
    
    return result


def cost_function(hamiltonian, PQC):
    
    qc = PQC
    backend = Aer.get_backend("statevector_simulator")
    state = execute(experiments = qc, backend = backend).result.get_statevector()
    
    qubit_mat = hamiltonian.to_matrix()
    
    expectation_value = np.matmul(np.conjugate(state.T), np.matmul(qubit_mat, state))
    
    return expectation_value


# def train_function(paramters):
    
#     bond_length = training_dist
#     energy = np.zeros(len(bond_length))
#     for i in range(len(bond_length)):
        
#         qc = neural_net(num_qubits = 2, bond_length = bond_length[i], parameters = paramters, backend = Aer.get_backend("qasm_simulator"))
#         qubit_op = hamiltonian_qubit_op(dist = bond_length[i])
#         energy[i] = cost_function(qubit_op, qc) + repulsion_energy_list[training_index[i]]
#     energy_net = sum(energy)/len(bond_length)
    
#     return energy_net