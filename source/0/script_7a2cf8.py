# https://github.com/MarcDrudis/GibbsPreparation/blob/06f5db3fe065407dcaee7db44c64af9c2d8ce052/saved_simulations/turbo/testingansatz/script.py
import numpy as np
from gibbs.utils import create_hamiltonian_lattice, create_heisenberg, lattice_hamiltonian
from gibbs.preparation.varqite import efficientTwoLocalansatz
from qiskit.algorithms.time_evolvers.variational import ForwardEulerSolver
from qiskit.quantum_info import SparsePauliOp
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis
from gibbs.dataclass import GibbsResult
import sys
from qiskit import transpile
from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit.algorithms.time_evolvers import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import VarQITE
from gibbs.customRK import CustomRK
from gibbs.qfiwrapper import variationalprinciplestorage
from qiskit.circuit.library import ECRGate



save_path = ""

num_qubits = 4
learning_locality = 3

horiginal = lattice_hamiltonian(num_qubits,1/4,-1,one_local=["Z"],two_local=["XX","ZZ"])

if len(sys.argv) > 1:
    coeffs = sys.argv[1::2]
    terms = sys.argv[2::2]
    control_field = SparsePauliOp.from_list([(terms[i],float(coeffs[i])) for i in range(len(coeffs))])
    horiginal = (horiginal+control_field).simplify()

coriginal = KLocalPauliBasis(learning_locality,num_qubits).pauli_to_vector(horiginal)

ansatz_arguments = {"num_qubits":num_qubits,"depth":2,"entanglement":"reverse_linear","su2_gates":["rz"],"ent_gates":[ECRGate()]}
ansatz,x0 = efficientTwoLocalansatz(**ansatz_arguments)
ansatz = transpile(ansatz,basis_gates=["rx", "ry", "rz", "cp", "crx", "cry", "crz"])
beta= 1

problem = TimeEvolutionProblem(hamiltonian = horiginal^"I"*num_qubits, time = beta/2)
variational_principle = variationalprinciplestorage(ImaginaryMcLachlanPrinciple)(gradient = ReverseEstimatorGradient(), qgt = ReverseQGT() )


varqite_kwargs = {
"ode_solver" : ForwardEulerSolver,
"num_timesteps": 10
}

varqite = VarQITE(ansatz,x0, variational_principle=variational_principle, **varqite_kwargs)
result_varqite = varqite.evolve(problem)

gibbs_result = GibbsResult(ansatz_arguments=ansatz_arguments,
                        parameters=result_varqite.parameter_values,
                        coriginal=coriginal,
                        num_qubits=num_qubits,
                        klocality=learning_locality,
                        betas = [2 *t for t in result_varqite.times],
                        stored_qgts = variational_principle.stored_qgts,
                        stored_gradients = variational_principle.stored_gradients,
                        cfaulties=[],  
)
gibbs_result.save(save_path+f"num_qubits{num_qubits}_controlfield={sys.argv[1:]}")
