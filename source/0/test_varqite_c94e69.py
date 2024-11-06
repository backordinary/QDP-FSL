# https://github.com/MarcDrudis/Gibbs/blob/9e53567d9e1af386062a2de4f0e277f6f4480188/test_varqite.py
from __future__ import annotations
from qiskit.algorithms import TimeEvolutionProblem
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_sampler import GibbsStateSampler
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_builder import GibbsStateBuilder
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseSampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import QuantumCircuit

import numpy as np

from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import (
    VarQiteGibbsStateBuilder,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.primitives import Sampler
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import build_ansatz, build_init_ansatz_params_vals
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector


sampler = Sampler()
time = 1000

hamiltonian = SparsePauliOp.from_list([("XI", 1), ("ZZ", 1)])
# param_dict = {**self._ansatz_init_params_dict, **problem_hamiltonian_param_dict}
extended_hamiltonian = hamiltonian ^ ("I" * hamiltonian.num_qubits)

# ansatz = build_ansatz(2*hamiltonian.num_qubits,1)
# init_param_values = build_init_ansatz_params_vals(2*hamiltonian.num_qubits,1)

ansatz = EfficientSU2(extended_hamiltonian.num_qubits, reps=2)
init_param_values = build_init_ansatz_params_vals(2*hamiltonian.num_qubits,1)


param_dict = dict(zip(ansatz.parameters, init_param_values))


evolution_problem = TimeEvolutionProblem(
    hamiltonian=extended_hamiltonian, time=time#, param_value_map = param_dict
)

var_princip = ImaginaryMcLachlanPrinciple()

# no sampler given so matrix multiplication will be used
qite_algorithm = VarQITE(ansatz, var_princip, init_param_values, num_timesteps=None)

result = qite_algorithm.evolve(evolution_problem).evolved_state

print(result)

# gibbs_state_function = self._qite_algorithm.evolve(evolution_problem).evolved_state
# print(gibbs_state_function.data[0][0])
# aux_registers = set(range(int(self._ansatz.num_qubits / 2), int(self._ansatz.num_qubits)))

# GibbsStateSampler(
#     sampler=sampler,
#     gibbs_state_function=gibbs_state_function,
#     hamiltonian=problem_hamiltonian,
#     temperature=temperature,
#     ansatz=self._ansatz,
#     ansatz_params_dict=None,
#     aux_registers=aux_registers,
# )