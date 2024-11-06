# https://github.com/smml1996/q_simulator/blob/c1fb1b689c39b8270aaee2bdd6a54a5894808cc4/get_all_probs.py
import sys
from qiskit import QuantumCircuit
from z3quantum_gate import *
from static_solver import StaticSolver
import warnings
from utils import build_state
from z3 import Real

# https://ericpony.github.io/z3py-tutorial/guide-examples.htm

warnings.filterwarnings('ignore', category=DeprecationWarning)  # DeprecationWarning: Back-references to from Bit
# instances to their containing Registers have been
# deprecated. Instead, inspect Registers to find
# their contained Bits.

input_file = sys.argv[1]  # path to OpenQASM file

# create quantum circuit from OpenQASM file
qc = QuantumCircuit.from_qasm_file(input_file)

StaticSolver.add_constants()
vars = []

# create necessary qubits
for instruction in qc.data:
    qubits = instruction.qubits
    for qubit in qubits:
        register_name = qubit.register.name
        qubit_index = qubit.index
        name = f"{register_name}_{qubit_index}"
        Z3QuantumGate.mapping[name] = Z3Qubit(name)
        if name not in vars:
            vars.append(name)

# apply gates
for instruction in qc.data:
    op = instruction.operation.name
    args = []
    qubits = instruction.qubits
    for qubit in qubits:
        register_name = qubit.register.name
        qubit_index = qubit.index
        name = f"{register_name}_{qubit_index}"
        args.append(name)

    Z3QuantumGate(op, args).execute()

# objective_function = StaticSolver.get_objective_function(Z3QuantumGate.mapping)
# print(objective_function)

# y = Real("y")
# StaticSolver.solver.add(y == objective_function)
vars.sort()
for i in range(2**len(vars)):
    state = build_state(vars, i)
    StaticSolver.get_state_probability(state, Z3QuantumGate.mapping)

# state = build_state(vars, 9)
# StaticSolver.get_state_probability(state, Z3QuantumGate.mapping, y)