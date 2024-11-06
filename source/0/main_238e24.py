# https://github.com/rohannakra/Superposition-Experiment/blob/0e01bf0615e526891953f851fa618a098ed46213/main.py
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# qiskit imports:
from qiskit import QuantumCircuit, assemble, Aer, execute, __qiskit_version__
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector, plot_state_qsphere
from qiskit_textbook.tools import array_to_latex
from qiskit.providers.ibmq import least_busy
from qiskit.tools import job_monitor

# other imports:
from math import sqrt, pi
import numpy as np

qasm_simulator = Aer.get_backend('qasm_simulator')
sv_simulator = Aer.get_backend('statevector_simulator')
unitary_simulator = Aer.get_backend('unitary_simulator')

print(__qiskit_version__)

# %% [markdown]
# ##### show superposition on bloch sphere.

# %%
def superposition_sv(initial_states):

    qc = QuantumCircuit(len(initial_states))

    for index, state in enumerate(initial_states):
        qc.initialize(state, index)
        qc.h(index)

    qobj = assemble(qc)

    statevector = sv_simulator.run(qobj).result().get_statevector()

    return statevector

# %% [markdown]
# ##### run tests.

# %%
tests = [
    [1, 0],
    [0, 1],
    [sqrt(3/4), sqrt(1/4)],
    [sqrt(1/8), sqrt(7/8)]
]

plot_bloch_multivector(superposition_sv(tests))

# %% [markdown]
# ##### check measurement for all tests

# %%
def measure_sv(initial_states):

    qc = QuantumCircuit(len(initial_states), len(initial_states))

    for index, state in enumerate(initial_states):
        qc.initialize(state, index)
        qc.h(index)
        qc.measure(index, index)

    qobj = assemble(qc)

    statevector = sv_simulator.run(qobj).result().get_statevector()

    return statevector


# %%
plot_bloch_multivector(
    measure_sv(
        tests
    )
)


