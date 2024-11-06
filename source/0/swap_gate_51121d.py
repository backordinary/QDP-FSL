# https://github.com/Schwarf/qiskit_fundamentals/blob/ba66945b4b57055cbd207e32c4d7105b8b1b01f3/multiple_qubits_gates/two_qubit_gates/swap_gate.py
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.visualization import plot_histogram
from typing import List, Tuple

control_qubit = 0
target_qubit = 1
svsim = Aer.get_backend('aer_simulator')


def use_swap(initial_state: Tuple[List[int], int] = None, do_draw: bool = False) -> QuantumCircuit:
    qc_direct = QuantumCircuit(2)
    if initial_state:
        qc_direct.initialize(initial_state[0], initial_state[1])
    qc_direct.swap(control_qubit, target_qubit)
    qc_direct.save_statevector()
    qobj = assemble(qc_direct)
    result = svsim.run(qobj).result()
    if do_draw:
        qc_direct.draw()
        plot_histogram(result.get_counts())
    return result.get_counts()


def constructed_swap(initial_state: Tuple[List[int], int] = None, do_draw: bool = False) -> QuantumCircuit:
    qc_indirect = QuantumCircuit(2)
    if initial_state:
        qc_indirect.initialize(initial_state[0], initial_state[1])
    qc_indirect.cx(control_qubit, target_qubit)
    qc_indirect.cx(target_qubit, control_qubit)
    qc_indirect.cx(control_qubit, target_qubit)

    qc_indirect.save_statevector()
    qobj = assemble(qc_indirect)
    result = svsim.run(qobj).result()
    if do_draw:
        qc_indirect.draw()
        plot_histogram(result.get_counts())
    return result.get_counts()


initial_states = [([1, 0], target_qubit),
                  ([0, 1], target_qubit),
                  ([1, 0], control_qubit),
                  ([0, 1], control_qubit)]

for initial_state in initial_states:
    counts_direct = use_swap(initial_state)
    counts_indirect = constructed_swap(initial_state)
    assert (counts_direct == counts_indirect)
plt.show()