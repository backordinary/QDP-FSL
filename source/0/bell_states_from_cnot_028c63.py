# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/multiple_qubits_gates/bell_states_from_cnot.py
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, Aer, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere


def get_first_bell_state(first_qubit: int = 0, draw: bool = False) -> QuantumCircuit:
    quantum_circuit = QuantumCircuit(2)
    quantum_circuit.h(first_qubit)
    quantum_circuit.cx(0, 1)
    if draw:
        quantum_circuit.draw()
    return quantum_circuit


def get_second_bell_state(first_qubit: int = 0, draw: bool = False) -> QuantumCircuit:
    quantum_circuit = get_first_bell_state(first_qubit)
    quantum_circuit.z(first_qubit)
    if draw:
        quantum_circuit.draw()
    return quantum_circuit


def get_third_bell_state(first_qubit: int = 0, draw: bool = False) -> QuantumCircuit:
    quantum_circuit = get_first_bell_state(first_qubit)
    quantum_circuit.x(first_qubit)
    if draw:
        quantum_circuit.draw()
    return quantum_circuit


def get_fourth_bell_state(first_qubit: int = 0, draw: bool = False) -> QuantumCircuit:
    quantum_circuit = get_first_bell_state(first_qubit)
    quantum_circuit.z(first_qubit)
    quantum_circuit.x(first_qubit)
    if draw:
        quantum_circuit.draw()
    return quantum_circuit


def run_simulator(qc: QuantumCircuit) -> None:
    svsim = Aer.get_backend('aer_simulator')
    qc.save_statevector()
    qobj = assemble(qc)
    result = svsim.run(qobj).result()
    final_state = svsim.run(qobj).result().get_statevector()
    plot_state_qsphere(final_state)


qc_first = get_first_bell_state()
qc_second = get_second_bell_state()
qc_third = get_third_bell_state()
qc_fourth = get_fourth_bell_state()

run_simulator(qc_first)
run_simulator(qc_second)
run_simulator(qc_third)
run_simulator(qc_fourth)
plt.show()
