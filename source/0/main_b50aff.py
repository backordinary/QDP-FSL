# https://github.com/zademi/QI_project_ichverzweifel/blob/2066ac99d208b29dbc6384f3dd6e8ea31dde5381/main.py
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

aer_sim = Aer.get_backend('aer_simulator')


def get_noise(p_meas, p_gate):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates

    return noise_model


noise_model = get_noise(0.1,0.1)

qc0 = QuantumCircuit(3) # initialize circuit with three qubits in the 0 state
qc0.measure_all() # measure the qubits

cq = QuantumRegister(3, 'code_qubit')
lq = QuantumRegister(2, 'auxiliary_qubit')
sb = ClassicalRegister(1, 'syndrome_bit')
qc = QuantumCircuit(cq, lq, sb)
qc.cx(cq[0], cq[1])
qc.cx(cq[1], cq[2])
qc.h(lq[0])
qc.h(lq[1])
qc.measure(lq[0], sb)
qc.draw(output='mpl')
plt.show()

# run the circuit with the noise model and extract the counts
qobj = assemble(qc)
counts = aer_sim.run(qobj, noise_model=noise_model).result().get_counts()

plot_histogram(counts)
plt.show()