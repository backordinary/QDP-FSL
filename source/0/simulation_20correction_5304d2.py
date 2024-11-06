# https://github.com/zademi/QI_project_ichverzweifel/blob/2066ac99d208b29dbc6384f3dd6e8ea31dde5381/simulation%20correction.py
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, transpile, assemble, IBMQ
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.ignis.verification import marginal_counts

aer_sim = Aer.get_backend('aer_simulator')


def get_noise(p_meas, p_gate):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])  # single qubit gate error is applied to x gates

    return noise_model


noise_model = get_noise(0.1, 0.1)

# Initialization
n_rounds = 10
n_resets = 3
plural = False
if n_rounds > 1:
    plural = True
qr = QuantumRegister(5)
cr_data = ClassicalRegister(3)
cr_ancilla = ClassicalRegister(2)
qc_simp = QuantumCircuit(qr, cr_data, cr_ancilla)
qc_simp.h(2)
qc_simp.cx([2, 2, 1, 3, 0, 4], [1, 3, 0, 4, 1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Rounds of error detection
n = 1

#cr_ancilla is the classical register where ancilla outcomes are being saved
while n <= n_rounds:
    qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
    qc_simp.measure([1, 3], cr_ancilla)
    # qc_simp.x(0).c_if(cr_ancilla, 1)
    # qc_simp.x(4).c_if(cr_ancilla, 2)
    # qc_simp.x(2).c_if(cr_ancilla, 3)
    qc_simp.reset([1, 3])
    qc_simp.barrier([0, 1, 2, 3, 4])
    n += 1

qc_simp.measure([0, 2, 4], cr_data)
qc_simp.draw('mpl')
plt.title('The circuit for {} {round} of error detection on a simulator'.format(n_rounds, round='rounds' if plural else 'round'))
plt.show()

# run the circuit with the noise model and extract the counts
qobj = assemble(qc_simp)
counts = aer_sim.run(qobj, noise_model=noise_model).result().get_counts()
data_counts = [marginal_counts(counts, [0, 1, 2])]
ancilla_counts = [marginal_counts(counts, [3, 4])]

plot_histogram(data_counts, title='The GHZ state after {} {round} of error detection on a simulator'.format(n_rounds, round='rounds' if plural else 'round'))
plot_histogram(ancilla_counts, title='Ancilla measurements after {} {round} of error detection on a simulator'.format(n_rounds, round='rounds' if plural else 'round'))
plt.show()

