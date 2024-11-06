# https://github.com/zademi/QI_project_ichverzweifel/blob/2066ac99d208b29dbc6384f3dd6e8ea31dde5381/Simulation.py
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, transpile, assemble
aer_sim = Aer.get_backend('aer_simulator')

from qiskit.result import marginal_counts
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt


def get_noise(p_meas, p_gate):
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model=NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")  # measurement error is applied to measurements
    #noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])  # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])  # two qubit gate error is applied to cx gates

    return noise_model
noise_model = get_noise(0,0.2)

# Initialization
n_rounds = 1
n_resets = 3
plural = False
if n_rounds > 1:
    plural = True
qr = QuantumRegister(5)
cr_qubit = ClassicalRegister(3 * n_rounds)
cr_ancilla = ClassicalRegister(2)
qc_simp = QuantumCircuit(qr, cr_qubit, cr_ancilla)
qc_simp.h(2)
qc_simp.cx([2, 2, 1, 3, 0, 4], [1, 3, 0, 4, 1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Rounds of error detection
n = 1
x = 0
#qc_simp.x(2)
while n <= n_rounds:
    qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
    qc_simp.measure([1, 3], cr_ancilla)
    qc_simp.x(2).c_if(cr_ancilla, 3)
    qc_simp.x(0).c_if(cr_ancilla, 1)
    qc_simp.x(4).c_if(cr_ancilla, 2)
    qc_simp.reset([1, 3])
    qc_simp.barrier([0, 1, 2, 3, 4])
    x += 2
    n += 1

# Measurement of the Data Qubits
qc_simp.measure([0, 2, 4], [x, x + 1, x + 2]) #we exit the loop, before exiting x is increased by 2
qc_simp.draw('mpl')
plt.title('The circuit for {} rounds of repetition code in a simulator'.format(n_rounds))
plt.show()

qobj = assemble(qc_simp)
counts = aer_sim.run(qobj, noise_model=noise_model).result().get_counts()
qubit_counts = marginal_counts(counts, indices=[x, x + 1, x + 2])
#ancilla_counts = marginal_counts(counts, indices=[x - 2, x - 1])

print(qubit_counts)
plot_histogram(qubit_counts)
plt.show()

# Fidelity Check
#perfect_counts = {'000': 512, '111': 512}
#fidelity = hellinger_fidelity(counts, perfect_counts)
#print(fidelity)
