# https://github.com/zademi/QI_project_ichverzweifel/blob/2066ac99d208b29dbc6384f3dd6e8ea31dde5381/Error%20Detection%20Take%202.2.py
from qiskit import IBMQ, QuantumCircuit, execute,  Aer, assemble, transpile, QuantumRegister, ClassicalRegister
from qiskit.result import marginal_counts
from qiskit.providers.ibmq.job import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_vector
import matplotlib.pyplot as plt

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
s_provider = 'ibmq_bogota'
backend = provider.get_backend(s_provider)

# Initialization
n_rounds = 2
plural = False
if n_rounds > 1:
    plural = True
qr = QuantumRegister(5)
cr_data = ClassicalRegister(3)
cr_ancilla_first_round = ClassicalRegister(2)
cr_ancilla_second_round = ClassicalRegister(2)
qc_simp = QuantumCircuit(qr, cr_data, cr_ancilla_first_round, cr_ancilla_second_round)
qc_simp.h(2)
qc_simp.cx([2, 2, 1, 3, 0, 4], [1, 3, 0, 4, 1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Rounds of error detection
# n = 2
# x = 0
# qc_simp.x(2)
# while n <= n_rounds:
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_first_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])
    # x += 2
    # n += 1
qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
qc_simp.measure([1, 3], cr_ancilla_second_round)
qc_simp.reset([1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Measurement of the Data Qubits
qc_simp.measure([0, 2, 4], cr_data)
qc_simp.draw('mpl')
plt.title('The circuit for {} {round} of error correction'.format(n_rounds, round='rounds' if plural else 'round'))
plt.show()

# Job Executing
simp_job = execute(qc_simp, backend=backend, shots=50, memory=True)
print(simp_job.job_id())
job_monitor(simp_job)

# Histogram GHZ state
# counts = marginal_counts(simp_job.result(), indices=cr_data[0, 3]).get_counts()
# print("Results: ", counts)
# plot_histogram(counts, title='GHZ state after {} {round} of error correction'.format(n_rounds, round='rounds' if plural else 'round'))
# plt.show()

# GHZ results
data_memory = simp_job.result().get_memory()
print("GHZ memory: ", data_memory)
#
# # Ancilla first round results
# ancilla_first_round_memory = marginal_counts(simp_job.result(), indices=cr_ancilla_first_round[0, 2]).get_memory()
# print("Ancilla first round memory: ", ancilla_first_round_memory)
#
# # Ancilla second round results
# ancilla_second_round_memory = marginal_counts(simp_job.result(), indices=cr_ancilla_second_round[0, 2]).get_memory()
# print("Ancilla second round memory: ", ancilla_second_round_memory)

ghz_counts = []
for item in data_memory:
    new_item = item[6:9]
    ghz_counts.append(new_item)

print("ghz_counts: ", ghz_counts)

#plot histogram GHZ
plt.title('GHZ state', fontsize=20)
plt.ylabel('probabilities', fontsize=15)
plt.hist(ghz_counts, bins=8, rwidth=0.7, density=True, color = 'blue', edgecolor = 'black')
plt.show()