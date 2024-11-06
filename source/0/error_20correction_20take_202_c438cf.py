# https://github.com/zademi/QI_project_ichverzweifel/blob/2066ac99d208b29dbc6384f3dd6e8ea31dde5381/Error%20Correction%20Take%202.py
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
cr = ClassicalRegister(2 * n_rounds + 3)
qc_simp = QuantumCircuit(qr, cr)
qc_simp.h(2)
qc_simp.cx([2, 2, 1, 3, 0, 4], [1, 3, 0, 4, 1, 3])
qc_simp.barrier([0, 1, 2, 3, 4])

# Rounds of error detection
n = 1
x = 0
# qc_simp.x(2)
while n <= n_rounds:
    qc_simp.cx([0, 4, 2, 2], [1, 3, 1, 3])
    qc_simp.measure([1, 3], [x, x + 1])
    qc_simp.reset([1, 3])
    qc_simp.barrier([0, 1, 2, 3, 4])
    x += 2
    n += 1

# Measurement of the Data Qubits
qc_simp.measure([0, 2, 4], [x, x + 1, x + 2])
qc_simp.draw('mpl')
plt.title('The circuit for {} {round} of error correction'.format(n_rounds, round='rounds' if plural else 'round'))
plt.show()

# Job Executing
simp_job = execute(qc_simp, backend=backend, shots=1, memory=True)
print(simp_job.job_id())
job_monitor(simp_job)

# Results
counts = marginal_counts(simp_job.result(), indices=[x, x + 1, x + 2]).get_counts()
print("Counts: ", counts)
plot_histogram(counts, title='GHZ state after {} {round} of error correction'.format(n_rounds, round='rounds' if plural else 'round'))
plt.show()

counts_memory = marginal_counts(simp_job.result(), indices=[x, x + 1, x + 2]).get_memory()
print("Counts_memory: ", counts_memory)

total = marginal_counts(simp_job.result(), indices=[0, 1, 2, 3, 4, 5, 6]).get_memory()
print("Total measurements: ", total)

j = 0
counts_memory_ancilla = []
counts_memory_ancilla_2 = []
while j < 2 * n_rounds:
    counts_memory_ancilla.append(marginal_counts(simp_job.result(), indices=[j, j + 1]).get_memory())
    print("Counts_memory ancilla {} round: ".format(int(j/2)+1), counts_memory_ancilla[int(j/2)])
    j += 2




