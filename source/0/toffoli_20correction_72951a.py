# https://github.com/zademi/QI_project_ichverzweifel/blob/2066ac99d208b29dbc6384f3dd6e8ea31dde5381/Toffoli%20Correction.py
from qiskit import IBMQ, QuantumCircuit, execute,  Aer, assemble, transpile, QuantumRegister, ClassicalRegister
from qiskit.result import marginal_counts
from qiskit.providers.ibmq.job import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_vector
import matplotlib.pyplot as plt
from qiskit.quantum_info.states.measures import state_fidelity
from math import sqrt
from qiskit.quantum_info.analysis import hellinger_fidelity

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
s_provider = 'ibmq_bogota'
backend = provider.get_backend(s_provider)

# Initialization
n_rounds = 1
n_resets = 3
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
#qc_simp.x(2)
while n <= n_rounds:
    qc_simp.toffoli(1, 3, 2)
    qc_simp.x(3)
    qc_simp.toffoli(1, 3, 0)
    qc_simp.x(3)
    qc_simp.x(1)
    qc_simp.toffoli(1, 3, 4)
    qc_simp.x(1)
    qc_simp.barrier([0, 1, 2, 3, 4])
    x += 2
    n += 1

# Measurement of the Data Qubits
qc_simp.measure([0, 2, 4], [x, x + 1, x + 2])
qc_simp.draw('mpl')
plt.title('The circuit for {} rounds of repetition code'.format(n_rounds))
plt.show()

# Job Executing
simp_job = execute(qc_simp, backend=backend, shots=1024)
print(simp_job.job_id())
job_monitor(simp_job)

# Results
counts = marginal_counts(simp_job.result(), indices=[x, x + 1, x + 2]).get_counts()
print("Results: ", counts)
plot_histogram(counts, title='GHZ state after {} rounds'.format(n_rounds))
plt.show()

# Fidelity Check
perfect_counts = {'000': 512, '111': 512}
fidelity = hellinger_fidelity(counts, perfect_counts)
print(fidelity)


