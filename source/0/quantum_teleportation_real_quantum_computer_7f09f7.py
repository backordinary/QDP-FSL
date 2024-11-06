# https://github.com/raz-mon/Quantum-Computation-Project/blob/d75e533fb9fe1e246a8ff9613292084506ffc3b8/quantum_teleportation_real_quantum_computer.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, array_to_latex
from qiskit.extensions import Initialize
from qiskit.quantum_info import random_statevector


def create_bell_pair(qc, a, b):
    """Creates a bell pair in qc using qubits a & b (given that they are in 00 state)"""
    qc.h(a)         # Put qubit a into state |+>
    qc.cx(a, b)     # CNOT with a as control and b as target


def alice_gates(qc, psi, a):
    qc.cx(psi, a)
    qc.h(psi)


# Create random 1-qubit state
psi = random_statevector(2)
init_gate = Initialize(psi)
init_gate.label = "init"
inverse_init_gate = init_gate.gates_to_uncompute()

def new_bob_gates(qc, a, b, c):
    qc.cx(b, c)
    qc.cz(a, c)


qc = QuantumCircuit(3, 1)

# First, let's initialize Alice's q0
qc.append(init_gate, [0])
qc.barrier()

# Now begins the teleportation protocol
create_bell_pair(qc, 1, 2)
qc.barrier()
# Send q1 to Alice and q2 to Bob
alice_gates(qc, 0, 1)
qc.barrier()
# Alice sends classical bits to Bob
new_bob_gates(qc, 0, 1, 2)

# We undo the initialization process
qc.append(inverse_init_gate, [2])

# See the results, we only care about the state of qubit 2
qc.measure(2, 0)

# View the results:
qc.draw()

# First, see what devices we are allowed to use by loading our saved accounts
IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
provider = IBMQ.get_provider(hub='ibm-q-research-2', group='ben-gurion-uni-1', project='main')


# get the least-busy backend at IBM and run the quantum circuit there
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
#Over 3 qubits (what's necessary for the quantum teleportation protocol):
# backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 3 and
#                                   not b.configuration().simulator and b.status().operational==True))

# Over 7 qubits (what's necessary for the scrambling scheme):
backend = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= 7 and
                                   not b.configuration().simulator and b.status().operational == True))

t_qc = transpile(qc, backend, optimization_level=3)
job = backend.run(t_qc)
job_monitor(job)  # displays job status under cell


# Get the results and display them
exp_result = job.result()
exp_counts = exp_result.get_counts(qc)
print(exp_counts)
plot_histogram(exp_counts)

print(f"The experimental error rate : {exp_counts['1']*100/sum(exp_counts.values()):.3f}%")





















