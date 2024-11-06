# https://github.com/Crabster/qiskit-learning/blob/3f14c39ee294f42e3f83a588910b659280556a68/run_circuit.py
from argparse import ArgumentParser
from qiskit import execute, BasicAer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy 
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt

from circuits.test import test_circuit
from circuits.quantum_teleportation import quantum_teleportation_example
from circuits.superdense_coding import superdense_coding_example
from circuits.deutsch_jozsa import deutsch_jozsa_example
from circuits.bernstein_vazirani import bernstein_vazirani_example
from circuits.quantum_fourier_transform import qft_example
from circuits.quantum_phase_estimation import qpe_example

parser = ArgumentParser(description='Run quantum circuit.')
parser.add_argument('-b', '--backend', metavar='BE', default='qasm_simulator',
                    choices=['auto', 'qasm_simulator', 'ibmqx2', 'ibmq_16_melbourne',
                             'ibmq_vigo', 'ibmq_ourense', 'ibmq_london',
                             'ibmq_burlington', 'ibmq_essex', 'ibmq_armonk', 'ibmq_rome'],
                    help='backend BE on which will the circuit run')
parser.add_argument('-c', '--circuit', metavar='QC', required=True,
                    choices=['test', 'tp', 'sc', 'dj', 'bv', 'qft', 'qpe'],
                    help='circuit QC to be run')
parser.add_argument('--shots', type=int, default=1024,
                    help='plot counts histogram of the result')
parser.add_argument('--plot', action="store_true",
                    help='plot counts histogram of the result')

args = parser.parse_args()

if args.circuit == 'test':
    qc = test_circuit()
if args.circuit == 'tp':
    qc = quantum_teleportation_example()
if args.circuit == 'sc':
    qc = superdense_coding_example()
if args.circuit == 'dj':
    qc = deutsch_jozsa_example()
if args.circuit == 'bv':
    qc = bernstein_vazirani_example()
if args.circuit == 'qft':
    qc = qft_example()
if args.circuit == 'qpe':
    qc = qpe_example()

if args.backend == 'qasm_simulator':
    backend = BasicAer.get_backend('qasm_simulator')
else:
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    if args.backend == 'auto':
        backends = provider.backends(filters=lambda b: 
                                                b.configuration().n_qubits >= qc.num_qubits and
                                                not b.configuration().simulator and
                                                b.status().operational==True)
        backend = least_busy(backends)
    else:
        backend = provider.get_backend(args.backend)

job = execute(qc, backend, shots=args.shots)
job_monitor(job)

counts = job.result().get_counts()
print(counts)
if args.plot:
    plot_histogram(counts)
    plt.show()


