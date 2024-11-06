# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Gidney/grover2.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, compile  # available_backends
from qiskit.tools.visualization import plot_histogram  # plot_state,
from qiskit import Aer, IBMQ
from qiskit.backends.jobstatus import JOB_FINAL_STATES
from qiskit.tools.visualization import circuit_drawer  # plot_state_qsphere
import time

# load acccount
IBMQ.load_accounts()
print(IBMQ.backends())


def run(qc, be, sim=True):
    """run

        :param be: Quantum Processor Chosen to use available
        """
    if sim:
        """run_sim"""
        # See a list of available local simulators
        # print("Local backends: ", Aer.available_backends())

        # compile and run the Quantum circuit on a simulator backend
        backend_sim = Aer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend_sim)
        result_sim = job_sim.result()
        # Show the results
        print("simulation: ", result_sim)
        counts = result_sim.get_counts(qc)
        plot_histogram(counts)
        return counts
    else:
        backend = IBMQ.get_backend(name=be)
        qobj = compile(qc, backend, shots=2000)
        job = backend.run(qobj)
        start_time = time.time()
        job_status = job.status()
        while job_status not in JOB_FINAL_STATES:
            print(
                f'Status @ {time.time()-start_time:0.0f} s: {job_status.name},'
                f' est. queue position: {job.queue_position()}')
            time.sleep(10)
            job_status = job.status()

        result = job.result()
        counts = result.get_counts()
        # get_data() contains time to run the application
        print(result.get_data())
        print(counts)
        plot_histogram(counts)
        return counts


# index to search for
index = '10'

# initialize quantum and classical registers
t = QuantumRegister(1, 'tar')
q = QuantumRegister(2, 'q')
c = ClassicalRegister(2, 'c')
qc = QuantumCircuit(q, c, t)

# qubit initialization
qc.x(t)
qc.barrier()
qc.h(q)
qc.h(t)
qc.barrier()

# oracle
for inx, i in enumerate(index):
    i = int(i)
    if i == 1:
        qc.x(q[inx])

qc.barrier()

qc.ccx(q[0], q[1], t[0])

qc.barrier()

for inx, i in enumerate(index):
    i = int(i)
    if i == 1:
        qc.x(q[inx])

qc.barrier()

# diffusion gate
qc.h(q)
qc.barrier()
qc.x(q)
qc.barrier()
qc.h(q[1])
qc.barrier()
qc.cx(q[0], q[1])
qc.barrier()
qc.h(q[1])
qc.barrier()
qc.x(q)
qc.barrier()
qc.h(q)
qc.barrier()

qc.measure(q, c)

circuit_drawer(qc, filename='grover2.png')

run(qc, '')
# run(qc, 'ibmq_16_melbourne', sim=False)
# run(qc, 'ibmqx4', sim=False)
