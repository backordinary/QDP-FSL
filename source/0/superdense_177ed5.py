# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/qiskit/superdense.py
from qiskit import *
from qiskit.visualization import *

#%%
def create_bell_pair(qc, a, b):
    qc.h(a)
    qc.cx(a,b)
    return qc

#%%
def encode_message(qc, qubit, msg):
    if msg == "00":
        pass
    elif msg == "10":
        qc.x(qubit)
    elif msg == "01":
        qc.z(qubit)
    elif msg == "11":
        qc.z(qubit)
        qc.x(qubit)
    else:
        print("Invalid message: Sending '00'")

#%%
def decode_message(qc, a, b):
    qc.cx(a, b)
    qc.h(a)
    return qc

#%%
qc = QuantumCircuit(2)

create_bell_pair(qc, 0, 1)
qc.barrier()
message = "11"
encode_message(qc, 0, message)
qc.barrier()

decode_message(qc, 0, 1)
qc.measure_all()

qc.draw()

#%%
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()

counts = result.get_counts()
print(counts)
plot_histogram(counts)

#%%
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

shots = 256

#%%
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x:
    x.configuration().n_qubits >= 2 and not x.configuration().simulator and
    x.status().operational==True))
print("Least busy backend: ", backend)

#%%
job = execute(qc, backend=backend, shots=shots)
job_monitor(job)

#%%
result = job.result()
plot_histogram(result.get_counts(qc))
