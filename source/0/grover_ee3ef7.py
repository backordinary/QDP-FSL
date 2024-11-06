# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/python/qiskit/grover.py
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor


#%% ####################
def initialize(qc, qbits):
    for q in qbits:
        qc.h(q)

    return qc

#%%
def oracle(qc):
    qc.cz(0,1)

    return qc

#%%
def diffuser(qc):
    qc.h(range(n))
    qc.z(range(n))
    qc.cz(0,1)
    qc.h(range(n))

    return qc

#%%
n = 2  # no. of qubits
grover_circuit = QuantumCircuit(n)
grover_circuit = initialize(grover_circuit, [0,1])
grover_circuit = oracle(grover_circuit)
grover_circuit = diffuser(grover_circuit)
grover_circuit.draw()

#%% Backend
sv_sim = Aer.get_backend('qasm_simulator')
grover_circuit.measure_all()
state = execute(grover_circuit, sv_sim).result().get_counts()
plot_histogram(state)
# np.round(state,3)

#%% IBMQ backend
provider = IBMQ.load_account()
device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                   not x.configuration().simulator and x.status().operational==True))
print("Running on current least busy device: ", device)

#%%
job = execute(grover_circuit, backend=device, shots=1024, optimization_level = 3)
job_monitor(job, interval=2)

#%%
results = job.result()
answer = result.get_counts(grover_circuit)
plot_histogram(answer)

##############################
# 3 qubit case: |101> & |110>
##############################

#%%
def oracle(n):
    qc = QuantumCircuit(n)

    qc.cz(0,1)
    qc.cz(1,2)

    oracle3 = qc.to_gate()
    oracle3.name = "$U_\omega$"


    return oracle3

#%%
def diffuser(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))

    qc.h(range(n-1))
    qc.mct(list(range(n-1)), n-1)
    qc.h(range(n-1))

    qc.x(range(n))
    qc.h(range(n))

    U_s = qc.to_gate()
    U_s.name = "$U_s$"

    return U_s


#%%
grover = QuantumCircuit(3)
grover = initialize(grover, [0,1,2])
grover.append(oracle(3), [0,1,2])
grover.append(diffuser(3), [0,1,2])
grover.measure_all()
grover.draw()

#%%
backend = Aer.get_backend('qasm_simulator')
counts = execute(grover, backend).result().get_counts()
plot_histogram(counts)
