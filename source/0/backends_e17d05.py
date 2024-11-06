# https://github.com/parasol4791/quantumComp/blob/0a9a56334d10280e86376df0d57fdf8f4051093d/utils/backends.py
# Utilities to create and run backends

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import execute, transpile
from qiskit.tools.monitor import job_monitor

def get_backend_aer():
    return Aer.get_backend('aer_simulator')

def get_job_aer(qc, shots=1024, memory=False):
    backend = get_backend_aer()
    return execute(qc, backend, shots=shots, memory=memory)

# Actual quantum hardware
def get_backend_ibmq(qubits=5):
    IBMQ.enable_account('02c89fe68aed70372ad187e85499d45bd6b22e1ed35c3cc50af084e4ed67ee6735756fa100928cf08840efb8ab5d95a12253ee1fd3a6bebce0e00c4b4d276af1')
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= qubits and
                                       not x.configuration().simulator and x.status().operational==True))
    print("least busy backend: ", backend)
    return backend

def get_job_ibmq(qc, qubits=5, shots=1024):
    backend = get_backend_ibmq(qubits)
    # transpiled_qc = transpile(qc, backend, optimization_level=3)
    # job = backend.run(transpiled_qc)
    job = execute(qc, backend, shots=shots)
    job_monitor(job, interval=2)
    return job