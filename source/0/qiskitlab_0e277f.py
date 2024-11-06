# https://github.com/GizmoBill/QuantumComputing/blob/c379a941f269271c8ce5a8b3d47fa6b85d2af41a/QiskitLab.py
# Copyright (c) 2021 Bill Silver. Licence granted to public under terms of MIT licence
# at https://github.com/GizmoBill/QuantumComputing/blob/main/LICENSE

from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit, execute, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy, IBMQFactory
from Grover import *

provider = IBMQFactory().load_account()
machine = "ibmq_santiago"

def stateVec(qc) :
    result = execute(qc, backend = Aer.get_backend('statevector_simulator')).result()
    print(result.get_statevector())

def simulate(qc, noise = False) :
    if noise :
        backend = QasmSimulator.from_backend(provider.get_backend(name = machine))
    else :
        backend = QasmSimulator()
    result = execute(qc, backend = backend, shots = 1000).result()
    print(result.get_counts())

def run(qc, leastBusy = False) :
    if leastBusy :
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= qc.num_qubits and 
                                               not x.configuration().simulator and x.status().operational==True))
    else :
        backend = provider.get_backend(name = machine)
    print("Backend: ", backend)
    job = execute(qc, backend = backend, shots = 1000, optimization_level = 3, qobj_id = qc.name)
    #job_monitor(job, interval = 2)
    return job
