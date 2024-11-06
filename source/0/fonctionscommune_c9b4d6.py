# https://github.com/doij1301/BSQ110_Projet1/blob/841fee4fc689748fea728edcada2a9bdb1ac38ac/FonctionsCommune.py
from qiskit import execute
import numpy as np
from qiskit import IBMQ
from qiskit.tools import job_monitor
from qiskit.providers.ibmq import least_busy


def execution_ibmq(circuit):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    # provider = IBMQ.get_provider(hub = 'ibm-q-qida',group='is-quantum-hub',project='algolab')

    for backends in provider.backends(filters=lambda x: not x.configuration().simulator):
        print(backends)

    backend = least_busy(provider.backends(filters=lambda
        x: x.configuration().n_qubits < 10 and not x.configuration().simulator and x.status().operational == True))

    job = execute(circuit, backend, shots=1000)
    job_monitor(job)
    counts = job.result().get_counts()
    print(counts)

def decimal_in_bit_string(nb_decimal):
    if nb_decimal == 0: nb_qbits = 1
    else:
        nb_qbits = np.math.ceil(np.math.log2(nb_decimal))
    return f"{nb_decimal+1:0{nb_qbits}b}"