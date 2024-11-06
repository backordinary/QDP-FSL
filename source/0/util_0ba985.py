# https://github.com/drobiu/quantum-project/blob/26a2a49cabb0868c9bf5ad68e6861062da7ad6ea/src/util/util.py
import os

from qiskit import execute, BasicAer
from quantuminspire.credentials import get_authentication
from quantuminspire.qiskit import QI


def run_qc(qc, with_QI=True, n_shots=1024, verbose=True):
    if with_QI:
        QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')

        authentication = get_authentication()
        QI.set_authentication(authentication, QI_URL)
        qi_backend = QI.get_backend('QX single-node simulator')

        qi_job = execute(qc, backend=qi_backend, shots=n_shots)
        qi_result = qi_job.result()
        histogram = qi_result.get_counts(qc)
        if verbose:
            print("\nResult from the remote Quantum Inspire backend:\n")
            print('State\tCounts')
            [print('{0}\t{1}'.format(state, counts)) for state, counts in histogram.items()]

        print("\nResult from the local Qiskit simulator backend:\n")
    backend = BasicAer.get_backend("qasm_simulator")
    job = execute(qc, backend=backend, shots=n_shots)
    result = job.result()
    if verbose:
        print(result.get_counts(qc))

    return result
