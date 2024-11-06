# https://github.com/wsdt/QuantumComputing_Python/blob/e52b0c833dc146b855b2962b84cf3f22832b4c4e/conf/controllers/c_qiskit_conf.py
from conf import USE_REAL_QCOMP, QISKIT_IBM_TOKEN


def get_quantum_machine():
    if USE_REAL_QCOMP:
        from qiskit import IBMQ
        IBMQ.enable_account(QISKIT_IBM_TOKEN)
        qm = IBMQ.backend(name="ibmqx5")[0]
    else:
        from qiskit import BasicAer
        qm = BasicAer.get_backend("qasm_simulator")
    return qm
