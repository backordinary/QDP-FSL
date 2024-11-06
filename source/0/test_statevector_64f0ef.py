# https://github.com/qcware/qiskit_qcware/blob/d89cb18f6c74b66058768fb7bc8ca5a122a3e8b1/tests/test_statevector.py
from qiskit_qcware import QcwareProvider
import qiskit
from qiskit.providers.aer import AerSimulator
import numpy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_statevector():
    qc: qiskit.QuantumCircuit = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    # addition of a measure gate "locks" the bit in qiskit-aer's
    # statevector simulator such that
    # the statevector measured after that has that bit "locked".
    # This is not how quasar works, so we will focus on gates without
    # measurements
    provider = QcwareProvider()
    sv1 = qiskit.execute(qc, backend=provider.get_backend(
        'local_statevector')).result().data()['statevector']
    aer_backend = AerSimulator(method="statevector")
    c = qc.copy()
    c.save_state("final_statevector")
    sv2 = qiskit.execute(c, aer_backend).result().data()['final_statevector']
    assert (numpy.allclose(sv1, sv2))
