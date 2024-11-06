# https://github.com/qcware/qcware_transpile/blob/6822b68d02dac7ca32643e107a4f8eb974cc0b02/tests/translations/quasar/test_ugates_with_qiskit.py
import quasar  # type: ignore
import qiskit  # type
from hypothesis.strategies import lists, floats
from hypothesis import given
import numpy as np

# There have been a series of errors in translation with
# statevectors not matching if two u2 or u3 matrices
# are placed sequentially; here we intend to test the
# equivalence of the matrices prepared by quasar and qiskit


def u1matrix(lam):
    return quasar.Matrix.u1(lam)


def u2matrix(phi, lam):
    return quasar.Matrix.u2(phi, lam)


def u3matrix(theta, phi, lam):
    return quasar.Matrix.u3(theta, phi, lam)


def U1matrix(lam):
    return qiskit.circuit.library.U1Gate(lam).to_matrix()


def U2matrix(phi, lam):
    return qiskit.circuit.library.U2Gate(phi, lam).to_matrix()


def U3matrix(theta, phi, lam):
    return qiskit.circuit.library.U3Gate(theta, phi, lam).to_matrix()


angles = floats(min_value=0, max_value=2 * np.pi)


@given(angles)
def test_u1(lam):
    assert np.allclose(u1matrix(lam), U1matrix(lam))
    assert u1matrix(lam).dtype == U1matrix(lam).dtype


@given(angles, angles)
def test_u2(phi, lam):
    assert np.allclose(u2matrix(phi, lam), U2matrix(phi, lam))
    assert u2matrix(phi, lam).dtype == U2matrix(phi, lam).dtype


@given(angles, angles, angles)
def test_u3(theta, phi, lam):
    assert np.allclose(u3matrix(theta, phi, lam), U3matrix(theta, phi, lam))
    assert u3matrix(theta, phi, lam).dtype == U3matrix(theta, phi, lam).dtype
