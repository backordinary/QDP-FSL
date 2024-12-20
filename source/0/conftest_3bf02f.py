# https://github.com/jakelishman/qiskit-qasm2/blob/e64bc5f07783c1e1663efdaadb28f53398fce993/tests/conftest.py
import contextlib
import uuid

from typing import Iterable

import pytest
from qiskit.circuit import QuantumCircuit, Parameter

import qiskit_qasm2.parse


def gate_builder(name: str, parameters: Iterable[Parameter], definition: QuantumCircuit):
    """Get a builder for a custom gate.  Ideally we would just use an eagerly defined `Gate`
    instance here, but limitations in how `QuantumCircuit.__eq__` and `Instruction.__eq__` work mean
    that we have to ensure we're using the same class as the parser for equality checks to work."""
    # Ideally we wouldn't have this at all, but hiding it away in one function is likely the safest
    # and easiest to update if the Python component of the library changes.
    # pylint: disable=protected-access
    def definer(*arguments):
        # We can supply empty lists for the gates and the bytecode, because we're going to override
        # the definition manually ourselves.
        gate = qiskit_qasm2.parse._DefinedGate(name, definition.num_qubits, arguments, (), ())
        gate._definition = definition.assign_parameters(dict(zip(parameters, arguments)))
        return gate

    return definer


class _TemporaryFilePathFactory:
    def __init__(self, basedir):
        self._basedir = basedir

    @contextlib.contextmanager
    def __call__(self):
        path = self._basedir / str(uuid.uuid4())
        path.touch()
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def tmp_file_path_factory(tmp_path_factory):
    """Get a path to a unique read/writeable temporary file that already exists on disk (with no
    content), has a valid file name, and can be opened for both reading and writing in any mode. The
    file will cleaned up after the function, if it still exists."""
    return _TemporaryFilePathFactory(tmp_path_factory.getbasetemp())
