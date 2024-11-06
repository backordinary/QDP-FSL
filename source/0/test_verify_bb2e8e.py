# https://github.com/cda-tum/qcec/blob/6d7de6920993cf0b3845763c17fd210802e2fdc2/test/python/test_verify.py
"""Test the verification of two circuits."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from mqt import qcec


@pytest.fixture()
def original_circuit() -> QuantumCircuit:
    """Fixture for a simple circuit."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure_all()
    return qc


@pytest.fixture()
def alternative_circuit() -> QuantumCircuit:
    """Fixture for an alternative version of the simple circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.swap(0, 1)
    qc.cx(1, 2)
    qc.measure(0, 1)
    qc.measure(1, 0)
    qc.measure(2, 2)
    return qc


def test_verify(original_circuit: QuantumCircuit, alternative_circuit: QuantumCircuit) -> None:
    """Test the verification of two equivalent circuits."""
    result = qcec.verify(original_circuit, alternative_circuit)
    assert result.equivalence == qcec.EquivalenceCriterion.equivalent


def test_verify_kwargs(original_circuit: QuantumCircuit, alternative_circuit: QuantumCircuit) -> None:
    """Test the verification of two equivalent circuits with some keyword arguments (one of each category)."""
    result = qcec.verify(
        original_circuit,
        alternative_circuit,
        alternating_scheme="one_to_one",
        timeout=3600,
        trace_threshold=1e-6,
        transform_dynamic_circuit=True,
        additional_instantiations=2,
        seed=42,
    )
    assert result.equivalence == qcec.EquivalenceCriterion.equivalent


def test_verify_config(original_circuit: QuantumCircuit, alternative_circuit: QuantumCircuit) -> None:
    """Test the verification of two equivalent circuits with a configuration object."""
    config = qcec.Configuration()
    config.execution.timeout = 3600
    result = qcec.verify(original_circuit, alternative_circuit, config)
    assert result.equivalence == qcec.EquivalenceCriterion.equivalent


def test_compiled_circuit_without_measurements() -> None:
    """Regression test for https://github.com/cda-tum/qcec/issues/236.

    It makes sure that circuits compiled without measurements are handled correctly.
    """
    from qiskit import transpile
    from qiskit.providers.fake_provider import FakeAthens

    qc = QuantumCircuit(1)
    qc.x(0)
    qc_compiled = transpile(qc, backend=FakeAthens())

    result = qcec.verify(qc, qc_compiled)
    assert result.equivalence == qcec.EquivalenceCriterion.equivalent