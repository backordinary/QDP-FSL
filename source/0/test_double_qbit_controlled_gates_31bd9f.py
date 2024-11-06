# https://github.com/svenpruefer/quantumcomputing/blob/e082b6b829ccabdf1c9c64b5cc310ba8feaad2d5/qsoc/tests/unit_tests/basics/test_double_qbit_controlled_gates.py
# -*- coding: utf-8 -*-

# This code ist part of sqoc.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *


class TestDoubleQubitControlledGates:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 2000,
                'relative_error': 0.05}

    def test_bell_state(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |                ┌───┐┌─┐
        |qreg_0: |0>─────┤ X ├┤M├───
        |           ┌───┐└─┬─┘└╥┘┌─┐
        |qreg_1: |0>┤ H ├──■───╫─┤M├
        |           └───┘      ║ └╥┘
        | creg_0: 0 ═══════════╩══╬═
        |                         ║
        | creg_1: 0 ══════════════╩═

        """
        # Given
        qreg = QuantumRegister(2, "qreg")
        creg = ClassicalRegister(2, "creg")
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.h(qreg[1])
        qc.cx(qreg[1], qreg[0])
        qc.measure(qreg, creg)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'00': 0.5, '11': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])
