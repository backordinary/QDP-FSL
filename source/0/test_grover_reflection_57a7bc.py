# https://github.com/svenpruefer/quantumcomputing/blob/e082b6b829ccabdf1c9c64b5cc310ba8feaad2d5/qsoc/tests/unit_tests/basics/test_grover_reflection.py
# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from qiskit import *
from qiskit.providers import *

from qsoc.circuits.grover import add_grover_reflection_no_ancilla, add_grover_reflection_with_ancilla


class TestGroverReflection:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 1000,
                'relative_error': 0.05}

    def test_grover_reflection_without_ancilla(self, simulator, config):
        register = QuantumRegister(5, name="input")
        qc = QuantumCircuit(register, name="test-circuit")
        add_grover_reflection_no_ancilla(qc, register)

        pass  # TODO Implement test for Grover reflection without ancilla qubits

    def test_grover_reflection_with_ancilla(self, simulator, config):
        register = QuantumRegister(5, name="input")
        ancilla = QuantumRegister(2, name="ancilla")
        qc = QuantumCircuit(register, ancilla, name="test-circuit")
        add_grover_reflection_with_ancilla(qc, register, list(ancilla))

        pass  # TODO Implement test for Grover reflection with ancilla qubits