# https://github.com/svenpruefer/quantumcomputing/blob/e082b6b829ccabdf1c9c64b5cc310ba8feaad2d5/qsoc/tests/unit_tests/graphs/test_max_cut.py
# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *

from qsoc.costs.costs import calc_total_costs
from qsoc.circuits.max_cut import add_max_cut_circuit


class TestMaxCut:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'absolute_error': 0.03}

    def test_max_cut_on_default_example(self, simulator, config) -> None:
        # Given
        vertices = QuantumRegister(4, 'vertices')
        edges = QuantumRegister(3, 'edges')
        summation = QuantumRegister(2, 'summation')
        ancilla = QuantumRegister(1, 'ancilla')
        vertices_measure = ClassicalRegister(4, 'vertices-measure')
        ancilla_measure = ClassicalRegister(1, 'ancilla-measure')
        qc = QuantumCircuit(vertices, edges, summation, ancilla, vertices_measure, ancilla_measure,
                            name="max-cut-circuit")

        # Add MAX CUT circuit
        add_max_cut_circuit(qc, vertices, edges, summation, ancilla[0])
        # Measure results
        qc.measure(ancilla, ancilla_measure)
        qc.measure(vertices, vertices_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'ancilla vertices'
        # TODO How to deal with ancilla qubits in a systematic way?
        expected_results: Dict[str, float] = {'0 0000': 0,
                                              '0 0001': 0.25,
                                              '0 0010': 0,
                                              '0 0011': 0,
                                              '0 0100': 0,
                                              '0 0101': 0,
                                              '0 0110': 0,
                                              '0 0111': 0,
                                              '0 1000': 0,
                                              '0 1001': 0,
                                              '0 1010': 0,
                                              '0 1011': 0,
                                              '0 1100': 0,
                                              '0 1101': 0,
                                              '0 1110': 0.25,
                                              '0 1111': 0,
                                              '1 0000': 0,
                                              '1 0001': 0.25,
                                              '1 0010': 0,
                                              '1 0011': 0,
                                              '1 0100': 0,
                                              '1 0101': 0,
                                              '1 0110': 0,
                                              '1 0111': 0,
                                              '1 1000': 0,
                                              '1 1010': 0,
                                              '1 1001': 0,
                                              '1 1011': 0,
                                              '1 1100': 0,
                                              '1 1101': 0,
                                              '1 1110': 0.25,
                                              '1 1111': 0}
        assert result == approx(expected_results, abs=config['absolute_error'])
        assert calc_total_costs(qc) == 1538
