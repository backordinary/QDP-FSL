# https://github.com/Alan-Robertson/quantum_measurement_error_mitigation/blob/98c7080d1f5c3aca7e0c6c819db77de181424c40/tests/test_jigsaw.py
import unittest as pyunit

from PatchedMeasCal import jigsaw
from PatchedMeasCal.benchmarks import bv
from PatchedMeasCal.fake_backends import Grid
from PatchedMeasCal.utils import norm_results_dict

import qiskit


class StatePrepTest(pyunit.TestCase):

    def test_jigsaw_advantage(self):
        bv_str = '111'
        backend = Grid(2, 2)
        n_qubits = 4

        # Base results
        initial_layout = list(range(n_qubits))
        circuit = bv.bv_circuit(bv_str, n_qubits)

        tc = qiskit.transpile(circuit, backend=backend, optimization_level=0, initial_layout=initial_layout)
        res_d = qiskit.execute(tc, backend, n_shots=1024, optimization_level=0, initial_layout=initial_layout).result().get_counts()
        norm_results_dict(res_d)

        jigsaw_res = jigsaw.jigsaw(circuit, backend, 16000, probs=None)
        
        # This should statistically almost always hold
        assert(jigsaw_res['1' * n_qubits] > res_d['1' * n_qubits])


    def test_convolve(self):
        '''
        Testing Bayes update using tables from the paper
        '''
        eps = 0.05

        global_pmf_table = {
        '000':0.1,
        '001':0.10,
        '010':0.15,
        '011':0.15,
        '100':0.10,
        '101':0.05,
        '110':0.15,
        '111':0.2}

        local_table = {'00':0.1, '01':0.1, '10':0.2, '11':0.6}

        local_pair = [1, 2] # Indices

        updated_table = jigsaw.convolve(global_pmf_table, local_table, local_pair)
        
        expected_table = {
        '000':0.05,
        '001':0.07,
        '010':0.13,
        '011':0.64,
        '100':0.05,
        '101':0.04,
        '110':0.13,
        '111':0.86
        }

        norm_val = sum(expected_table.values())
        for i in expected_table:
            expected_table[i] /= norm_val

        for i in expected_table:
            assert(abs(expected_table[i] - updated_table[i]) < eps)

            



        
    
