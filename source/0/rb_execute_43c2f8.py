# https://github.com/dveldhuiz/BEP-RB/blob/fe8bf6d983a9411864cdce5da1f8415660cd6460/modules/rb_execute.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:59:08 2021

@author: daniel
"""
import qiskit

def execute(rb_circs, basis_gates, noise_model, shots, backend, coupling_map=None):
    """ Directly implemented  from the qiskit RB tutorial"""
    result_list = []
    transpile_list = []
    
    for rb_seed,rb_circ_seed in enumerate(rb_circs):
        print('Compiling seed %d'%rb_seed)
        rb_circ_transpile = qiskit.transpile(rb_circ_seed, basis_gates=basis_gates, coupling_map=coupling_map)
        print('Simulating seed %d'%rb_seed)
        job = qiskit.execute(rb_circ_transpile, noise_model=noise_model, shots=shots, backend=backend, max_parallel_experiments=0)
        result_list.append(job.result())
        transpile_list.append(rb_circ_transpile)
    print("Finished Simulating")
    return result_list, transpile_list