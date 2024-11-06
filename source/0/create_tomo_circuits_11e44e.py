# https://github.com/jarndejong/FTSWAP_experiment/blob/b9d9d171ea25d712d3f77285119c490b018e46e0/filesfortijn/Functions/Create_tomo_circuits.py
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:19:11 2018

@author: Jarnd
"""

# import tomography libary
import qiskit.tools.qcvv.tomography as tomo


def create_tomo_circuits(Quantum_program,
                         circuit_name,
                         quantum_register,
                         classical_register,
                         qubit_list,
                         meas_basis='Pauli',
                         prep_basis='Pauli'):
    '''
    Create tomo set and tomo circuits for the circuit circuit_name in quantum_program.
    The circuits are put in the quantum program, and their names are returned as a list tomo_circuits
    The measurement basis and the preperation basis for the tomography circuits can be specified
    Standard is a Pauli basis for both meas and prep.
    The set of tomography experiments is also returned as tomo_set.
    This function is basically a wrapper for two qiskit:
    functions 'process_tomography_set' and 'create_tomograpy_circuits' in qiskit.tools.qcvv.tomography
    For more information on the containments of tomo_set and tomo_circuits see the documentation of these two functions
    '''
    tomo_set = tomo.process_tomography_set(qubit_list, meas_basis, prep_basis)
    tomo_circuits = tomo.create_tomography_circuits(Quantum_program,
                                                    circuit_name,
                                                    quantum_register,
                                                    classical_register,
                                                    tomo_set)
    return [Quantum_program, tomo_set, tomo_circuits]


def extract_data(results, circuit_name, tomo_set):
    '''
    Extract the data from the results of a tomography experiment.
    Tomo_data is a dictionary containing keys 'data','meas_basis' and 'prep_basis'
    '''
    tomo_data = tomo.tomography_data(results, circuit_name, tomo_set)
    return tomo_data
