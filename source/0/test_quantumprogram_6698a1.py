# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/test/python/test_quantumprogram.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except

"""Quantum Program QISKit Test."""

import os
import unittest
from sys import version_info

import numpy as np

from qiskit import (ClassicalRegister, QISKitError, QuantumCircuit,
                    QuantumRegister, QuantumProgram, Result)
from qiskit.qobj import Qobj
from qiskit.tools import file_io
from .common import requires_qe_access, QiskitTestCase, Path


class TestQuantumProgram(QiskitTestCase):
    """QISKit QuantumProgram Object Tests."""

    def setUp(self):
        self.QASM_FILE_PATH = self._get_resource_path(
            'qasm/entangled_registers.qasm', Path.EXAMPLES)
        self.QASM_FILE_PATH_2 = self._get_resource_path(
            'qasm/plaquette_check.qasm', Path.EXAMPLES)

        self.QPS_SPECS = {
            "circuits": [{
                "name": "circuitName",
                "quantum_registers": [{
                    "name": "q_name",
                    "size": 3}],
                "classical_registers": [{
                    "name": "c_name",
                    "size": 3}]
            }]
        }

    ###############################################################
    # Tests to initiate an build a quantum program
    ###############################################################

    def test_create_program_with_specs(self):
        """Test Quantum Object Factory creation using Specs definition object.

        If all is correct we get a object instance of QuantumProgram

        Previously:
            Objects:
                QPS_SPECS
            Libraries:
                from qiskit import QuantumProgram

        """
        result = QuantumProgram(specs=self.QPS_SPECS)
        self.assertIsInstance(result, QuantumProgram)

    def test_create_program(self):
        """Test Quantum Object Factory creation Without Specs definition object.

        If all is correct we get a object instance of QuantumProgram

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        result = QuantumProgram()
        self.assertIsInstance(result, QuantumProgram)

    @requires_qe_access
    def test_config_scripts_file(self, QE_TOKEN, QE_URL, hub=None, group=None, project=None):
        """Test Qconfig.

        in this case we check if the QE_URL API is defined.

        Previously:
            Libraries:
                import Qconfig
        """
        # pylint: disable=unused-argument
        import re
        self.assertTrue(re.match('^https?://[0-9.:/A-Za-z_-]+/api', QE_URL))

    def test_create_classical_register(self):
        """Test create_classical_register.

        If all is correct we get a object instance of ClassicalRegister

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import ClassicalRegister
        """
        q_program = QuantumProgram()
        cr = q_program.create_classical_register("cr", 3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_quantum_register(self):
        """Test create_quantum_register.

        If all is correct we get a object instance of QuantumRegister

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("qr", 3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_fail_create_quantum_register(self):
        """Test create_quantum_register.

        If all is correct we get a object instance of QuantumRegister and
        QISKitError

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
                from qiskit import QISKitError
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register("qr", 3)
        self.assertIsInstance(qr1, QuantumRegister)
        self.assertRaises(QISKitError, q_program.create_quantum_register,
                          "qr", 2)

    def test_fail_create_classical_register(self):
        """Test create_quantum_register.

        If all is correct we get a object instance of QuantumRegister and
        QISKitError

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
                from qiskit import QISKitError
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register("cr", 3)
        self.assertIsInstance(cr1, ClassicalRegister)
        self.assertRaises(QISKitError,
                          q_program.create_classical_register, "cr", 2)

    def test_create_quantum_register_same(self):
        """Test create_quantum_register of same name and size.

        If all is correct we get a single classical register

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register("qr", 3)
        qr2 = q_program.create_quantum_register("qr", 3)
        self.assertIs(qr1, qr2)

    def test_create_classical_register_same(self):
        """Test create_classical_register of same name and size.

        If all is correct we get a single classical register

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import ClassicalRegister
        """
        q_program = QuantumProgram()
        cr1 = q_program.create_classical_register("cr", 3)
        cr2 = q_program.create_classical_register("cr", 3)
        self.assertIs(cr1, cr2)

    def test_create_classical_registers(self):
        """Test create_classical_registers.

        If all is correct we get a object instance of list[ClassicalRegister]

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import ClassicalRegister
        """
        q_program = QuantumProgram()
        classical_registers = [{"name": "c1", "size": 4},
                               {"name": "c2", "size": 2}]
        crs = q_program.create_classical_registers(classical_registers)
        for i in crs:
            self.assertIsInstance(i, ClassicalRegister)

    def test_create_quantum_registers(self):
        """Test create_quantum_registers.

        If all is correct we get a object instance of list[QuantumRegister]

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumRegister
        """
        q_program = QuantumProgram()
        quantum_registers = [{"name": "q1", "size": 4},
                             {"name": "q2", "size": 2}]
        qrs = q_program.create_quantum_registers(quantum_registers)
        for i in qrs:
            self.assertIsInstance(i, QuantumRegister)

    def test_destroy_classical_register(self):
        """Test destroy_classical_register."""
        q_program = QuantumProgram()
        _ = q_program.create_classical_register('c1', 3)
        self.assertIn('c1', q_program.get_classical_register_names())
        q_program.destroy_classical_register('c1')
        self.assertNotIn('c1', q_program.get_classical_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_classical_register('c1')
        self.assertIn('Not present', str(context.exception))

    def test_destroy_quantum_register(self):
        """Test destroy_quantum_register."""
        q_program = QuantumProgram()
        _ = q_program.create_quantum_register('q1', 3)
        self.assertIn('q1', q_program.get_quantum_register_names())
        q_program.destroy_quantum_register('q1')
        self.assertNotIn('q1', q_program.get_quantum_register_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_quantum_register('q1')
        self.assertIn('Not present', str(context.exception))

    def test_create_circuit(self):
        """Test create_circuit.

        If all is correct we get a object instance of QuantumCircuit

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumCircuit
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("qr", 3)
        cr = q_program.create_classical_register("cr", 3)
        qc = q_program.create_circuit("qc", [qr], [cr])
        self.assertIsInstance(qc, QuantumCircuit)

    def test_create_several_circuits(self):
        """Test create_circuit with several inputs.

        If all is correct we get a object instance of QuantumCircuit

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QuantumCircuit
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register("qr1", 3)
        cr1 = q_program.create_classical_register("cr1", 3)
        qr2 = q_program.create_quantum_register("qr2", 3)
        cr2 = q_program.create_classical_register("cr2", 3)
        qc1 = q_program.create_circuit("qc1", [qr1], [cr1])
        qc2 = q_program.create_circuit("qc2", [qr2], [cr2])
        qc3 = q_program.create_circuit("qc2", [qr1, qr2], [cr1, cr2])
        self.assertIsInstance(qc1, QuantumCircuit)
        self.assertIsInstance(qc2, QuantumCircuit)
        self.assertIsInstance(qc3, QuantumCircuit)

    def test_destroy_circuit(self):
        """Test destroy_circuit."""
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register('qr', 3)
        cr = q_program.create_classical_register('cr', 3)
        _ = q_program.create_circuit('qc', [qr], [cr])
        self.assertIn('qc', q_program.get_circuit_names())
        q_program.destroy_circuit('qc')
        self.assertNotIn('qc', q_program.get_circuit_names())

        # Destroying an invalid register should fail.
        with self.assertRaises(QISKitError) as context:
            q_program.destroy_circuit('qc')
        self.assertIn('Not present', str(context.exception))

    def test_load_qasm_file(self):
        """Test load_qasm_file and get_circuit.

        If all is correct we should get the qasm file loaded in QASM_FILE_PATH

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram()
        name = q_program.load_qasm_file(self.QASM_FILE_PATH, name="")
        result = q_program.get_circuit(name)
        to_check = result.qasm()
        self.log.info(to_check)
        self.assertEqual(len(to_check), 430)

    def test_fail_load_qasm_file(self):
        """Test fail_load_qasm_file.

        If all is correct we should get a QISKitError

        Previously:
            Libraries:
                from qiskit import QuantumProgram
                from qiskit import QISKitError
        """
        q_program = QuantumProgram()
        self.assertRaises(QISKitError,
                          q_program.load_qasm_file, "", name=None)

    def test_load_qasm_text(self):
        """Test load_qasm_text and get_circuit.

        If all is correct we should get the qasm file loaded from the string

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram()
        QASM_string = "// A simple 8 qubit example\nOPENQASM 2.0;\n"
        QASM_string += "include \"qelib1.inc\";\nqreg a[4];\n"
        QASM_string += "qreg b[4];\ncreg c[4];\ncreg d[4];\nh a;\ncx a, b;\n"
        QASM_string += "barrier a;\nbarrier b;\nmeasure a[0]->c[0];\n"
        QASM_string += "measure a[1]->c[1];\nmeasure a[2]->c[2];\n"
        QASM_string += "measure a[3]->c[3];\nmeasure b[0]->d[0];\n"
        QASM_string += "measure b[1]->d[1];\nmeasure b[2]->d[2];\n"
        QASM_string += "measure b[3]->d[3];"
        name = q_program.load_qasm_text(QASM_string)
        result = q_program.get_circuit(name)
        to_check = result.qasm()
        self.log.info(to_check)
        self.assertEqual(len(to_check), 430)

    def test_get_register_and_circuit(self):
        """Test get_quantum_registers, get_classical_registers, and get_circuit.

        If all is correct we get a object instance of QuantumCircuit,
        QuantumRegister, ClassicalRegister

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertIsInstance(qr, QuantumRegister)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_get_register_and_circuit_names(self):
        """Get the names of the circuits and registers.

        If all is correct we should get the arrays of the names

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram()
        qr1 = q_program.create_quantum_register("qr1", 3)
        cr1 = q_program.create_classical_register("cr1", 3)
        qr2 = q_program.create_quantum_register("qr2", 3)
        cr2 = q_program.create_classical_register("cr2", 3)
        q_program.create_circuit("qc1", [qr1], [cr1])
        q_program.create_circuit("qc2", [qr2], [cr2])
        q_program.create_circuit("qc2", [qr1, qr2], [cr1, cr2])
        qrn = q_program.get_quantum_register_names()
        crn = q_program.get_classical_register_names()
        qcn = q_program.get_circuit_names()
        self.assertEqual(qrn, ['qr1', 'qr2'])
        self.assertEqual(crn, ['cr1', 'cr2'])
        self.assertEqual(qcn, ['qc1', 'qc2'])

    def test_get_qasm(self):
        """Test the get_qasm.

        If all correct the qasm output should be of a certain length

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm("circuitName")
        self.assertEqual(len(result), 225)

    def test_get_qasms(self):
        """Test the get_qasms.

        If all correct the qasm output for each circuit should be of a certain
        length

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("qr", 3)
        cr = q_program.create_classical_register("cr", 3)
        qc1 = q_program.create_circuit("qc1", [qr], [cr])
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc1.h(qr[0])
        qc1.cx(qr[0], qr[1])
        qc1.cx(qr[1], qr[2])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc1.measure(qr[2], cr[2])
        qc2.h(qr)
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        result = q_program.get_qasms(["qc1", "qc2"])
        self.assertEqual(len(result[0]), 173)
        self.assertEqual(len(result[1]), 159)

    def test_get_qasm_all_gates(self):
        """Test the get_qasm for more gates.

        If all correct the qasm output should be of a certain length

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc.u1(0.3, qr[0])
        qc.u2(0.2, 0.1, qr[1])
        qc.u3(0.3, 0.2, 0.1, qr[2])
        qc.s(qr[1])
        qc.s(qr[2]).inverse()
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.x(qr[2]).c_if(cr, 0)
        qc.y(qr[2]).c_if(cr, 1)
        qc.z(qr[2]).c_if(cr, 2)
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        result = q_program.get_qasm('circuitName')
        self.assertEqual(len(result), 565)

    def test_get_initial_circuit(self):
        """Test get_initial_circuit.

        If all correct is should be of the circuit form.

        Previously:
            Libraries:
                from qiskit import QuantumProgram
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_initial_circuit()
        self.assertIsInstance(qc, QuantumCircuit)

    def test_save(self):
        """Test save.

        Save a Quantum Program in Json file
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)

        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")

        qc.u3(0.3, 0.2, 0.1, qr[0])
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.z(qr[2]).c_if(cr, 1)
        qc.x(qr[2]).c_if(cr, 1)
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        result = q_program.save(self._get_resource_path('test_save.json'),
                                beauty=True)

        self.assertEqual(result['status'], 'Done')

    def test_save_wrong(self):
        """Test save wrong.

        Save a Quantum Program in Json file: Errors Control
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(LookupError, q_program.load)

    def test_load(self):
        """Test load Json.

        Load a Json Quantum Program
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)

        result = q_program.load(self._get_resource_path('test_load.json'))
        self.assertEqual(result['status'], 'Done')

        check_result = q_program.get_qasm('circuitName')
        self.log.info(check_result)
        # TODO: revise Sympy 1.2 compatibility. The length is 1775 for
        # sympy=1.1.x, and 1781 for sympy=1.2
        self.assertIn(len(check_result), (1775, 1781))

    def test_load_wrong(self):
        """Test load Json.

        Load a Json Quantum Program: Errors Control.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(LookupError, q_program.load)

    ###############################################################
    # Tests for working with backends
    ###############################################################

    @requires_qe_access
    def test_setup_api(self, QE_TOKEN, QE_URL,
                       hub=None, group=None, project=None):
        """Check the api is set up.

        If all correct is should be true.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        config = q_program.get_api_config()
        self.assertTrue(config)

    @requires_qe_access
    def test_available_backends_exist(self, QE_TOKEN, QE_URL,
                                      hub=None, group=None, project=None):
        """Test if there are available backends.

        If all correct some should exists (even if offline).
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        available_backends = q_program.available_backends()
        self.assertTrue(available_backends)

    @requires_qe_access
    def test_online_backends_exist(self, QE_TOKEN, QE_URL,
                                   hub=None, group=None, project=None):
        """Test if there are online backends.

        If all correct some should exists.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        online_backends = q_program.online_backends()
        self.log.info(online_backends)
        self.assertTrue(online_backends)

    @requires_qe_access
    def test_online_simulators(self, QE_TOKEN, QE_URL,
                               hub=None, group=None, project=None):
        """Test if there are online backends (which are simulators).

        If all correct some should exists. NEED internet connection for this.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        qp.set_api(QE_TOKEN, QE_URL, hub, group, project)
        online_simulators = qp.online_simulators()
        self.log.info(online_simulators)
        self.assertTrue(isinstance(online_simulators, list))

    @requires_qe_access
    def test_online_devices(self, QE_TOKEN, QE_URL,
                            hub=None, group=None, project=None):
        """Test if there are online backends (which are devices).

        If all correct some should exists. NEED internet connection for this.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        qp.set_api(QE_TOKEN, QE_URL, hub, group, project)
        online_devices = qp.online_devices()
        self.log.info(online_devices)
        self.assertTrue(isinstance(online_devices, list))

    def test_backend_status(self):
        """Test backend_status.

        If all correct should return dictionary with available: True/False.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        out = q_program.get_backend_status("local_qasm_simulator")
        self.assertIn(out['operational'], [True])

    def test_backend_status_fail(self):
        """Test backend_status.

        If all correct should return dictionary with available: True/False.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(LookupError, qp.get_backend_status, "fail")

    def test_get_backend_configuration(self):
        """Test configuration.

        If all correct should return configuration for the
        local_qasm_simulator.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        config_keys = {'name', 'simulator', 'local', 'description',
                       'coupling_map', 'basis_gates'}
        backend_config = qp.get_backend_configuration("local_qasm_simulator")
        self.assertTrue(config_keys < backend_config.keys())

    @requires_qe_access
    def test_get_backend_configuration_online(self, QE_TOKEN, QE_URL,
                                              hub=None, group=None, project=None):
        """Test configuration.

        If all correct should return configuration for the
        local_qasm_simulator.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        config_keys = {'name', 'simulator', 'local', 'description',
                       'coupling_map', 'basis_gates'}
        qp.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend_list = qp.available_backends()
        backend_list.remove('ibmq_qasm_simulator')
        if backend_list:
            backend = backend_list[0]
        backend_config = qp.get_backend_configuration(backend)
        self.log.info(backend_config)
        self.assertTrue(config_keys < backend_config.keys())

    def test_get_backend_configuration_fail(self):
        """Test configuration fail.

        If all correct should return LookupError.
        """
        qp = QuantumProgram(specs=self.QPS_SPECS)
        self.assertRaises(LookupError, qp.get_backend_configuration, "fail")

    @requires_qe_access
    def test_get_backend_calibration(self, QE_TOKEN, QE_URL,
                                     hub=None, group=None, project=None):
        """Test get_backend_calibration.

        If all correct should return dictionary on length 4.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend_list = q_program.online_devices()
        if backend_list:
            backend = backend_list[0]
        result = q_program.get_backend_calibration(backend)
        self.log.info(result)
        self.assertEqual(len(result), 4)

    @requires_qe_access
    def test_get_backend_parameters(self, QE_TOKEN, QE_URL,
                                    hub=None, group=None, project=None):
        """Test get_backend_parameters.

        If all correct should return dictionary on length 4.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend_list = q_program.online_devices()
        if backend_list:
            backend = backend_list[0]
        result = q_program.get_backend_parameters(backend)
        self.log.info(result)
        self.assertTrue(all(key in result for key in ('last_update_date',
                                                      'qubits', 'backend')))

    ###############################################################
    # Test for compile
    ###############################################################

    def test_compile_program(self):
        """Test compile_program.

        If all correct should return COMPLETED.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        out = q_program.compile(['circuitName'], backend=backend,
                                coupling_map=coupling_map, qobj_id='cooljob')
        self.log.info(out)
        # FIXME should validate the Qobj when defined
        self.assertIsInstance(out, Qobj)

    def test_get_compiled_configuration(self):
        """Test compiled_configuration.

        If all correct should return length 4 dictionary.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = q_program.compile(['circuitName'], backend=backend,
                                 coupling_map=coupling_map)
        result = q_program.get_compiled_configuration(qobj, 'circuitName')
        self.log.info(result)
        self.assertEqual(len(result), 5)

    def test_get_compiled_qasm(self):
        """Test get_compiled_qasm.

        If all correct should return length  dictionary.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        # cannot interchange simulators here due to differing basis
        backend = 'local_qasm_simulator_py'
        coupling_map = None
        qobj = q_program.compile(['circuitName'], backend=backend,
                                 coupling_map=coupling_map)
        result = q_program.get_compiled_qasm(qobj, 'circuitName',)
        self.log.info(result)
        self.assertEqual(len(result), 190)

    def test_get_execution_list(self):
        """Test get_execution_list.

        If all correct should return {'local_qasm_simulator': ['circuitName']}.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qc = q_program.get_circuit("circuitName")
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        backend = 'local_qasm_simulator'
        coupling_map = None
        qobj = q_program.compile(['circuitName'], backend=backend,
                                 coupling_map=coupling_map, qobj_id='cooljob')
        result = q_program.get_execution_list(qobj, print_func=self.log.info)
        self.log.info(result)
        self.assertEqual(result, ['circuitName'])

    def test_compile_coupling_map(self):
        """Test compile_coupling_map.

        If all correct should return data with the same stats. The circuit may
        be different.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 3)
        c = q_program.create_classical_register("c", 3)
        qc = q_program.create_circuit("circuitName", [q], [c])
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.cx(q[0], q[2])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        backend = 'local_qasm_simulator'
        shots = 1024
        coupling_map = [[0, 1], [1, 2]]
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2)}
        circuits = ["circuitName"]
        qobj = q_program.compile(circuits, backend=backend, shots=shots,
                                 coupling_map=coupling_map,
                                 initial_layout=initial_layout, seed=88)
        result = q_program.run(qobj)
        to_check = q_program.get_qasm("circuitName")
        self.assertEqual(len(to_check), 160)

        counts = result.get_counts("circuitName")
        target = {'000': shots / 2, '111': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compile_coupling_map_as_dict(self):
        """Test compile_coupling_map in dict format (to be deprecated).

        TODO: This test is very specific, and should be removed when the only
        format allowed for the coupling map is a `list`.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 3)
        c = q_program.create_classical_register("c", 3)
        qc = q_program.create_circuit("circuitName", [q], [c])
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.cx(q[0], q[2])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        backend = 'local_qasm_simulator'
        shots = 1024
        coupling_map = {0: [1], 1: [2]}  # as dict
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2)}
        circuits = ["circuitName"]
        with self.assertWarns(DeprecationWarning):
            qobj = q_program.compile(circuits, backend=backend, shots=shots,
                                     coupling_map=coupling_map,
                                     initial_layout=initial_layout, seed=88)
        result = q_program.run(qobj)
        to_check = q_program.get_qasm("circuitName")
        self.assertEqual(len(to_check), 160)
        counts = result.get_counts("circuitName")
        target = {'000': shots / 2, '111': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_change_circuit_qobj_after_compile(self):
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc3 = q_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024
        backend = 'local_qasm_simulator'
        config = {'seed': 10, 'shots': 1, 'xvals': [1, 2, 3, 4]}
        qobj1 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        qobj1.experiments[0].config.shots = 50
        qobj1.experiments[0].config.xvals = [1, 1, 1]
        config['shots'] = 1000
        config['xvals'][0] = 'only for qobj2'
        qobj2 = q_program.compile(circuits, backend=backend, shots=shots,
                                  seed=88, config=config)
        self.assertTrue(qobj1.experiments[0].config.shots == 50)
        self.assertTrue(qobj1.experiments[1].config.shots == 1)
        self.assertTrue(qobj1.experiments[0].config.xvals == [1, 1, 1])
        self.assertTrue(qobj1.experiments[1].config.xvals == [1, 2, 3, 4])
        self.assertTrue(qobj1.config.shots == 1024)
        self.assertTrue(qobj2.experiments[0].config.shots == 1000)
        self.assertTrue(qobj2.experiments[1].config.shots == 1000)
        self.assertTrue(qobj2.experiments[0].config.xvals == [
            'only for qobj2', 2, 3, 4])
        self.assertTrue(qobj2.experiments[1].config.xvals == [
            'only for qobj2', 2, 3, 4])

    ###############################################################
    # Test for running programs
    ###############################################################

    def test_run_program(self):
        """Test run.

        If all correct should the data.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc3 = q_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr, cr)
        qc3.measure(qr, cr)
        circuits = ['qc2', 'qc3']
        shots = 1024
        backend = 'local_qasm_simulator'
        qobj = q_program.compile(circuits, backend=backend, shots=shots,
                                 seed=88)
        result = q_program.run(qobj)
        counts2 = result.get_counts("qc2")
        counts3 = result.get_counts("qc3")
        target2 = {'000': shots / 2, '111': shots / 2}
        target3 = {'000': shots / 8, '001': shots / 8, '010': shots / 8,
                   '011': shots / 8, '100': shots / 8, '101': shots / 8,
                   '110': shots / 8, '111': shots / 8}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts2, target2, threshold)
        self.assertDictAlmostEqual(counts3, target3, threshold)

    def test_combine_results(self):
        """Test run.

        If all correct should the data.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("qr", 1)
        cr = q_program.create_classical_register("cr", 1)
        qc1 = q_program.create_circuit("qc1", [qr], [cr])
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc1.measure(qr[0], cr[0])
        qc2.x(qr[0])
        qc2.measure(qr[0], cr[0])
        shots = 1024
        backend = 'local_qasm_simulator'
        res1 = q_program.execute(['qc1'], backend=backend, shots=shots)
        res2 = q_program.execute(['qc2'], backend=backend, shots=shots)
        counts1 = res1.get_counts('qc1')
        counts2 = res2.get_counts('qc2')
        res1 += res2  # combine results
        counts12 = [res1.get_counts('qc1'), res1.get_counts('qc2')]
        self.assertEqual(counts12, [counts1, counts2])

    def test_local_qasm_simulator(self):
        """Test execute.

        If all correct should the data.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc3 = q_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc3.h(qr)
        qc2.measure(qr[0], cr[0])
        qc3.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc3.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        qc3.measure(qr[2], cr[2])
        circuits = ['qc2', 'qc3']
        shots = 1024
        backend = 'local_qasm_simulator'
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   seed=88)
        counts2 = result.get_counts("qc2")
        counts3 = result.get_counts("qc3")
        target2 = {'000': shots / 2, '111': shots / 2}
        target3 = {'000': shots / 8, '001': shots / 8, '010': shots / 8,
                   '011': shots / 8, '100': shots / 8, '101': shots / 8,
                   '110': shots / 8, '111': shots / 8}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts2, target2, threshold)
        self.assertDictAlmostEqual(counts3, target3, threshold)

    def test_local_qasm_simulator_one_shot(self):
        """Test single shot of local simulator .

        If all correct should the quantum state.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc3 = q_program.create_circuit("qc3", [qr], [cr])
        qc2.h(qr[0])
        qc3.h(qr[0])
        qc3.cx(qr[0], qr[1])
        qc3.cx(qr[0], qr[2])
        circuits = ['qc2', 'qc3']
        # the behavior of getting statevector from 1 shot only existed in py simulator
        backend = 'local_qasm_simulator_py'
        shots = 1
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   seed=9)
        statevector = np.array([0.70710678+0.j, 0.70710678+0.j,
                                0.00000000+0.j, 0.00000000+0.j,
                                0.00000000+0.j, 0.00000000+0.j,
                                0.00000000+0.j, 0.00000000+0.j])
        norm = np.dot(np.conj(statevector),
                      result.get_data('qc2')['statevector'])
        self.assertAlmostEqual(norm, 1)
        statevector = np.array([0.70710678+0.j, 0+0.j,
                                0.00000000+0.j, 0.00000000+0.j,
                                0.00000000+0.j, 0.00000000+0.j,
                                0.00000000+0.j, 0.70710678+0.j])
        norm = np.dot(np.conj(statevector),
                      result.get_data('qc3')['statevector'])
        self.assertAlmostEqual(norm, 1)

    def test_local_unitary_simulator(self):
        """Test unitary simulator.

        If all correct should the hxh and cx.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 2)
        c = q_program.create_classical_register("c", 2)
        qc1 = q_program.create_circuit("qc1", [q], [c])
        qc2 = q_program.create_circuit("qc2", [q], [c])
        qc1.h(q)
        qc2.cx(q[0], q[1])
        circuits = ['qc1', 'qc2']
        backend = 'local_unitary_simulator'
        result = q_program.execute(circuits, backend=backend)
        unitary1 = result.get_data('qc1')['unitary']
        unitary2 = result.get_data('qc2')['unitary']
        unitaryreal1 = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
                                 [0.5, 0.5, -0.5, -0.5],
                                 [0.5, -0.5, -0.5, 0.5]])
        unitaryreal2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1],
                                 [0., 0, 1, 0], [0, 1, 0, 0]])
        norm1 = np.trace(np.dot(np.transpose(np.conj(unitaryreal1)), unitary1))
        norm2 = np.trace(np.dot(np.transpose(np.conj(unitaryreal2)), unitary2))
        self.assertAlmostEqual(norm1, 4)
        self.assertAlmostEqual(norm2, 4)

    def test_run_program_map(self):
        """Test run_program_map.

        If all correct should return 10010.
        """
        q_program = QuantumProgram()
        backend = 'local_qasm_simulator'
        shots = 100
        max_credits = 3
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2), ("q", 3): ("q", 3),
                          ("q", 4): ("q", 4)}
        q_program.load_qasm_file(self.QASM_FILE_PATH_2, name="circuit-dev")
        circuits = ["circuit-dev"]
        qobj = q_program.compile(circuits, backend=backend, shots=shots,
                                 max_credits=max_credits, seed=65,
                                 coupling_map=coupling_map,
                                 initial_layout=initial_layout)
        result = q_program.run(qobj)
        self.assertEqual(result.get_counts("circuit-dev"), {'10010': 100})

    def test_execute_program_map(self):
        """Test execute_program_map.

        If all correct should return 10010.
        """
        q_program = QuantumProgram()
        backend = 'local_qasm_simulator'
        shots = 100
        max_credits = 3
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]
        initial_layout = {("q", 0): ("q", 0), ("q", 1): ("q", 1),
                          ("q", 2): ("q", 2), ("q", 3): ("q", 3),
                          ("q", 4): ("q", 4)}
        q_program.load_qasm_file(self.QASM_FILE_PATH_2, "circuit-dev")
        circuits = ["circuit-dev"]
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   max_credits=max_credits,
                                   coupling_map=coupling_map,
                                   initial_layout=initial_layout, seed=5455)
        self.assertEqual(result.get_counts("circuit-dev"), {'10010': 100})

    def test_average_data(self):
        """Test average_data.

        If all correct should return the data.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 2)
        c = q_program.create_classical_register("c", 2)
        qc = q_program.create_circuit("qc", [q], [c])
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        circuits = ['qc']
        shots = 10000
        backend = 'local_qasm_simulator'
        results = q_program.execute(circuits, backend=backend, shots=shots)
        observable = {"00": 1, "11": 1, "01": -1, "10": -1}
        mean_zz = results.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": 1, "10": -1}
        mean_zi = results.average_data("qc", observable)
        observable = {"00": 1, "11": -1, "01": -1, "10": 1}
        mean_iz = results.average_data("qc", observable)
        self.assertAlmostEqual(mean_zz, 1, places=1)
        self.assertAlmostEqual(mean_zi, 0, places=1)
        self.assertAlmostEqual(mean_iz, 0, places=1)

    @requires_qe_access
    def test_execute_one_circuit_simulator_online(self, QE_TOKEN, QE_URL,
                                                  hub=None, group=None, project=None):
        """Test execute_one_circuit_simulator_online.

        If all correct should return the data.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("q", 1)
        cr = q_program.create_classical_register("c", 1)
        qc = q_program.create_circuit("qc", [qr], [cr])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        shots = 1024
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend = 'ibmq_qasm_simulator'
        result = q_program.execute(['qc'], backend=backend,
                                   shots=shots, max_credits=3,
                                   seed=73846087)
        counts = result.get_counts('qc')
        target = {'0': shots / 2, '1': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    @requires_qe_access
    def test_simulator_online_size(self, QE_TOKEN, QE_URL,
                                   hub=None, group=None, project=None):
        """Test test_simulator_online_size.

        If all correct should return the data.
        """
        backend_name = 'ibmq_qasm_simulator'
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("q", 31)
        cr = q_program.create_classical_register("c", 31)
        qc = q_program.create_circuit("qc", [qr], [cr])
        qc.h(qr)
        qc.measure(qr, cr)
        shots = 1
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        result = q_program.execute(['qc'], backend=backend_name, shots=shots,
                                   max_credits=3, seed=73846087)
        self.assertRaises(QISKitError, result.get_data, 'qc')

    @requires_qe_access
    def test_execute_several_circuits_simulator_online(self, QE_TOKEN, QE_URL,
                                                       hub=None, group=None, project=None):
        """Test execute_several_circuits_simulator_online.

        If all correct should return the data.
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("q", 2)
        cr = q_program.create_classical_register("c", 2)
        qc1 = q_program.create_circuit("qc1", [qr], [cr])
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc1.h(qr)
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc1.measure(qr[0], cr[0])
        qc1.measure(qr[1], cr[1])
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        circuits = ['qc1', 'qc2']
        shots = 1024
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend = 'ibmq_qasm_simulator'
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   max_credits=3, seed=1287126141)
        counts1 = result.get_counts('qc1')
        counts2 = result.get_counts('qc2')
        target1 = {'00': shots / 4, '01': shots / 4,
                   '10': shots / 4, '11': shots / 4}
        target2 = {'00': shots / 2, '11': shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts1, target1, threshold)
        self.assertDictAlmostEqual(counts2, target2, threshold)

    @requires_qe_access
    def test_execute_one_circuit_real_online(self, QE_TOKEN, QE_URL,
                                             hub=None, group=None, project=None):
        """Test execute_one_circuit_real_online.

        If all correct should return a result object
        """
        q_program = QuantumProgram()
        qr = q_program.create_quantum_register("qr", 1)
        cr = q_program.create_classical_register("cr", 1)
        qc = q_program.create_circuit("circuitName", [qr], [cr])
        qc.h(qr)
        qc.measure(qr[0], cr[0])
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend = 'ibmq_qasm_simulator'
        shots = 1
        status = q_program.get_backend_status(backend)
        if not status.get('operational', False):
            pass
        else:
            result = q_program.execute(['circuitName'], backend=backend,
                                       shots=shots, max_credits=3)
            self.assertIsInstance(result, Result)

    @unittest.skipIf(version_info.minor == 5,
                     "Due to gate ordering issues with Python 3.5 "
                     "we have to disable this test until fixed")
    def test_local_qasm_simulator_two_registers(self):
        """Test local_qasm_simulator_two_registers.

        If all correct should the data.
        """
        q_program = QuantumProgram()
        q1 = q_program.create_quantum_register("q1", 2)
        c1 = q_program.create_classical_register("c1", 2)
        q2 = q_program.create_quantum_register("q2", 2)
        c2 = q_program.create_classical_register("c2", 2)
        qc1 = q_program.create_circuit("qc1", [q1, q2], [c1, c2])
        qc2 = q_program.create_circuit("qc2", [q1, q2], [c1, c2])

        qc1.x(q1[0])
        qc2.x(q2[1])
        qc1.measure(q1[0], c1[0])
        qc1.measure(q1[1], c1[1])
        qc1.measure(q2[0], c2[0])
        qc1.measure(q2[1], c2[1])
        qc2.measure(q1[0], c1[0])
        qc2.measure(q1[1], c1[1])
        qc2.measure(q2[0], c2[0])
        qc2.measure(q2[1], c2[1])
        circuits = ['qc1', 'qc2']
        shots = 1024
        backend = 'local_qasm_simulator'
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   seed=8458)
        result1 = result.get_counts('qc1')
        result2 = result.get_counts('qc2')
        self.assertEqual(result1, {'00 01': 1024})
        self.assertEqual(result2, {'10 00': 1024})

    @requires_qe_access
    def test_online_qasm_simulator_two_registers(self, QE_TOKEN, QE_URL,
                                                 hub=None, group=None, project=None):
        """Test online_qasm_simulator_two_registers.

        If all correct should the data.
        """
        q_program = QuantumProgram()
        q1 = q_program.create_quantum_register("q1", 2)
        c1 = q_program.create_classical_register("c1", 2)
        q2 = q_program.create_quantum_register("q2", 2)
        c2 = q_program.create_classical_register("c2", 2)
        qc1 = q_program.create_circuit("qc1", [q1, q2], [c1, c2])
        qc2 = q_program.create_circuit("qc2", [q1, q2], [c1, c2])

        qc1.x(q1[0])
        qc2.x(q2[1])
        qc1.measure(q1[0], c1[0])
        qc1.measure(q1[1], c1[1])
        qc1.measure(q2[0], c2[0])
        qc1.measure(q2[1], c2[1])
        qc2.measure(q1[0], c1[0])
        qc2.measure(q1[1], c1[1])
        qc2.measure(q2[0], c2[0])
        qc2.measure(q2[1], c2[1])
        circuits = ['qc1', 'qc2']
        shots = 1024
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        backend = 'ibmq_qasm_simulator'
        result = q_program.execute(circuits, backend=backend, shots=shots,
                                   seed=8458)
        result1 = result.get_counts('qc1')
        result2 = result.get_counts('qc2')
        self.assertEqual(result1, {'00 01': 1024})
        self.assertEqual(result2, {'10 00': 1024})

    ###############################################################
    # More test cases for interesting examples
    ###############################################################

    def test_example_multiple_compile(self):
        """Test a toy example compiling multiple circuits.

        Pass if the results are correct.
        """
        coupling_map = [[0, 1], [0, 2],
                        [1, 2],
                        [3, 2], [3, 4],
                        [4, 2]]
        QPS_SPECS = {
            "circuits": [
                {
                    "name": "ghz",
                    "quantum_registers": [{
                        "name": "q",
                        "size": 5
                    }],
                    "classical_registers": [{
                        "name": "c",
                        "size": 5}]
                },
                {
                    "name": "bell",
                    "quantum_registers": [{
                        "name": "q",
                        "size": 5
                    }],
                    "classical_registers": [{
                        "name": "c",
                        "size": 5}]
                }
            ]
        }
        qp = QuantumProgram(specs=QPS_SPECS)
        ghz = qp.get_circuit("ghz")
        bell = qp.get_circuit("bell")
        q = qp.get_quantum_register("q")
        c = qp.get_classical_register("c")
        # Create a GHZ state
        ghz.h(q[0])
        ghz.cx(q[0], q[0+1])
        ghz.cx(q[1], q[1+1])
        ghz.cx(q[2], q[2+1])
        ghz.cx(q[3], q[3+1])
        # Insert a barrier before measurement
        ghz.barrier()
        ghz.measure(q[0], c[0])
        ghz.measure(q[1], c[1])
        ghz.measure(q[2], c[2])
        ghz.measure(q[3], c[3])
        ghz.measure(q[4], c[4])
        # Create a Bell state
        bell.h(q[0])
        bell.cx(q[0], q[1])
        bell.barrier()
        bell.measure(q[0], c[0])
        bell.measure(q[1], c[1])
        shots = 2048
        bellobj = qp.compile(["bell"], backend='local_qasm_simulator',
                             shots=shots, seed=10)
        ghzobj = qp.compile(["ghz"], backend='local_qasm_simulator',
                            shots=shots, coupling_map=coupling_map,
                            seed=10)
        bellresult = qp.run(bellobj)
        ghzresult = qp.run(ghzobj)
        self.log.info(bellresult.get_counts("bell"))
        self.log.info(ghzresult.get_counts("ghz"))

        threshold = 0.04 * shots
        counts_bell = bellresult.get_counts('bell')
        target_bell = {'00000': shots / 2, '00011': shots / 2}
        self.assertDictAlmostEqual(counts_bell, target_bell, threshold)

        counts_ghz = ghzresult.get_counts('ghz')
        target_ghz = {'00000': shots / 2, '11111': shots / 2}
        self.assertDictAlmostEqual(counts_ghz, target_ghz, threshold)

    def test_example_swap_bits(self):
        """Test a toy example swapping a set bit around.

        Uses the mapper. Pass if results are correct.
        """
        coupling_map = [[0, 1], [0, 8], [1, 2], [1, 9], [2, 3], [2, 10],
                        [3, 4], [3, 11], [4, 5], [4, 12], [5, 6], [5, 13],
                        [6, 7], [6, 14], [7, 15], [8, 9], [9, 10], [10, 11],
                        [11, 12], [12, 13], [13, 14], [14, 15]]

        def swap(qc, q0, q1):
            """Swap gate."""
            qc.cx(q0, q1)
            qc.cx(q1, q0)
            qc.cx(q0, q1)
        n = 3  # make this at least 3
        QPS_SPECS = {
            "circuits": [
                {
                    "name": "swapping",
                    "quantum_registers": [
                        {
                            "name": "q",
                            "size": n
                        },
                        {
                            "name": "r",
                            "size": n
                        }
                    ],
                    "classical_registers": [
                        {
                            "name": "ans",
                            "size": 2*n
                        },
                    ]
                }
            ]
        }
        qp = QuantumProgram(specs=QPS_SPECS)
        backend = 'local_qasm_simulator'
        qc = qp.get_circuit("swapping")
        q = qp.get_quantum_register("q")
        r = qp.get_quantum_register("r")
        ans = qp.get_classical_register("ans")
        # Set the first bit of q
        qc.x(q[0])
        # Swap the set bit
        swap(qc, q[0], q[n-1])
        swap(qc, q[n-1], r[n-1])
        swap(qc, r[n-1], q[1])
        swap(qc, q[1], r[1])
        # Insert a barrier before measurement
        qc.barrier()
        # Measure all of the qubits in the standard basis
        for j in range(n):
            qc.measure(q[j], ans[j])
            qc.measure(r[j], ans[j+n])
        # First version: no mapping
        result = qp.execute(["swapping"], backend=backend,
                            coupling_map=None, shots=1024,
                            seed=14)
        self.assertEqual(result.get_counts("swapping"),
                         {'010000': 1024})
        # Second version: map to coupling graph
        result = qp.execute(["swapping"], backend=backend,
                            coupling_map=coupling_map, shots=1024,
                            seed=14)
        self.assertEqual(result.get_counts("swapping"),
                         {'010000': 1024})

    def test_offline(self):
        import string
        import random
        qp = QuantumProgram()
        FAKE_TOKEN = 'this_token_is_not_going_to_be_sent_nowhere'
        FAKE_URL = 'http://{0}.com'.format(
            ''.join(random.choice(string.ascii_lowercase) for _ in range(63))
        )
        # SDK will throw ConnectionError on every call that implies a connection
        self.assertRaises(QISKitError, qp.set_api, FAKE_TOKEN, FAKE_URL)

    def test_results_save_load(self):
        """Test saving and loading the results of a circuit.

        Test for the 'local_unitary_simulator' and 'local_qasm_simulator'
        """
        q_program = QuantumProgram()
        metadata = {'testval': 5}
        q = q_program.create_quantum_register("q", 2)
        c = q_program.create_classical_register("c", 2)
        qc1 = q_program.create_circuit("qc1", [q], [c])
        qc2 = q_program.create_circuit("qc2", [q], [c])
        qc1.h(q)
        qc2.cx(q[0], q[1])
        circuits = ['qc1', 'qc2']

        result1 = q_program.execute(circuits, backend='local_unitary_simulator')
        result2 = q_program.execute(circuits, backend='local_qasm_simulator')

        test_1_path = self._get_resource_path('test_save_load1.json')
        test_2_path = self._get_resource_path('test_save_load2.json')

        # delete these files if they exist
        if os.path.exists(test_1_path):
            os.remove(test_1_path)

        if os.path.exists(test_2_path):
            os.remove(test_2_path)

        file1 = file_io.save_result_to_file(result1, test_1_path, metadata=metadata)
        file2 = file_io.save_result_to_file(result2, test_2_path, metadata=metadata)

        _, metadata_loaded1 = file_io.load_result_from_file(file1)
        _, metadata_loaded2 = file_io.load_result_from_file(file1)

        self.assertAlmostEqual(metadata_loaded1['testval'], 5)
        self.assertAlmostEqual(metadata_loaded2['testval'], 5)

        # remove files to keep directory clean
        os.remove(file1)
        os.remove(file2)

    def test_qubitpol(self):
        """Test the results of the qubitpol function in Results. Do two 2Q circuits
        in the first do nothing and in the second do X on the first qubit.
        """
        q_program = QuantumProgram()
        q = q_program.create_quantum_register("q", 2)
        c = q_program.create_classical_register("c", 2)
        qc1 = q_program.create_circuit("qc1", [q], [c])
        qc2 = q_program.create_circuit("qc2", [q], [c])
        qc2.x(q[0])
        qc1.measure(q, c)
        qc2.measure(q, c)
        circuits = ['qc1', 'qc2']
        xvals_dict = {circuits[0]: 0, circuits[1]: 1}

        result = q_program.execute(circuits, backend='local_qasm_simulator')

        yvals, xvals = result.get_qubitpol_vs_xval(2, xvals_dict=xvals_dict)

        self.assertTrue(np.array_equal(yvals, [[-1, -1], [1, -1]]))
        self.assertTrue(np.array_equal(xvals, [0, 1]))

    def test_ccx(self):
        """Checks a Toffoli gate.

        Based on https://github.com/QISKit/qiskit-terra/pull/172.
        """
        Q_program = QuantumProgram()
        q = Q_program.create_quantum_register('q', 3)
        c = Q_program.create_classical_register('c', 3)
        pqm = Q_program.create_circuit('pqm', [q], [c])

        # Toffoli gate.
        pqm.ccx(q[0], q[1], q[2])
        pqm.measure(q[0], c[0])
        pqm.measure(q[1], c[1])
        pqm.measure(q[2], c[2])

        # Prepare run.
        circuits = ['pqm']
        backend = 'local_qasm_simulator'
        shots = 1024

        # Run.
        result = Q_program.execute(circuits, backend=backend, shots=shots,
                                   max_credits=3, timeout=240)

        self.assertEqual({'000': 1024}, result.get_counts('pqm'))

    def test_reconfig(self):
        """Test reconfiguring the qobj from 1024 shots to 2048 using
        reconfig instead of recompile
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc2.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])
        qc2.measure(qr[2], cr[2])
        shots = 1024
        backend = 'local_qasm_simulator'
        test_config = {'foo': 0, 'bar': 1}
        qobj = q_program.compile(['qc2'], backend=backend, shots=shots, config=test_config)
        out = q_program.run(qobj)
        results = out.get_counts('qc2')

        # change the number of shots and re-run to test if the reconfig does not break
        # the ability to run the qobj
        qobj = q_program.reconfig(qobj, shots=2048)
        out2 = q_program.run(qobj)
        results2 = out2.get_counts('qc2')

        self.assertEqual(results, {'000': 1024})
        self.assertEqual(results2, {'000': 2048})

        # change backend
        qobj = q_program.reconfig(qobj, backend='local_unitary_simulator')
        self.assertEqual(qobj.config.backend, 'local_unitary_simulator')
        # change maxcredits
        qobj = q_program.reconfig(qobj, max_credits=11)
        self.assertEqual(qobj.config.max_credits, 11)
        # change seed
        qobj = q_program.reconfig(qobj, seed=11)
        self.assertEqual(qobj.experiments[0].config.seed, 11)
        # change the config
        test_config_2 = {'foo': 2}
        qobj = q_program.reconfig(qobj, config=test_config_2)
        self.assertEqual(qobj.experiments[0].config.foo, 2)
        self.assertEqual(qobj.experiments[0].config.bar, 1)

    def test_timeout(self):
        """Test run.

        If all correct should the data.
        """
        # TODO: instead of skipping, the test should be fixed in Windows
        # platforms. It currently fails during registering DummySimulator.
        if os.name == 'nt':
            raise unittest.SkipTest('Test not supported in Windows')

        # TODO: use the backend directly when the deprecation is completed.
        from ._mockutils import DummyProvider
        import qiskit.wrapper
        qiskit.wrapper._wrapper._DEFAULT_PROVIDER.add_provider(DummyProvider())

        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc2.measure(qr, cr)
        circuits = ['qc2']
        shots = 1024
        backend = 'local_dummy_simulator'
        qobj = q_program.compile(circuits, backend=backend, shots=shots,
                                 seed=88)
        from concurrent import futures
        self.assertRaises(futures.TimeoutError, q_program.run, qobj,
                          timeout=0.01)

    @requires_qe_access
    def test_hpc_parameter_is_correct(self, QE_TOKEN, QE_URL,
                                      hub=None, group=None, project=None):
        """Test for checking HPC parameter in compile() method.
        It must be only used when the backend is ibmq_qasm_simulator (HPC).
        It will warn the user if the parameter is passed correctly but the
        backend is not ibmq_qasm_simulator (HPC).
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc2.measure(qr, cr)
        circuits = ['qc2']
        shots = 1
        backend = 'ibmq_qasm_simulator'
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        qobj = q_program.compile(circuits, backend=backend, shots=shots,
                                 seed=88,
                                 hpc={'multi_shot_optimization': True,
                                      'omp_num_threads': 16})
        self.assertTrue(qobj)

    @requires_qe_access
    def test_hpc_parameter_is_incorrect(self, QE_TOKEN, QE_URL,
                                        hub=None, group=None, project=None):
        """Test for checking HPC parameter in compile() method.
        It must be only used when the backend is ibmq_qasm_simulator (HPC).
        If the parameter format is incorrect, it will raise a QISKitError.
        """
        q_program = QuantumProgram(specs=self.QPS_SPECS)
        qr = q_program.get_quantum_register("q_name")
        cr = q_program.get_classical_register("c_name")
        qc2 = q_program.create_circuit("qc2", [qr], [cr])
        qc2.h(qr[0])
        qc2.cx(qr[0], qr[1])
        qc2.cx(qr[0], qr[2])
        qc2.measure(qr, cr)
        circuits = ['qc2']
        shots = 1
        backend = 'ibmq_qasm_simulator'
        q_program.set_api(QE_TOKEN, QE_URL, hub, group, project)
        self.assertRaises(QISKitError, q_program.compile, circuits,
                          backend=backend, shots=shots, seed=88,
                          hpc={'invalid_key': None})


if __name__ == '__main__':
    unittest.main(verbosity=2)
