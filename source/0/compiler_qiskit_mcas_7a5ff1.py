# https://github.com/maxkeller321/Compiler-mcas-qiskit/blob/6568f1e0a970e27f532f04cc59fc66ed7da38f39/compiler_qiskit_mcas.py
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.compiler.transpile import CouplingMap
from qiskit.circuit import Parameter

import numpy as np 
import textwrap
import os 
import time
from collections.abc import Iterable
from .file_builder import Mcas_file

default_sweeps = 30

def get_initial_and_final_NV_state(qc):
    """
        Returns the initial as well as final state in case they were specified by the users 
            If they were not specified a None is returned
        Input: 
            qc: Instance of qiskits QuantumCircuit
        Returns: 
            initial_state: Inital NV-State or None  
            readout_state: Final NV-State or None 
    """
    if isinstance(qc, QuantumCircuit):
        initial_state = '+++'
        readout_state = '+++'
        print(qc.parameters)
        for param in list(qc.parameters): 
            if 'initial_state' == param.name: 
                initial_state = param.values
            if 'readout_state' == param.name: 
                readout_state = param.values

    return initial_state, readout_state

def transpile_ciruit_for_diamond(quantum_circuit_instance):
    """
    Compiles/ transpiles the QuantumCircuit instance from qiskit to a quantum_circuit_instance instance which can be run on our NV-Center QC
        params: 
            quantum_circuit_instance: instance of quantum_circuit_instance
        return: 
            transpiled_qc: transpiled QunatumCircuit instance
    """

    if not isinstance(quantum_circuit_instance, QuantumCircuit): 
        raise Exception("The passed instance (quantum_circuit_instance) must come from the qiskit class QuantumCircuit")
    
    coupling_string = [[0, 1], [1, 0], [2, 0], [0,2], [1,2], [2, 1]]
    CM = CouplingMap(coupling_string)
    basis_gates = ['id', 'ry', 'rx', 'cx']
    transpiled_qc = transpile(quantum_circuit_instance, coupling_map=CM, basis_gates=basis_gates, optimization_level=3, seed_transpiler=1)

    return transpiled_qc

def construct_full_mcas_file(qc, username, desired_file_path): 
    """
        Takes the qc instance received from the Website and converts it into a mcas_file
        with a timestamp and the username included in the filename. 
        
        params: 
            qc: instance of QuantumCircuit
            username: Username of the code submitter, to create a mapping between him and the mcas_file

        return;
            name of the python file which was constructed 
    """
    initial_state, readout_state = get_initial_and_final_NV_state(qc)
    transpiled_circuit = transpile_ciruit_for_diamond(qc)
    operation_json = construct_operation_list_from_qc_instance(transpiled_circuit)

    mcas_writer = McasTranslator()
    mcas_writer.add_num_sweeps(default_sweeps)
    mcas_writer.add_nuclear_spin_initialisation(initial_state)
    mcas_writer.interprete_json_operations(operation_json)
    mcas_writer.add_nuclear_spin_readout(readout_state)

    time_created = int(time.time())
    file_name = '_'.join(('mcas_file', username, str(time_created)))
    mcas_writer.write_mcas_file(desired_file_path, file_name)

    return file_name



def construct_operation_list_from_qc_json(qc_data):
    """
        Constructs a list of json objects from the json data of a QuantumCircuit instance.
        Each json object contains the data for a single gate/ operation
        The order in the list is equivalent to the execution order for the quantum computer.
        params: 
            qc_data: json data of QuantumCircuit
        return: 
            list of json objects where each object contains the important data for a single gate/ operation
    """
    qc_operations = []

    for gate_data in qc_data:
        operation_json = {}

        gate_instance = gate_data[0]
        operation_name = gate_instance.name 
        if len(gate_instance.params) > 0: 
            gate_params = gate_instance.params[0]
            if isinstance(gate_params, Parameter): 
                operation_json['operation_params'] = gate_params.name
                operation_json['operation_params_iterable'] = gate_params.values
            else: 
                operation_json['operation_params'] = gate_params

        operation_json['operation_name'] = operation_name

        if gate_instance.num_qubits == 2: 
            controlled_qubit = gate_data[1][1].index
            controlling_qubit = gate_data[1][0].index
            operation_json['controlled_qubit'] = controlled_qubit
            operation_json['controlling_qubit'] = controlling_qubit 
        else: 
            qubit_index = gate_data[1][0].index
            operation_json['qubit_index'] = qubit_index

        qc_operations.append(operation_json)

    return qc_operations

def construct_operation_list_from_qc_instance(qc_instance): 
    """
        Constructs a list of json objects from a qiskit QuantumCircuit instance.
        Each json object contains the data for a single gate/ operation.
        The order in the list is equivalent to the execution order for the quantum computer.
        params: 
            qc_instance: instance of QuantumCircuit
        return: 
            list of json objects where each object contains the important data for a single gate/ operation
    """

    if not isinstance(qc_instance, QuantumCircuit): 
        raise Exception("The passed instance (quantum_circuit_instance) must come from the qiskit class QuantumCircuit")
    
    quantum_circuit_data = qc_instance.data

    return construct_operation_list_from_qc_json(quantum_circuit_data)


def match_integers_with_nuclear_spin(qubit_integer):
    """
        this function maps the integer qubit representation to our corresponding 
        nuclear spins
        param: 
            qubit_integer: qubit integer from qiskit (e.g: 0 or 1 or 2)
    """

    if qubit_integer == 0:
        nuclear_spin = '14n'
    elif qubit_integer == 1: 
        nuclear_spin = '13c414'
    elif qubit_integer == 2: 
        nuclear_spin = '13c90'
    else: 
        raise Exception("qubit_integer must be between 0<=x<=2, but was {}".format(qubit_integer))
    
    return nuclear_spin

class McasTranslator:
    """
    Contains all functions to translate a list of json quantum-operations 
    into an mcas file which can be executed from our worker
    """

    def __init__(self):
        self.mcas_sequence_list = [] # list of strings with the operations for the mcas file
        self.list_of_iterables = [] # list of strings with the iterable parameters (for e.g. rabis)

    def interprete_json_operations(self, gate_operation_json):
        if isinstance(gate_operation_json, Iterable):
            
            for operation in gate_operation_json:
                if 'qubit_index' in operation and 'operation_name' in operation and 'operation_params' in operation: 
                    qubit_index = operation['qubit_index'] 

                    if 'operation_params_iterable' in operation: 
                        self.add_iterable_parameter(operation['operation_params'], operation['operation_params_iterable'])
                        operation['operation_params'] = "_I_['{}']".format(operation['operation_params'])

                    if operation['operation_name'] not in ['barrier', 'measure']:
                        if qubit_index == 0: 
                            if operation['operation_name'] == 'rx': 
                                self.add_rx_nitrogen(operation['operation_params'])
                            elif operation['operation_name'] == 'ry': 
                                self.add_ry_nitrogen(operation['operation_params'])
                            else: 
                                raise Exception("Invalid operation encountered in qubit {} the operation json" \
                                            "Only rx, ry and cnot are valid for our Backend.".format(qubit_index))
                        elif qubit_index == 1: 
                            if operation['operation_name'] == 'rx': 
                                self.add_rx_c_414(operation['operation_params'])
                            elif operation['operation_name'] == 'ry': 
                                self.add_ry_c_414(operation['operation_params'])
                            else: 
                                raise Exception("Invalid operation encountered in qubit {} the operation json" \
                                            "Only rx, ry and cnot are valid for our Backend.".format(qubit_index))

                        elif qubit_index == 2: 
                            if operation['operation_name'] == 'rx': 
                                self.add_rx_c_90(operation['operation_params'])
                            elif operation['operation_name'] == 'ry': 
                                self.add_ry_c_90(operation['operation_params'])
                            else: 
                                raise Exception("Invalid operation encountered in qubit {} the operation json" \
                                            "Only rx, ry and cnot are valid for our Backend.".format(qubit_index))
                        else: 
                            raise Exception("Invalid qubit index encountered. Our system does only have 3 qubits: 0, 1 & 2")
            
                elif 'controlled_qubit' in operation and 'controlling_qubit' in operation and 'operation_name' in operation:
                    self.add_cx(operation['controlled_qubit'], operation['controlling_qubit'])

                elif 'operation_name' in operation: 
                    if operation['operation_name'] == 'barrier' or operation['operation_name'] == 'measure': 
                        pass
                    else: 
                        raise Exception("The passed operation: {} is not supported!".format(operation['operation_name']))
                    
                else: 
                    raise Exception("The passed operation is not valid!")
        else: 
            raise Exception("{} is not iterable!. Insert a valid list with json operations!".format(gate_operation_json))
    
    def add_rx_nitrogen(self, theta, ms=0):
        self.mcas_sequence_list.append("rx_nitrogen(mcas, {}, ms={})".format(theta, ms))

    def add_ry_nitrogen(self, theta, ms=0):
        self.mcas_sequence_list.append("ry_nitrogen(mcas, {}, ms={})".format(theta, ms))

    def add_rx_c_414(self, theta, ms=-1):
        self.add_electron_pi()
        self.mcas_sequence_list.append("rx_carbon_414(mcas, {}, ms={})".format(theta, ms))
        self.add_electron_pi()

    def add_ry_c_414(self, theta, ms=-1):
        self.add_electron_pi()
        self.mcas_sequence_list.append("ry_carbon_414(mcas, {}, ms={})".format(theta, ms))
        self.add_electron_pi()

    def add_rx_c_90(self, theta, ms=-1):
        self.add_electron_pi()
        self.mcas_sequence_list.append("rx_carbon_90(mcas, {}, ms={})".format(theta, ms))
        self.add_electron_pi()

    def add_ry_c_90(self, theta, ms=-1):
        self.add_electron_pi()
        self.mcas_sequence_list.append("ry_carbon_90(mcas, {}, ms={})".format(theta, ms))
        self.add_electron_pi()

    def add_cx(self, controlled_qubit, controlling_qubit): 
        controlled_nuclear_spin = match_integers_with_nuclear_spin(controlled_qubit)
        controlling_nuclear_spin = match_integers_with_nuclear_spin(controlling_qubit)
        self.mcas_sequence_list.append("crot_pi_x_between_nuclear_spins(mcas, '{}', '{}')".format(controlled_nuclear_spin, controlling_nuclear_spin))

    def add_nuclear_spin_initialisation(self, state): 
        self.mcas_sequence_list.append("full_initialisation(mcas, '{}')".format(state))

    def add_nuclear_spin_readout(self, state):
        self.mcas_sequence_list.append("readout_nuclear_spin_state(mcas, '{}')".format(state))
    
    def add_num_sweeps(self, num_sweeps): 
        self.list_of_iterables.append("('sweeps', range({})),".format(num_sweeps))
    
    def add_electron_pi(self):
        self.mcas_sequence_list.append("electron_pi_pulse(mcas)")

    def add_iterable_parameter(self, name, iterable): 
        self.list_of_iterables.append("('{}', {}),".format(name, list(iterable)))

    def write_mcas_file(self, destination, mcas_file_name): 
        mcas_file = Mcas_file(self.mcas_sequence_list, parameters=self.list_of_iterables)
        with open(os.path.join(destination, mcas_file_name + '.py'), 'w') as output_file: 
            output_file.write(mcas_file.get_content())
            

