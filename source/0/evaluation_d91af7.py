# https://github.com/tomtheweber/quantumcomputing/blob/1e84a85f01ecf8ea996de038bd5d3ba241f8baa2/my_custom/evaluation.py
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:05:14 2020

@author: Tom Weber
"""
from typing import Optional, Union, List
from qiskit import  QuantumCircuit, Aer
from qiskit.aqua.operators import (OperatorBase, ExpectationBase, ExpectationFactory, StateFn,
                                   CircuitStateFn, LegacyBaseOperator, CircuitSampler)
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.providers import BaseBackend
from qiskit.circuit import Instruction, Parameter
import numpy as np

import logging



logger = logging.getLogger(__name__)


"""
Class for evaluating operators on a circuit or a variational form.
"""
class Evaluation:
    
    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 circuit: Optional[Union[QuantumCircuit, VariationalForm]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None,
                 wrapping_gates: Optional[List[Instruction]] = None
                 ) -> None:
        
        
        self._expectation=expectation
        self._wrapping_gates=wrapping_gates
        
        if quantum_instance:
            self.quantum_instance=quantum_instance
        else: 
            self.quantum_instance=Aer.get_backend('qasm_simulator')
            
            
        self._circuit_sampler = CircuitSampler(self.quantum_instance)
        
        
        """
        Give error message if circuit or operator is missing.
        """
        if circuit is None:
            raise AquaError("Quantum circuit needed for evaluation")
        else: 
            self.circuit=circuit
        
        if operator is None:
            raise AquaError("No operator was passed")
        
        else:
            self.operator=operator
            
        
        self._var_form_params = sorted(self.circuit.parameters, key=lambda p: p.name)
            
        
    
    """
    Function for computing the expectation value
    """
    def call(self,
             parameters: Optional[Union[List[float], List[Parameter], np.ndarray]] = []
             ) -> float:
        
        if self.operator.num_qubits!=self.circuit.num_qubits:
            raise AquaError("Number of qubits of the circuit does not match the one of the operator.")
            
            
        if len(parameters)!=self.circuit.num_parameters:
            raise AquaError("Number of parameters provided does not match number of parameters of the circuit.") 
        
        num_parameters=self.circuit.num_parameters
        
        
        self._expect_op = self.construct_circuit(self._var_form_params)
        
        
        if len(parameters)!=0:
            
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            # Create dict associating each parameter with the lists of parameterization values for it
            param_bindings = dict(zip(self._var_form_params, parameter_sets.transpose().tolist()))

        else:
            
            param_bindings=[]


        sampled_expect_op = self._circuit_sampler.convert(self._expect_op, params=param_bindings)
        means = np.real(sampled_expect_op.eval())


        

        return means #if len(means) > 1 else means[0]
        
        

    
    
    
    """
    Function for creating an evaluation circuit composed of the operator and ansatz circuit.
    """
    def construct_circuit(self,
                          parameter: Optional[Union[List[float], List[Parameter], np.ndarray]] = []
                          ) -> OperatorBase:
        
        if self.operator.num_qubits!=self.circuit.num_qubits:
            raise AquaError("Number of qubits of the circuit does not match the one of the operator.")
            
            
        if len(parameter)!=self.circuit.num_parameters:
            raise AquaError("Number of parameters provided does not match number of parameters of the circuit.")
        
        
        elif self.circuit.num_parameters!=0:
            
            
            
            if isinstance(self.circuit, QuantumCircuit):
                param_dict = dict(zip(self._var_form_params, parameter))
                wave_function = self.circuit.assign_parameters(param_dict)
            else:
                wave_function = self.circuit.construct_circuit(parameter)
                
        else: wave_function=self.circuit
        
                
        
        self.expectation = ExpectationFactory.build(operator=self.operator,
                                                        backend=self.quantum_instance,
                                                        include_custom=False)
        
        
        observable_meas = self.expectation.convert(StateFn(self.operator, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        return observable_meas.compose(ansatz_circuit_op).reduce()        
        