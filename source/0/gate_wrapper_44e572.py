# https://github.com/tomtheweber/quantumcomputing/blob/1e84a85f01ecf8ea996de038bd5d3ba241f8baa2/my_custom/gate_wrapper.py
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:08:35 2020

@author: Hando
"""
from typing import Optional, List, Callable, Union, Dict
from qiskit import QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.circuit.library.standard_gates import YGate, ZGate, XGate, HGate, IGate



class GateWrapper:
    
    def __init__(self,
                 circuit)->None:
        
        if circuit is None:
            raise AquaError("No quantum circuit was passed.")
        else: 
            self.circuit=circuit
        
        
    def wrap(self, gates=None):
        
        if gates is None:
            raise AquaError("No gates were passed.")
        
        wrapped_qc=QuantumCircuit(self.circuit.num_qubits)
        
        for gate in gates:
        
            for inst, qargs, cargs in self.circuit.data:
            
                
                wrapped_qc.append(inst, qargs, cargs)            
            
                if inst.name in ("barrier", "measure"):
                    continue
                else:
        
                    if len(qargs)==1:
                        wrapped_qc.append(gate, qargs, cargs)
                        """
                    else:
                        wrapped_qc.append(HGate(), [qargs[0]], [])
                        wrapped_qc.append(ZGate(), [qargs[1]], [])
                        """
                    
    
            
        
        return wrapped_qc