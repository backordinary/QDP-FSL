# https://github.com/Gnomebert/Dwave/blob/07cce80f7c12af5c05e257c56e03dfca56394e67/STFC_grant/VQE_Qiskit.py
#VQE_Qiskit
#VQE inherited classes specific to Rigetti, and Rigetti QPU commands
from collections import Counter
from qiskit.circuit import Parameter
from qiskit.circuit import  QuantumCircuit 
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import Z,I, StateFn
#from qiskit.aqua.operators import (OperatorBase, X, I, H, Z, Y, Zero, CircuitStateFn,
#                                  EvolutionFactory, LegacyBaseOperator, OperatorStateFn)
import sys
from VQE_creation import ansatz_theta, ansatz_diff, ansatz_build, EV_all_edges,abstract_ops ,observable_aqc
"""classes created 
        class rigetti_ops(): passed  param 'qpu_skd_class= rigetti_ops' to ansatz_theta, or its subclasses

        Still need to create more _AQC progs
"""
class qiskit_ops(abstract_ops):
        """
        The QPU commands of a 'qiskit' 
        qpu_skd_class=qiskit_ops
        ops_class = qpu_skd_class()
        eg ansatz_theta.ops_class.Program_AQC() creates ;
            self.circ = QuantumCircuit(self.n_q)
        """
        def __init__(self):
            self.n_q = 0 #this is set by ansatz_build_qiskit.get_QPU_ops()
            self.name ='qiskit'
        def Program_AQC(self):
                """A virtual function that overides the base function of the same name.
                creates the qiskit equivalent of Program_AQC() ie 
                qiskit_ops.Program_AQC() = QuantumCircuit(self.n_q)
                """
                self.circ = QuantumCircuit(self.n_q)
                return self.circ
        
       
        def CNOT_AQC(self,cntrl=0,target=1):
            """
                A virtual function that overides the base function of the same name.

                cntrl: The qubit that controls the X gate
                target: The target qubit. The target qubit has an X-gate applied to it if the control

                Produces a controlled-NOT (controlled-X) gate and adds it to a self.circ:
            """
            self.circ.cnot(cntrl,target)
        def H_AQC(self,target):
             self.circ.h(target)
        def I_AQC(self,target):
                 self.circ.I(target)
        def X_AQC(self,target):
            self.circ.x(target)
                 
        def Y_AQC(self,target):
            self.circ.y(target)
        def Z_AQC(self,target):
            self.circ.z(target)
        
        def RY_AQC(self,angle,target):
            self.circ.ry(angle,target)
        def RX_AQC(self,angle,target):
            self.circ.rx(angle,target)
        def RZ_AQC(self,angle,target):
            self.circ.rz(angle,target)
        # Controlled ops
        def CZ_AQC(self,target,cntrl):
            self.circ.cz(target,cntrl)
        """
        """

class ansatz_build_qiskit(ansatz_build): 
    
        def get_QPU_ops(self)    :
            """Virtual function
            creates an instance of the class qiskit_ops() of pauli operators that use qiskit functions
            """
            #instance 
            self.ops_class = qiskit_ops()
            #set number of qubit without an ancilla
            self.ops_class.n_q = self.n_q
class ansatz_diff_qiskit(ansatz_diff):  
    def get_QPU_ops(self)    :
                """Virtual function
                creates an instance of the class qiskit_ops() of pauli operators that use qiskit functions
                """
                #instance 
                self.ops_class = qiskit_ops()  
                if self.no_had_gates==0:
                    #set number of qubit including the ancilla
                    self.ops_class.n_q = self.n_q+1         
                else:   
                    # exclude the ancilla
                    self.ops_class.n_q = self.n_q
    def pauli_term_to_control_gates(self, p_term:observable_aqc):
        """ Return a program with controlled pauli operators to represent the observables measured by the Had test   
                type:  Program      eg CZ(0,ancilla) + CZ(1,ancilla)
        Return also the weight of the observable. p_term.coefficient
        
        param p_term, the Pauli operators that represent the observable eg p_term=2*sZ(0)*sZ(1) p_term.coefficient =2
                type: Pauli Term 
        param self.edge_qubits
        """
        # label the edge qubits
        #p = self.ops_class.Program_AQC() 
        self.edge_qubits = []
        self.offset = 0
        if p_term.coefficient !=0:
                if p_term.is_identity_op():
                        self.offset = 1             # used to ensure the Ev_ancilla() returns 1*p_term.coefficient
                else:
                        for idx, op in enumerate(p_term.ops): 
                                # Convert full circuit qubit number, to a new number in the sub_circuit.
                                
                                q_sub_cir = self.sub_cir(p_term.target[idx])
                                if op.lower()=='z' and self.no_had_gates==0:
                                        self.ops_class.CZ_AQC(q_sub_cir,self.ancilla_num)
                                self.edge_qubits.append(p_term.target[idx])
        return p_term.coefficient        