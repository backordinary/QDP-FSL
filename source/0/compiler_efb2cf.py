# https://github.com/SagarDollin/QOSF-Mentorship-Prgram---Task-3-solution/blob/c006a14d63126b1c5bd7ebbffa88f25aed9101b5/Compile/Compile/Compiler.py
from qiskit import QuantumCircuit
import sys
from math import pi

def compile_h(qc,qubit):
    qc.rz(pi/2,qubit)
    qc.rx(pi/2,qubit)
    qc.rz(pi/2,qubit)
    
def compile_x(qc,qubit):
    qc.rx(pi,qubit)

    
def compile_y(qc,qubit):
    qc.rz(pi/2,qubit)
    compile_x(qc,qubit)
    qc.rz(-pi/2,qubit)
    
def compile_z(qc,qubit):
    qc.rz(pi,qubit)
    
def compile_i(qc,qubit):
    compile_x(qc, qubit)
    compile_x(qc, qubit)

    
    
def compile_ry(qc,theta,qubit):
    qc.rz(-pi/2,qubit)
    qc.rx(-theta,qubit)
    qc.rz(pi/2,qubit)
    
def compile_cnot(qc,c,t):
    compile_h(qc,t)
    qc.cz(c,t)
    compile_h(qc,t)

def compiler_override():
    module = sys.modules['qiskit']
    module.QuantumCircuit.h = compile_h
    module.QuantumCircuit.x = compile_x
    module.QuantumCircuit.y = compile_y
    module.QuantumCircuit.z = compile_z
    module.QuantumCircuit.ry = compile_ry
    module.QuantumCircuit.cnot = compile_cnot
    module.QuantumCircuit.i = compile_i
    sys.modules['qiskit'] = module
    
