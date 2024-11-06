# https://github.com/TimVroomans/Quantum-Mastermind/blob/1c3ae129f2d6d7272320593bf67b233971397a7c/src/mastermind/arithmetic/increm.py
"""
Created on Wed Dec  2 18:25:35 2020

@author: Giel Coemans
"""
from math import pi
from qiskit import *
from qiskit.circuit.library.standard_gates import PhaseGate
from mastermind.arithmetic.qft import qft, iqft


def increment(circuit, q, do_qft=True, amount=1):
    """Performs +1 on the value of register q

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to be appended with increment.
    q : QuantumRegister
        Register to be incremented.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*a).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with increment.
    
    """
    
    # Constants
    n = len(q)
    
    # Optional QFT
    if do_qft:
        qft(circuit, q)
    
    # Actual increm core gates
    for (i,qubit) in enumerate(q):
        circuit.rz(amount*pi/2**(n-1-i), qubit)  
    
    # Optional iQFT
    if do_qft:
        iqft(circuit, q)
    
    return circuit


def cnincrement(circuit, c, q, do_qft=True, amount=1):
    """Performs +1 on the value of register q, controlled by register c

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to be appended with increment.
    q : QuantumRegister
        Register to be incremented.
    c : Qubit List
        Control qubits
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*a).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with increment.
    
    """
    
    # Constants
    n = len(q)
    nc = len(c)
    
    # Optional QFT
    if do_qft:
        circuit.barrier()
        qft(circuit, q)
        circuit.barrier()
    
    # Actual core (controlled) increm gates
    for (i,qubit) in enumerate(q):
        # qcs = QuantumCircuit(1)
        # qcs.rz(amount*pi/2**(n-i-1),0)
        # ncrz = qcs.to_gate().control(nc)
        # circuit.append(ncrz, [*c, qubit])
        ncp = PhaseGate(amount*pi/2**(n-i-1)).control(nc)
        circuit.append(ncp, [*c, qubit])
        
    
    # Optional iQFT
    if do_qft:
        circuit.barrier()
        iqft(circuit, q)
        circuit.barrier()
    
    return circuit


def decrement(circuit, q, do_qft=True, amount=1):
    """
    See: increment
    
    """
    
    # Just increment with flipped sign
    increment(circuit, q, do_qft, -amount)
    
    return circuit


def cndecrement(circuit, c, q, do_qft=True, amount=1):
    """
    See: cnincrement
    
    """
    
    # Just cnincrement with flipped sign
    cnincrement(circuit, c, q, do_qft, -amount)
    
    return circuit