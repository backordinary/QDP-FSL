# https://github.com/TimVroomans/Quantum-Mastermind/blob/b3c814c35e16f697c0fdb291a6c3a10ed6036a06/src/mastermind/arithmetic/count.py
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:56:56 2020

@author: Giel
"""
from qiskit import *
from mastermind.arithmetic.qft import qft, iqft
from mastermind.arithmetic.increm import cnincrement, cndecrement

def count(circuit, a, b, step=1, do_qft=True, amount=1):
    '''
    Count function for k colours. Takes register a as control qubits. 
    Counts in register b 

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to be appended with counter.
    a : QuantumRegister
        Control register a.
    b : QuantumRegister
        Count register b.
    step : int
        the length of each individual sub-interval in register a.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*|a|).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with counter

    '''
    
    # Constants
    an = len(a)
    bn = len(b)
    
    # QFT
    if do_qft:
        circuit.barrier()
        qft(circuit, b)
        circuit.barrier()
    
    # Core count sub blocks
    for (i,qubit) in enumerate(a[0:an:step]):
        cnincrement(circuit, a[(i*step):(i+1)*step], b, do_qft=False, amount=amount)
        circuit.barrier() if do_qft else None
    
    # iQFT
    if do_qft:
        circuit.barrier()
        iqft(circuit, b)
        circuit.barrier()
    
    return circuit 


def icount(circuit, a, b, step=1, do_qft=True, amount=1):
    '''
    Count function for k colours. Takes register a as control qubits. 
    Counts in register b 

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to be appended with counter.
    a  : QuantumRegister
        Control register a
    b  : QuantumRegister
        Count register b
    step : int
        the length of each individual sub-interval in register a.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*|a|).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with counter

    '''
    
    # Just count with inverted sign
    count(circuit, a, b, step, do_qft, -amount)
    
    return circuit 