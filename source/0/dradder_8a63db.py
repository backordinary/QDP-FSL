# https://github.com/TimVroomans/Quantum-Mastermind/blob/1c3ae129f2d6d7272320593bf67b233971397a7c/src/mastermind/arithmetic/dradder.py
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:46:05 2020

@author: Giel Coemans
implements positive integer addition and subtraction functions based on a Draper Adder
"""

from math import pi
from qiskit import *
from qiskit.circuit.library.standard_gates import PhaseGate
from mastermind.arithmetic.qft import qft, iqft


def add(circuit, a, b, do_qft=True, amount=1):
    '''
    Adds the value of reg a to that of reg b.

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to perform counting on.
    a : QuantumRegister, length na
        Register with value to be added to register b.
    b : QuantumRegister, length nb (>= na)
        Register to which the value is added.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*a).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with add circuit.
    
    '''
    
    # Constants
    na = len(a)
    nb = len(b)
    if na > nb:
        raise ValueError("Length of reg a cannot be larger than that of reg b for ADD/SUB")
    
    # Optional QFT
    if do_qft:
        circuit.barrier()
        qft(circuit, b)
        circuit.barrier()
    
    # Actual add loop
    for i in range(na):
        for j in range(nb-i):
            circuit.cp(amount*pi/2**(nb-i-j-1), a[i], b[j])
    
    # Optional iQFT
    if do_qft:
        iqft(circuit, b)
        circuit.barrier()
        
    return circuit


def sub(circuit, a, b, do_qft=True, amount=1):
    '''
    Subtracts the value of reg a from that of reg b.

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to perform counting on.
    a : QuantumRegister, length na
        Register with value to be added to register b.
    b : QuantumRegister, length nb (>= na)
        Register to which the value is added.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b-amount*a).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with sub circuit.
    
    '''
    
    # Run ADD circuit, but with negative amount
    add(circuit, a, b, do_qft, -amount)
    
    return circuit


def cadd(circuit, a, b, c, do_qft=True, amount=1):
    '''
    Adds the value of reg a to that of reg b, controlled by reg c.

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to perform counting on.
    a : QuantumRegister, length na
        Register with value to be added to register b.
    b : QuantumRegister, length nb (>= na)
        Register to which the value is added.
    c : QuantumRegister, length nc (>= 1)
        Register which controls whether the addition is performed.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*a).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with cadd circuit.
    
    '''
    
    na = len(a) 
    nb = len(b)
    nc = len(c)
    
    if do_qft:
        circuit.barrier()
        qft(circuit, b)
        circuit.barrier()
    
    # Actual add loop
    
    ### OPTION 1: using standard gates; nicest printing result (als enable import statement at top of file)
    for i in range(na):
        for j in range(nb-i):
            ncp = PhaseGate(amount*pi/2**(nb-i-j-1)).control(nc+1)
            circuit.append(ncp, [*c, a[i], b[j]])
    
    ### OPTION 2: using individual gate circuits
    # for i in range(na):
    #     for j in range(nb-i):
    #         qc = QuantumCircuit(2)
    #         qc.cp(amount*pi/2**(nb-i-j-1), 0, 1)
    #         ccp = qc.to_gate().control(nc)
    #         circuit.append(ccp, [*c, a[i], b[j]])
    
    ### OPTION 3: using circuit defined by numbers
    # qc = QuantumCircuit(na+nb)
    # add(qc, [*range(na)], [*range(na, na+nb)], do_qft=False, amount=amount)
    # ncadd = qc.to_gate().control(nc)
    # circuit.append(ncadd, [*c, *a, *b])
    
    ### OPTION 4: using circuit defined by registers
    # qc = QuantumCircuit(a, b)
    # add(qc, a, b, do_qft=False, amount=amount)
    # ncadd = qc.to_gate().control(nc)
    # circuit.append(ncadd, [*c, *a, *b])
    
    if do_qft:
        circuit.barrier()
        iqft(circuit, b)
        circuit.barrier()
        
    return circuit


def csub(circuit, a, b, c, do_qft=True, amount=1):
    '''
    Subtracts the value of reg a to that from reg b, controlled by reg c.

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to perform counting on.
    a : QuantumRegister, length na
        Register with value to be added to register b.
    b : QuantumRegister, length nb (>= na)
        Register to which the value is added.
    c : QuantumRegister, length nc (>= 1)
        Register which controls whether the addition is performed.
    do_qft : bool (default: True)
        Whether to include the QFT and iQFT on reg b.
    amount : float (default: 1)
        Multiplication factor on addition (i.e. get b+amount*a).

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit appended with cadd circuit.
    
    '''
    
    # Just cadd, but with negative amount
    cadd(circuit, a, b, c, do_qft, -amount)
        
    return circuit