# https://github.com/Mirkesx/quantum_programming/blob/17866eb358771172052b05931fa5adae29589946/Exercises/exercises.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:38:49 2021

@author: mc
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator
import quantum_circuits as qcl


def cx(b1,b2):
    inp = QuantumRegister(1, "inp")
    out = QuantumRegister(1, "out")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(inp, out, cr)
    qc.barrier()
    if b1 == 1:
        qc.x(inp)
    if b2 == 1:
        qc.x(out)
    qc.barrier()
    qc.cx(inp, out)
    qc.measure(out, cr)
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    return list(counts.keys())[0]

def adder(x, y):
    n = len(x)
    m = n*2 - 1
    if len(y) is not n:
        return "You must provide two bit-strings with the same length"
    xr = QuantumRegister(n, "x")
    yr = QuantumRegister(n, "y")
    out = QuantumRegister(m, "output")
    cr = ClassicalRegister(m, "cr")
    qc = QuantumCircuit(xr, yr, out, cr)
    qc.barrier()
    for i in range(n):
        if x[i] == '1':
            qc.x(xr[i])
        if y[i] == '1':
            qc.x(yr[i])
    for i in range(n):
        qc.barrier()
        qc.cx(xr[i], out[i])
        qc.cx(yr[i], out[i])
        qc.ccx(xr[i], yr[i], out[i+1])
    for i in range(n):
        qc.barrier()
        qc.cx(xr[i], out[i])
        qc.cx(yr[i], out[i])
        qc.ccx(xr[i], yr[i], out[i+1])
    qc.barrier()
    qc.measure(out, cr)
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    return list(counts.keys())[0]


'''
ESERCIZIO: Controlled Not simulato
Sapreste simulare il gate CX (Controlled Not) utilizzando qualsiasi altro gate, escludendo il gate X e tutte 
le sue versioni controllate (CX, CCX, GCX)?
Se riuscite nella simulazione, provate anche a dare una dimostrazione formale, utilizzando l'algebra lineare, 
che questa simulazioni sia corretta.
'''
def simcx(b1,b2):
    inp = QuantumRegister(1, "inp")
    out = QuantumRegister(1, "out")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(inp, out, cr)
    qc.barrier()
    if b1 == 1:
        qc.h(inp)
        qc.z(inp)
        qc.h(inp)
    if b2 == 1:
        qc.h(out)
        qc.z(out)
        qc.h(out)
    qc.barrier()
    qc.h(out)
    qc.cz(inp,out)
    qc.h(out)
    qc.measure(out, cr)
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    return list(counts.keys())[0]


'''
ESERCIZIO: Si consideri la funzione Booleana PowMul. La funzione prende in input due stringe binarie x e y, 
entrambe di lunghezza n, le quali sono la rappresentazione binaria di potenze esatte di 2 il cui valore è quindi 
compreso tra 1 e 2^(n-1). La funzione restituisce una stringa di lunghezza 2n-1, rappresentazione binaria della 
moltiplicazione di x per y.
Siete in grado di fornire un circuito quantistico in grado di implementare la funzione PowMul?

ESEMPI:
PowMul("0100","0010") = "0001000" (2^2 x 2^1 = 2^3)
PowMul("1000","1000") = "1000000" (2^3 x 2^3 = 2^6)
'''
def is_not_pow_of_2(x):
    list_of_numbers = [int(bit) for bit in x]
    return sum(list_of_numbers) > 1

def powmul(x, y):
    n = len(x)
    m = n*2 - 1
    if len(y) is not n:
        return "You must provide two bit-strings with the same length"
    if is_not_pow_of_2(x):
        return "The first parameter must be an exact power of 2."
    if is_not_pow_of_2(y):
        return "The second parameter must be an exact power of 2."
    xr = QuantumRegister(n, "x")
    yr = QuantumRegister(n, "y")
    anc = QuantumRegister(1, "ancilla_swap")
    out = QuantumRegister(m, "output")
    cr = ClassicalRegister(m, "cr")
    qc = QuantumCircuit(xr, yr, anc, out, cr)
    qc.barrier()
    for i in range(n):
        if x[i] == '1':
            qc.x(xr[i])
        if y[i] == '1':
            qc.x(yr[i])
    qc.x(anc)
    qc.barrier()
    for i in range(n):
        qc.cx(xr[i],out[i])
    for i in range(n):
        qc.barrier()
        qc.cx(yr[i],anc)
        for j in range(m-1, i, -1):
            qc.cswap(anc,out[j-1],out[j])
    qc.barrier()
    qc.measure(out, cr)
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    shots = 1
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    new_counts = qcl.reformat_counts(counts, shots)
    return new_counts






'''
ESERCIZIO: Si consideri la funzione Booleana IntDiv. La funzione prende in input due stringe binarie x e y, di 
lunghezza n e m, rispettivamente. La funzione restituisce una stringa z, di lunghezza n, la quale rappresenta la 
divisione intera di x per 2^k, dove k=set(y) è il numero di 1 presenti nella stringa y. Più formalmente 
IntDiv(x,y) = floor(x/2^set(y)) 

Siete in grado di fornire un circuito quantistico in grado di implementare la funzione IntDiv?
 

ESEMPI:
IntDiv("1011","001") = "0101" (floor(11/2) = 5)
IntDiv("1101","101") = "0011" (floor(13/4) = 3)
'''


def intdiv(x, y):
    x = x[::-1]
    y = y[::-1]
    n = len(x)
    m = len(y)
    xr = QuantumRegister(n, "x")
    yr = QuantumRegister(m, "y")
    out = QuantumRegister(n, "output")
    anc = QuantumRegister(m, "anc")
    cr = ClassicalRegister(n, "cr")
    qc = QuantumCircuit(xr, yr, anc, out, cr)
    qc.barrier()
    for i in range(n):
        if x[i] == '1':
            qc.x(xr[i])
    for j in range(m):
        if y[j] == '1':
            qc.x(yr[j])
    qc.barrier()
    for i in range(n):
        qc.cx(xr[i],out[i])
    for j in range(m):
        qc.barrier()
        for i in range(n-1):
            qc.cswap(yr[j], out[i], out[i+1])
    
        qc.cx(out[n-1], anc[j])
        qc.ccx(anc[j], yr[j], out[n-1])
    
    qc.barrier()
    qc.measure(out, cr)
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    shots = 1
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    new_counts = qcl.reformat_counts(counts, shots)
    return new_counts



'''
ESERCIZIO 4: COMPARE
Si supponga di ricevere in input due numeri interi, x e y, entrambi codificati da sequenze binarie di lunghezza n. 
Ne consegue che x e y sono valori compresi tra 0 e 2^n-1. Si realizzi un circuito che implementi la funzione 
Booleana f:{0,1}^n x {0,1}^n --> {0,1} definita come f(x,y)=1 se e solo se x<y.
Nel caso in cui facciate uso di ancilla bits, la garbage dovrà essere eliminata prima della fine 
della computazione.
'''

def cxorgate():
    circuit = QuantumCircuit(4)
    circuit.ccx(0,2,3)
    circuit.ccx(1,2,3)
    g = circuit.to_gate()
    g.name = "cxor"
    return g


def compare(x, y):
    n = len(x)
    if len(y) is not n:
        return "You must provide two bit-strings with the same length"
    xr = QuantumRegister(n, "x")
    yr = QuantumRegister(n, "y")
    out = QuantumRegister(1, "output")
    anc = QuantumRegister(2, "ancilla")
    fo = QuantumRegister(1, "final-output")
    cr = ClassicalRegister(1, "cr")
    qc = QuantumCircuit(xr, yr, anc, out, fo, cr)
    for i in range(n):
        if x[i] == '1':
            qc.x(xr[i])
        if y[i] == '1':
            qc.x(yr[i])
    qc.barrier()
    qc.x(anc[1])
    for i in range(n):
        qc = qc.compose(cxorgate(),[xr[i], yr[i], anc[1], anc[0]])
        qc.ccx(yr[i],anc[0],out)
        qc.cx(anc[0],anc[1])
    qc.barrier()
    qc.cx(out,fo)
    #qc.measure(out, cr)
    qc.barrier()
    for i in range(n-1,-1,-1):
        qc.cx(anc[0],anc[1])
        qc.ccx(yr[i],anc[0],out)
        qc = qc.compose(cxorgate(),[xr[i], yr[i], anc[1], anc[0]])
    qc.x(anc[1])
    
    qc.measure_all()
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    shots = 1
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    #qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    new_counts = qcl.reformat_counts(counts, shots)
    return new_counts



