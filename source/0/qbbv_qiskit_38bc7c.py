# https://github.com/JacobViertel/Biped/blob/02ad195a4f7469e27f418cb2b127d6387a1f30a7/scripts/qiskit/qbbv_qiskit.py
import numpy as np
import math

# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
import os, fnmatch
import sys
import csv
import pandas as pd
import shutil

with open("/home/biped/catkin_ws/src/jacob/scripts/results/face_ident.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            name = row[0]
print(name)
if name == "Jacob_Viertel":
    action = "walking"
else:
    action = "random"

if action == "random":
    qn = 1
    cn = 1
    creg = QuantumRegister(qn, "count")
    clreg = ClassicalRegister(cn, "classical")
    qc = QuantumCircuit(creg, clreg)
    qc.h(0)
    qc.measure(creg, clreg)
    emulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, emulator, shots=1)
    hist = job.result().get_counts()
    for x in hist:
        rand_numb = x
        if x == '0':
            action = "walking"
            print("walking")
        elif x == '1':
            action = "turning"
        else:
            print("review!")

if action == "walking":
    qn = 2
    cn = 2
    creg = QuantumRegister(qn, "count")
    clreg = ClassicalRegister(cn, "calssical")
    qc = QuantumCircuit(creg, clreg)
    qc.h(0)
    qc.h(1)
    qc.measure(creg, clreg)

    emulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, emulator, shots=1 )
    hist = job.result().get_counts()
    for x in hist:
        rand_numb = x
        if x == '00':
            rand_numb = 1
        elif x == '01':
            rand_numb = 2
        elif x == '10':
            rand_numb = 3
        elif x == '11':
            rand_numb = 4
        else:
            print("review!")

    start_motion_file = "/home/biped/catkin_ws/src/jacob/motions/s2w.txt"
    motion_file = "/home/biped/catkin_ws/src/jacob/motions/w2w.txt"
    end_motion_file = "/home/biped/catkin_ws/src/jacob/motions/w2s.txt"
    # start_motion_file = "/home/biped/catkin_ws/src/jacob/motions/stand.txt"
    # motion_file = "/home/biped/catkin_ws/src/jacob/motions/stand.txt"
    # end_motion_file = "/home/biped/catkin_ws/src/jacob/motions/stand.txt"
    data = data2 = "" 

    with open(start_motion_file) as fp: 
            data2 = fp.read()
    data += data2
    data += "\n" 
    with open(motion_file) as fp: 
            data2 = fp.read()
    for i in range(rand_numb):
        data += data2
        data += "\n" 
    with open(end_motion_file) as fp: 
            data2 = fp.read()
    data += data2
    data += "\n" 
    with open ("/home/biped/catkin_ws/src/jacob/motions/combined.txt", "w") as fp: 
        fp.write(data)
    with open ("/home/biped/catkin_ws/src/jacob/scripts/results/status.txt", "w") as fp:
        fp.write("finish")

elif action == "turning":
    qn = 2
    cn = 1
    creg = QuantumRegister(qn, "count")
    clreg = ClassicalRegister(cn, "calssical")
    qc = QuantumCircuit(creg, clreg)
    qc.h(0)
    qc.cx(0,1)
    qc.measure(creg[1], clreg[0])

    emulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, emulator, shots=1 )
    hist = job.result().get_counts()
    lor = '-1'
    for x in hist:
        lor = x
    if lor == '0':
        print("going left")
        motion = 'tl.txt'
        # motion = 'stand.txt'
    elif lor == '1':
        print("going right")
        motion = 'tr.txt'
        # motion = 'stand.txt'
    else:
        print("review!")

    qn = 2
    cn = 2
    creg = QuantumRegister(qn, "count")
    clreg = ClassicalRegister(cn, "calssical")
    qc = QuantumCircuit(creg, clreg)
    qc.h(0)
    qc.h(1)
    qc.measure(creg, clreg)

    emulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, emulator, shots=1 )
    hist = job.result().get_counts()
    for x in hist:
        rand_numb = x
        if x == '00':
            rand_numb = 1
        elif x == '01':
            rand_numb = 2
        elif x == '10':
            rand_numb = 3
        elif x == '11':
            rand_numb = 4
        else:
            print("review!")

    motion_file = "/home/biped/catkin_ws/src/jacob/motions/"+motion
    data = data2 = "" 
    with open(motion_file) as fp: 
            data2 = fp.read()
    for i in range(rand_numb):
        data += data2
        data += "\n" 
    with open ("/home/biped/catkin_ws/src/jacob/motions/combined.txt", "w") as fp: 
        fp.write(data) 