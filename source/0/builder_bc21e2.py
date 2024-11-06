# https://github.com/echo724/qunix/blob/21bf50c2619decdf7cde41b1e062bc5f3147ffa0/qcb_draw/builder.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse import *
from qiskit import QuantumCircuit
from qiskit import execute
from qiskit import BasicAer


qn,cn,prob,state,instruct = argparsing()

qc = QuantumCircuit(qn,cn)

try:
    for i in range(len(instruct)):
        if instruct[i].isalpha():
            if instruct[i] == "h":
                qc.h(int(instruct[i+1]))
            if instruct[i] == "x":
                qc.x(int(instruct[i+1]))
            if instruct[i] == "y":
                qc.y(int(instruct[i+1]))
            if instruct[i] == "z":
                qc.z(int(instruct[i+1]))
            if instruct[i] == "cx":
                qc.cx(int(instruct[i+1]),int(instruct[i+2]))
            if instruct[i] == "rx":
                qc.rx(float(instruct[i+1]),int(instruct[i+2]),int(instruct[i+3]))
            if instruct[i] == "rz":
                qc.rz(float(instruct[i+1]),int(instruct[i+2]),int(instruct[i+3]))
            if instruct[i] == "ry":
                qc.ry(float(instruct[i+1]),int(instruct[i+2]),int(instruct[i+3]))
            if instruct[i] == "crx":
                qc.crx(float(instruct[i+1]),int(instruct[i+2]),int(instruct[i+3]))
            if instruct[i] == "m":
                try:
                    if instruct[i+1] == ".":
                        for j in range(qn):
                            qc.measure(j,j)
                    else:
                        qc.measure(int(instruct[i+1]),int(instruct[i+1]))
                except Exception as e:
                    print("Instruction of measurement is Invalid. Please check the measurement:", e)
        else:
            continue
    print(qc.draw(output='text'))
    result = ""
    try:
        if prob:
            backend = BasicAer.get_backend('qasm_simulator')
            result = execute(qc, backend).result().get_counts()
        elif state:
            backend = BasicAer.get_backend('statevector_simulator')
            result = execute(qc, backend).result().get_statevector()
        else:
            backend = BasicAer.get_backend('qasm_simulator')
            result = execute(qc, backend).result().get_counts()
        print(result)
    except Exception as e:
        print(e)

except:
    print("Invalid instruction arguments. Please Check the arguments")

