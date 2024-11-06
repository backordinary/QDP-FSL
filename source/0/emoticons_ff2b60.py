# https://github.com/quantum-italy-developer/qiskit_app_carbon_design/blob/ef83f56fcf9af8437a34044642fc110068920c05/emoticons.py
import os
from flask import Flask, jsonify, request
import json
import requests

# New QISKit libraries
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit import IBMQ
from qiskit import Aer
#from qiskit import BasicAer basic, Python only, implementation called BasicAer in Terra that can be used as a drop-in replacement for Aer
from qiskit.providers.aer import noise
from noisedev import setup_noise,emo_noise

def generateCircuit():
    #collect params
    req_data = request.get_json()
    sim=req_data['set']
    if sim:
        json_output = {"emo":emo_noise(),"shots":400}
        print(json_output)
        #return jsonify(json_output)
        return json_output #jsonify put in main welcome.py file due to errors
    else:
        rawEmoticon = req_data['emo']
        operation = req_data['operation']
        print(rawEmoticon)
        emoticon = []
        # split each emoticon in a list of 1s and 0s
        for string in rawEmoticon:
            print(string)
            print(list(string))
            emoticon.append(list(string))

        print(" ")
        print(emoticon)


        # ======= MICHELE =============
        A = emoticon[0]
        B = emoticon[1]

        qasm = []
        posizione = []

        for i in range(len(A)):
            print(len(A)-i-1, A[i], B[i])

            if A[i] == '0' and B[i] == '0' :
                print("ok 0")
            if A[i] == '1' and B[i] == '1' :
                print("ok 1")
                qasm.append("qc.x(qr["+str(len(A)-i-1)+"])")
                print("QASM", qasm, i)
            if A[i] != B[i] :
                posizione.append(i)
                print("posizione",(len(A)-i-1))
                print("pos, A, B")

        for j in range(len(posizione)):
            if j == 0 :
                qasm.append("qc.h(qr["+str(len(A) - posizione[j] -1)+"])")
            else:
                if A[posizione[0]] == A[posizione[j]] :
                    qasm.append("qc.cx(qr["+str(len(A) - posizione[0] -1)+"],qr["+str(len(A) - posizione[j] -1)+"])")
                else:
                    qasm.append("qc.cx(qr["+str(len(A) - posizione[0] -1)+"],qr["+str(len(A) - posizione[j] -1)+"])")
                    qasm.append("qc.x(qr["+str(len(A) - posizione[j] -1)+"])")

        print("======== qasm ==========")
        print(qasm)
        print("============= ==========")
        # ========= END ===============

        r = sendQuantum(qasm,len(A),sim)
        print("collecting execution results")
        print(r)

        array_output = []
        shots = 0
        for key in r.keys():
            array_output.append({"value": key,"shots": r[key]})
            shots += r[key]

        json_output = {"emo": array_output,"shots":shots}
        print(json_output)
        #return jsonify(json_output)
        return json_output #jsonify put in main welcome.py file due to errors





def sendQuantum(commandList,qubitNr,operation):
    qr = QuantumRegister(qubitNr)
    cr = ClassicalRegister(qubitNr)
    qc = QuantumCircuit(qr, cr)
    for command in commandList:
        print(command)
        # insert check on command received
        exec(command)
    for j in range(qubitNr):
        qc.measure(qr[j], cr[j])
    # check if request is for simulator or real hardware
    #b = "ibmq_qasm_simulator"
    #back=IBMQ.get_backend(b)
    #backend=Aer.backends()[0].name() ## deprecated way to call backand
    backend = Aer.get_backend('qasm_simulator')
    shots_sim = 400
    print("executing algorithm")
    job_exp = execute(qc, backend, shots=shots_sim)
    stats_sim = job_exp.result().get_counts()
    print(stats_sim)
    return stats_sim

    # show available backends
    # IBMQ.backends()
    # # find least busy backend
    # backend = least_busy(IBMQ.backends(simulator=False))
    # print("The least busy backend is " + backend.name())
    # # execute on hardware
    # job_exp = execute(qc, backend=backend, shots=1024, max_credits=3)
    #backend = "ibmq_qasm_simulator"
    #backend = "ibmqx5"
    #shots_sim = 256
    #print("executing algorithm")
    # job_sim = execute(qc, backend, shots=shots_sim)
    #job_exp = execute(qc, backend=IBMQ.get_backend(backend), shots=shots_sim)
    #stats_sim = job_exp.result().get_counts()
    #print(stats_sim)
    #return stats_sim



def executeCircuit():

    #collect params
    commandList =[]
    qubitNr = 0
    print("collecting params")
    req_data = request.get_json()
    commandList = req_data['command']
    qubitNr = req_data['qubitNr']

    # set up registers and program
    qr = QuantumRegister(qubitNr)
    cr = ClassicalRegister(qubitNr)
    qc = QuantumCircuit(qr, cr)

    return json.dumps({"results": sendQuantum(commandList,qubitNr)})
