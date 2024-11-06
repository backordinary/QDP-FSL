# https://github.com/Caballero-Software-Inc/qportfolio/blob/6a25d73de8473beb6708dda26beb6ead19f6d28f/app.py
from flask import Flask, render_template, request, jsonify
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import base64
from qiskit import(
    execute,
    QuantumCircuit,
    IBMQ)
from qiskit.tools.monitor import job_monitor
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.compiler import assemble
import os
import time
import random


def quantumBackend(circuit, token, nshots):
    IBMQ.save_account(token, overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    qcomp = provider.get_backend('ibmq_lima') # 5 qubits, 8 quantum volume
    job = execute(circuit, backend=qcomp, shots = nshots)
    job_monitor(job)
    return job.result()

def simulationBackend(circuit, token, nshots):
    simulator = Aer.get_backend('qasm_simulator')
    circuit = transpile(circuit, simulator)
    my_qobj = assemble(circuit) 
    return simulator.run(my_qobj, shots = nshots).result()

optionsBackend =  {
    'simulation': simulationBackend,
    'quantum': quantumBackend
}

def getCircuitPicture(circuit):
    name = "circ"+str(time.time())+str(random.random())+".png"
    circuit.draw( output='mpl', filename = name )
    s = str(base64.b64encode(open(name, "rb").read()))
    os.remove(name)
    return s

def deutschBackend(input):
    f = input['f']
    #quantum circuit
    #before oracle
    circuit = QuantumCircuit(2, 2)
    circuit.x(1)
    circuit.h(0)
    circuit.h(1)

    #oracle
    #if f[0] and f[1]:
    #    do nothing

    if f[0] and f[1]:
        circuit.x(1)

    if f[0] and (not f[1]):
        circuit.cx(0,1)
    
    if (not f[0]) and f[1]:
        circuit.x(1)
        circuit.cx(0,1)

    #after oracle  
    circuit.h(0)
    circuit.measure([0, 1], [0, 1])

    #measurement
    result = optionsBackend[input['backend']](circuit, input['token'], 20)

    # Returns counts
    counts = result.get_counts(circuit)
    keys = list(counts)
    max_val = 0
    max_index = 0

    
    # after measurement
    for j in range(len(keys)):
        if counts[ keys[j] ] > max_val:
            max_val = counts[ keys[j] ]
            max_index = j

    return json.dumps({'const': (keys[max_index][1] == '0'), 'circ': getCircuitPicture(circuit) })


def randomBitBackend(input):
    #quantum circuit
    #before oracle
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    
    #after oracle  
    circuit.measure([0], [0])

    #measurement
    result = optionsBackend[input['backend']](circuit, input['token'], 1)

    # Returns counts
    counts = result.get_counts(circuit)
    keys = list(counts)
    
    # after measurement
    
    return json.dumps({'bit': (keys[0]=='1'), 'circ': getCircuitPicture(circuit) })





# client-server
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

options =  {
    'Deutsch': deutschBackend,
    'randomBit': randomBitBackend
}

@app.route('/api', methods=['POST'])
def main():
    return options[request.get_json()['selection']](request.get_json()['input'])





if __name__ == '__main__':
    app.run(debug=True)

