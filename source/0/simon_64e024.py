# https://github.com/decoyindus/qulib-simon/blob/4f992df4832a1a4d64f5bf931d58142882f1b53c/simon.py
import math

import flask
from flask import request, jsonify, send_from_directory, send_file, redirect
import numpy as np
# importing Qiskit
from qiskit import BasicAer, IBMQ
from qiskit import QuantumCircuit, assemble, execute,ClassicalRegister,QuantumRegister
# import basic plot tools
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from random import choice
from sympy import Matrix,mod_inverse
import io
import json
import base64
from qiskit.circuit import qpy_serialization
from qiskit.aqua.components.oracles import TruthTableOracle
from flask_swagger_ui import get_swaggerui_blueprint

import operator

app = flask.Flask(__name__)
app.config['DEBUG'] = True


def gaussian_elimination(msr):
    # print(msr)
    lst = [(k[::-1], v) for k, v in msr.items() if k != "0" * len(k)]
    lst.sort(key=lambda x: x[1], reverse=True)
    n = len(lst[0][0])
    eqn = []
    for k, _ in lst:
        eqn.append([int(c) for c in k])
    y = Matrix(eqn)
    yt = y.rref(iszerofunc=lambda x: x % 2 == 0)

    def mod(x, modulus):
        num, den = x.as_numer_denom()
        return num * mod_inverse(den, modulus) % modulus

    y_new = yt[0].applyfunc(lambda x: mod(x, 2))
    rows, _ = y_new.shape
    hidden = [0] * n
    for r in range(rows):
        yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
        if len(yi) == 2:
            hidden[yi[0]] = '1'
            hidden[yi[1]] = '1'
    key = "".join(str(h) for h in hidden)[::-1]
    return key

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('/static', path)


swagger_url = '/home'
API_url = '/static/simon_api.json'
swagger_ui_blueprint = get_swaggerui_blueprint(swagger_url,API_url,config={'app_name':'QuLib'})
app.register_blueprint(swagger_ui_blueprint, url_prefix=swagger_url)

@app.route('/',methods=['GET'])
def indx():
    return redirect('/home')


@app.route('/demo/get_simon_oracle',methods=['GET'])
def get_oracle():
    if 'key' in request.args:
        b = request.args['key']
        n = len(b)
        ones = []
        for i in range(n):
            if b[i] == '1':
                ones.append(i)
    else:
        return jsonify({'ERROR': 'Cannot specify the key bitstring.'})
    qr1 = QuantumRegister(n,'reg1')
    qr2 = QuantumRegister(n,'reg2')
    mr = ClassicalRegister(n)
    orcle = QuantumCircuit(qr1,qr2,mr)

    orcle.h(qr1)
    orcle.barrier()
    orcle.cx(qr1,qr2)
    if ones:
        x = n - ones[-1] - 1
        # x = ones[0]
        for one in ones:
            orcle.cx(qr1[x], qr2[n-one-1])
            # orcle.cx(qr1[x], qr2[one])
    orcle.barrier()

    orcle.measure(qr2,mr)
    orcle.barrier()

    orcle.h(qr1)
    orcle.measure(qr1,mr)
    # orcle = QuantumCircuit(n * 2, n)
    # orcle.h(range(n))
    # orcle.barrier()
    # orcle += simon_oracle(b)
    # orcle.barrier()
    # orcle.h(range(n))
    # orcle.measure(range(n), range(n))
    buf = io.BytesIO()
    qpy_serialization.dump(orcle, buf)
    orcle.draw(output='mpl').savefig('simon_img.png')
    response = send_file('simon_img.png', mimetype='image/png')
    response.headers['oracle'] = base64.b64encode(buf.getvalue()).decode('utf8')
    response.headers['key'] = b
    # json_str = json.dumps({
    #     'oracle': base64.b64encode(buf.getvalue()).decode('utf8'),
    #     'key': b
    # })
    return response


@app.route('/demo/get_simon_key',methods=['GET'])
def getsimonkey():
    if 'circuit' in request.args:
        orcle_json = request.args['circuit']
        qpy_file = io.BytesIO(base64.b64decode(orcle_json))
        orcle = qpy_serialization.load(qpy_file)[0]
    else:
        return jsonify({'ERROR': 'No circuit provided.'})

    simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(orcle, simulator, shots=1024, memory=True)
    result = job.result()
    msr = result.get_counts()
    # print(msr)
    lst = [(k[::-1], v) for k, v in msr.items() if k != "0" * len(k)]
    lst.sort(key=lambda x: x[1], reverse=True)
    n = len(lst[0][0])
    eqn = []
    for k, _ in lst:
        eqn.append([int(c) for c in k])
    y = Matrix(eqn)
    yt = y.rref(iszerofunc=lambda x: x % 2 == 0)

    def mod(x, modulus):
        num, den = x.as_numer_denom()
        return num * mod_inverse(den, modulus) % modulus

    y_new = yt[0].applyfunc(lambda x: mod(x, 2))
    rows, _ = y_new.shape
    hidden = [0] * n
    for r in range(rows):
        yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
        if len(yi) == 2:
            hidden[yi[0]] = '1'
            hidden[yi[1]] = '1'
    key = "".join(str(h) for h in hidden)[::-1]
    return jsonify({'key': key})

@app.route('/Simon',methods=['GET'])
def apply_simon():
    print(request.args.getlist('bitmap'))
    if 'bitmap' in request.args:
        bmp = request.args.getlist('bitmap')
        n = len(bmp[0])
        print(bmp)
        for b in bmp:
            if len(b) != n:
                return jsonify({'ERROR': 'Unequal length of bitmap outputs.'})
    else:
        return jsonify({'ERROR': 'Bitmap not  provided.'})
    if 'key' in request.args:
        API_KEY = request.args['key']
    else:
        return jsonify({'ERROR':'IBM-Q Quantum Experience key not provided.'})
    oracle = TruthTableOracle(bmp, optimization=True, mct_mode='noancilla')
    orcle = oracle.construct_circuit()
    circuit = QuantumCircuit(*orcle.qregs)
    circuit.h(oracle.variable_register)
    circuit.compose(orcle, inplace=True)
    circuit.h(oracle.variable_register)
    msr = ClassicalRegister(oracle.variable_register.size)
    circuit.add_register(msr)
    circuit.measure(oracle.variable_register, msr)

    provider = IBMQ.enable_account(API_KEY)
    # provider = IBMQ.get_provider('ibm-q')
    # backend = least_busy(backends=provider.backends(filters=lambda x: x.configuration().n_qubits >= int(math.log2(n)) and
    #                                        not x.configuration().simulator and
    #                                        x.status().operational is True))
    # print(provider.backends())
    # backend = provider.get_backend('ibmq_lima')


    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1024)
    job_monitor(job)
    result = job.result()
    measurements = result.get_counts()
    key = gaussian_elimination(measurements)
    IBMQ.disable_account()
    return jsonify({'key': key})



if __name__ == '__main__':
    app.run()
