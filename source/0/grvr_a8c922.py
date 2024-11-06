# https://github.com/decoyindus/qulib-Grover/blob/96d3f4acfc701fd88e217d8dd2f2381804aa8dbe/grvr.py
import flask
from flask import request, jsonify, send_from_directory, make_response, send_file, redirect
import numpy as np
# importing Qiskit
from qiskit import BasicAer, IBMQ
from qiskit import QuantumCircuit, assemble, execute,ClassicalRegister,QuantumRegister
# import basic plot tools
import io
import json
import base64
from qiskit.circuit import qpy_serialization
from qiskit.aqua.components.oracles import TruthTableOracle, LogicalExpressionOracle
from qiskit.aqua.algorithms import Grover
from qiskit.aqua import QuantumInstance
import operator
from qiskit.quantum_info import Operator
from flask_swagger_ui import get_swaggerui_blueprint
# from routes import request_api

app = flask.Flask(__name__)
app.config['DEBUG'] = True


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('/static', path)


swagger_url = '/home'
API_url = '/static/grvr_api.json'
swagger_ui_blueprint = get_swaggerui_blueprint(swagger_url,API_url,config={'app_name':'QuLib'})
app.register_blueprint(swagger_ui_blueprint, url_prefix=swagger_url)



@app.errorhandler(500)
def handle_500_error(_error):
    """Return a http 500 error to client"""
    return make_response(jsonify({'error': 'Server error'}), 500)

@app.route('/',methods=['GET'])
def idx():
    return redirect('/home')


@app.route('/demo/get_grover_circuit',methods=['GET'])

def get_circuit():

    if 'qubits' in request.args:
        n = int(request.args['qubits'])

    else:
        return jsonify({'ERROR': 'Cannot specify the number of qubits for circuit.'})
    if 'good states' in request.args:
        W = request.args.getlist('good states')
        print(W)

    else:
        return jsonify({'ERROR': 'Cannot specify the winning states.'})

    def phase(nqbits,G):
        name = 'Uf'
        circuit = QuantumCircuit(nqbits, name=name)
        mtx = np.identity(pow(2,nqbits))
        for g in G:
            mtx[int(g)][int(g)] = -1
        circuit.unitary(Operator(mtx),range(nqbits))
        return circuit

    def diffuse(nqbits):
        circuit = QuantumCircuit(nqbits,name="V")
        circuit.h(range(nqbits))
        circuit.append(phase(nqbits,[0]), range(nqbits))
        circuit.h(range(nqbits))
        return circuit

    orcle = QuantumCircuit(n, n)
    iterations = int(np.floor((np.pi/4)*np.sqrt(pow(2,n)/len(W))))
    orcle.h(range(n))
    for _ in range(iterations):
        orcle.append(phase(n,W), range(n))
        orcle.append(diffuse(n),range(n))
    orcle.measure(range(n), range(n))
    buf = io.BytesIO()
    qpy_serialization.dump(orcle, buf)
    orcle.draw(output='mpl').savefig('grvr_img.png')
    response = send_file('grvr_img.png', mimetype='image/png')
    response.headers['circuit'] = base64.b64encode(buf.getvalue()).decode('utf8')
    # json_str = json.dumps({
    #     'circuit': base64.b64encode(buf.getvalue()).decode('utf8'),
    # })
    return response

@app.route('/Grover/bitmap',methods=['GET'])

def grover_bitmap():
    if 'bitmap' in request.args:
        bmp = request.args['bitmap']
    else:
        return jsonify({'ERROR':'bitmap not provided.'})
    if 'API_key' in request.args:
        key = request.args['API_key']
    else:
        return jsonify({'ERROR': 'API Key not provided.'})
    if 'good states' in request.args:
        num_good_states = int(request.args['good states'])
    else:
        return jsonify({'ERROR': 'Number of solutions not provided.'})

    orcle = TruthTableOracle(bmp)
    gr = Grover(orcle)
    msr = gr.run(QuantumInstance(BasicAer.get_backend('qasm_simulator')))
    print(msr['measurement'])
    counts = sorted(msr['measurement'].items(),key=lambda x:x[1],reverse=True)
    results = [int(item[0], 2) for item in counts[:num_good_states]]
    return jsonify(results)

@app.route('/Grover/boolean',methods=['GET'])

def grover_boolexpr():
    if 'expr' in request.args:
        bool_expression = request.args['expr']
    else:
        return jsonify({'ERROR':'boolean expression not provided.'})
    if 'API_key' in request.args:
        key = request.args['API_key']
    else:
        return jsonify({'ERROR': 'API Key not provided.'})

    orcle = LogicalExpressionOracle(bool_expression)
    gr = Grover(orcle)
    msr = gr.run(QuantumInstance(BasicAer.get_backend('qasm_simulator')))
    print(msr['measurement'])
    res = max(msr['measurement'].items(), key=lambda x: x[1])[0]
    print(res)
    return jsonify(res[::-1])



if __name__ == '__main__':
    app.run()
