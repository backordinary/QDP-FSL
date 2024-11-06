# https://github.com/decoyindus/qulib-BVZR/blob/3c891916b05793791fb5ffb080be70053aacad23/b_vazir.py
import flask
from flask import request,jsonify,send_from_directory,send_file, redirect
import numpy as np
# importing Qiskit
from qiskit import BasicAer, IBMQ
from qiskit import QuantumCircuit, assemble, execute,ClassicalRegister
# import basic plot tools
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from PIL import Image
import io
import json
import base64
from qiskit.circuit import qpy_serialization
from qiskit.aqua.components.oracles import TruthTableOracle
import operator
from flask_swagger_ui import get_swaggerui_blueprint

app = flask.Flask(__name__)
app.config['DEBUG'] = True

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('/static', path)


swagger_url = '/home'
API_url = '/static/bv_api_new.json'
swagger_ui_blueprint = get_swaggerui_blueprint(swagger_url,API_url,config={'app_name':'QuLib'})
app.register_blueprint(swagger_ui_blueprint, url_prefix=swagger_url)

@app.route('/',methods=['GET'])
def index():
    return redirect('/home')

@app.route('/demo/get_BV_oracle',methods=['GET'])
def build_BV_oracle():
    if 'key' in request.args:
        key = request.args['key']
        n = len(key)
    elif 'qubits' in request.args:
        n = int(request.args['qubits'])
        key = np.random.randint(0, pow(2, n) - 1)
        key = format(key, '0' + str(n) + 'b')
    else:
        return jsonify({'ERROR': 'Cannot specify key bitstring'})
    oracle = QuantumCircuit(n + 1, n)
    for i in range(n):
        oracle.h(i)
    oracle.x(n)
    oracle.h(n)
    oracle.barrier()
    for i, v in enumerate(key):
        if v == '1':
            oracle.cx(i, n)
    oracle.barrier()
    for i in range(n):
        oracle.h(i)
        oracle.measure(i, i)
    buf = io.BytesIO()
    qpy_serialization.dump(oracle, buf)
    oracle.draw(output='mpl').savefig('circuit_img.png')
    response = send_file('circuit_img.png',mimetype='image/png')
    response.headers['oracle']=base64.b64encode(buf.getvalue()).decode('utf8')
    response.headers['key'] = key
    # pil_img = Image.open('circuit_img.png',mode='r')
    # byte_arr = io.BytesIO()
    # pil_img.save(byte_arr,format='PNG')
    # enc_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    # json_str = json.dumps({
    #     'oracle': base64.b64encode(buf.getvalue()).decode('utf8'),
    #     'key': key,
    #     'img': send_file('circuit_img.png',mimetype='image/gif')
    # })
    return response


@app.route('/demo/get_BV_key',methods=['GET'])
def get_key_():
    if 'oracle' in request.args:
        circuit_json = request.args['oracle']
        qpy_file = io.BytesIO(base64.b64decode(circuit_json))
        circuit = qpy_serialization.load(qpy_file)[0]
    else:
        return jsonify({'ERROR': 'No Oracle circuit found.'})
    simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1, memory=True)
    result = job.result()
    measurement = result.get_memory()[0]
    measurement = measurement[::-1]
    return jsonify({'key': measurement})


@app.route('/BVazirani',methods=['GET'])
def apply_bv():
    if 'bitmap' in request.args:
        bitmap = request.args['bitmap']
        n = int(np.log2(len(bitmap)))
        if len(bitmap) != pow(2, n):
            return jsonify({'ERROR': 'bitmap length should be in powers of 2.'})
    else:
        return jsonify({'ERROR': 'Please provide bitmap.'})
    if 'api_key' in request.args:
        API_KEY = request.args['api_key']
    else:
        return jsonify({'ERROR': 'No IBM-Q API key found.'})

    oracle = TruthTableOracle(bitmap, optimization=True, mct_mode='noancilla')
    superpos = QuantumCircuit(oracle.variable_register, oracle.output_register)
    superpos.h(oracle.variable_register)
    superpos.x(oracle.output_register)
    superpos.h(oracle.output_register)
    circuit = oracle.construct_circuit()
    circuit = circuit.compose(superpos, front=True)
    desup = QuantumCircuit(oracle.variable_register, oracle.output_register)
    desup.h(oracle.variable_register)
    circuit = circuit.compose(desup)
    msr = ClassicalRegister(oracle.variable_register.size)
    circuit.add_register(msr)
    circuit.measure(oracle.variable_register, msr)

    IBMQ.enable_account(API_KEY)
    provider = IBMQ.get_provider('ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n + 1) and
                                                             not x.configuration().simulator and
                                                             x.status().operational == True))
    # backend = provider.get_backend('ibmq_5_yorktown')
    #     simulator = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1024)
    job_monitor(job)
    result = job.result()
    noisy_keys = result.get_counts()
    key = max(noisy_keys.items(), key=operator.itemgetter(1))[0]
    IBMQ.disable_account()
    return jsonify({'key': key})

if __name__ == '__main__':
    app.run()
