# https://github.com/decoyindus/qulib-DeutschJozsa/blob/0877583d2200e7e5cb64e6957370d9a993229a2b/d_josza.py
##########################
import flask
from flask import request, jsonify, send_from_directory, send_file, redirect
import numpy as np
# importing Qiskit
from qiskit.providers.ibmq import least_busy
from qiskit import Aer,BasicAer,IBMQ
from qiskit import QuantumCircuit, execute,ClassicalRegister
# import basic plot tools
import io
import json
import base64
from qiskit.circuit import qpy_serialization
from qiskit.aqua.components.oracles import TruthTableOracle
import operator
from flask_swagger_ui import get_swaggerui_blueprint


app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('/static', path)


swagger_url = '/home'
API_url = '/static/dj_api.json'
swagger_ui_blueprint = get_swaggerui_blueprint(swagger_url,API_url,config={'app_name':'QuLib'})
app.register_blueprint(swagger_ui_blueprint, url_prefix=swagger_url)


@app.route('/', methods=['GET'])
def home():
    return redirect('/home')


@app.route('/demo/get_oracle',methods=['GET'])
def build_oracle():
    if 'n' in request.args:
        n = int(request.args['n'])
    else:
        return "Please specify number of qubits"

    if 'case' in request.args:
        case = str(request.args['case'])
    else:
        return "Please specify type of oracle"
        # if case == None:
        #     case = choice(['b', 'c'])

    oracle = QuantumCircuit(n + 1, n + 1)
    for i in range(0, n):
        oracle.h(i)

    oracle.x(n)
    oracle.h(n)

    oracle.barrier()
    if case == 'b':
        b = np.random.randint(0, pow(2, n) - 1)
        bitmap = format(b, '0' + str(n) + 'b')
        # print(f'Bit String for input: {bitmap}')
        for qbit in range(len(bitmap)):
            if bitmap[qbit] == '1':
                oracle.x(qbit)
        # oracle.barrier()
        for qbit in range(n):
            oracle.cx(qbit, n)
        # oracle.barrier()
        for qbit in range(len(bitmap)):
            if bitmap[qbit] == '1':
                oracle.x(qbit)
    elif case == 'c':
        output = np.random.randint(2)
        case += str(output)
        print(output)
        if output == 0:
            oracle.z(n)
    oracle.barrier()
    for q in range(n + 1):
        oracle.h(q)
        oracle.measure(q, q)

    buf = io.BytesIO()
    qpy_serialization.dump(oracle, buf)
    oracle.draw(output='mpl').savefig('dj_oracle.png')
    response = send_file('dj_oracle.png', mimetype='image/png')
    response.headers['oracle'] = base64.b64encode(buf.getvalue()).decode('utf8')
    # json_str = json.dumps({
    #     'oracle': base64.b64encode(buf.getvalue()).decode('utf8')
    # })

    return response


@app.route('/demo/get_type',methods=['GET'])
def get_type():
    if 'circuit' in request.args:
        # circuit_json = str(json.loads(request.args['circuit']))
        circuit_json = request.args['circuit']
        qpy_file = io.BytesIO(base64.b64decode(circuit_json))
        circuit = qpy_serialization.load(qpy_file)[0]
    else:
        return "Please provide circuit."

    aer_sim = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, aer_sim, shots=1024, memory=True)
    result = job.result()
    measurements = result.get_memory()[0]
    # msr = result.get_memory()
    # c1 = len(list(x for x in msr if x[0] == '0'))
    # c2 = len(list(x for x in msr if x[0] == '1'))
    # print(f'0:{c1},1:{c2}')
    # print(measurements)
    query_state = measurements[-1]
    if query_state == '1':
        # print(aux_state)
        type = 'BALANCED'
    else:  # constant query
        aux_output = measurements[0]
        if aux_output == '1':
            type = 'Constantly 1'
        else:
            type = 'Constantly 0'
    resp = {'type': type}
    return jsonify(resp)

#
# @app.route('/d_josza',methods=['GET'])
# def

@app.route('/d_josza',methods=['GET'])
def D_Josza():
    if 'bitmap' in request.args:
        bitmap = str(request.args['bitmap'])
        n = int(np.log2(len(bitmap)))
        if len(bitmap) != pow(2, n):
            return 'Length of bitmap should be in powers of 2'
    else:
        return 'No Bitmap provided'
    if 'key' in request.args:
        API_KEY = str(request.args['key'])
    else:
        return 'No IBMQ API key found.'
    oracle = TruthTableOracle(bitmap, optimization=True, mct_mode='noancilla')
    pre = QuantumCircuit(oracle.variable_register, oracle.output_register)
    pre.h(oracle.variable_register)
    pre.x(oracle.output_register)
    pre.h(oracle.output_register)
    mid = oracle.construct_circuit()
    post = QuantumCircuit(oracle.variable_register, oracle.output_register)
    post.h(oracle.variable_register)
    circuit = pre+mid+post
    res = ''
    msr = ClassicalRegister(oracle.variable_register.size)
    circuit.add_register(msr)
    # msr2 = ClassicalRegister(reg_out.size)
    # circuit.add_register(msr2)
    circuit.measure(oracle.variable_register, msr)
    # circuit.measure(reg_out,msr2)
    # backend = Aer.get_backend('qasm_simulator')
    # job = execute(circuit, backend, shots=1)
    # result = job.result()
    # m = result.get_counts()

    IBMQ.enable_account(API_KEY)
    provider = IBMQ.get_provider('ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n + 1) and
                                           not x.configuration().simulator and
                                           x.status().operational == True))
    # backend = provider.get_backend('ibmq_5_yorktown')
    job = execute(circuit, backend, shots=1024)
    result = job.result()
    m = result.get_counts()

    # aer_sim = Aer.get_backend('aer_simulator')
    # qobj = assemble(circuit, aer_sim)
    # results = aer_sim.run(qobj).result()
    # m = results.get_counts()

        # print(m)
    top_measurement = max(m.items(), key=operator.itemgetter(1))[0]
    print(top_measurement)
    query_state = int(top_measurement)
    if query_state == 0:
        res = 'constant'
    else: res = 'balanced'
    json_str = {'type': res}
        # buf = io.BytesIO()
        # qpy_serialization.dump(circuit, buf)
        # json_str = json.dumps({
        #     'dj_oracle': base64.b64encode(buf.getvalue()).decode('utf8'),
        #     'qubits': str(n),
        #     'bmp': bitmap
        # })
        # json_str = json.dumps(assemble(circuit).to_dict())
        # resp = {'dj_oracle': json_str}
    IBMQ.disable_account()
    return jsonify(json_str)

if __name__=='__main__':
    app.run()

