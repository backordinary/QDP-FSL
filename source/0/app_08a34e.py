# https://github.com/Trufelek124/projekt_ppp/blob/b28123e980af925bca46dd8bdbabad67a75a8163/app.py
from flask import Flask, jsonify
from flask import render_template, request, redirect, url_for, flash
import time
import quantum_settings_service as qss
import ibmq_quantum_service as iqs
from qiskit import *
import json

app = Flask("Projekt Domański")

@app.route('/api/getAvailableBackends', methods=['GET'])
def get_available_backends():
    available_backends = qss.return_available_backends_and_simulator()
    data = {"available_backends" : available_backends};
    return jsonify(data), 200

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify('Jestem'),  200

@app.route('/api/getBackendProperties/<backend_name>', methods=['GET'])
def run_quantum_program(backend_name):
    properties = qss.get_backend_properties(backend_name)
    data = {"properties": properties}
    return jsonify(data), 200

@app.route('/api/runQuantumProgram', methods=['POST'])
def get_backend_properties():
    request_body = request.get_json()
    backend_name = request_body["backend_name"]
    num_of_shots = request_body["num_of_shots"]
    backend = qss.get_backend(backend_name)
    result = iqs.run_program_on_backend(backend, num_of_shots)
    data = {"results": result}
    return json.dumps(data), 200

def setup_quantum_connection():
    IBMQ.save_account("2990848b1062e426869449f092ae9d5295b38df83c7eeab0e100538c6ae107c6124e91b5063430ef6d9532fe5e69ebc882306c0e8e8fa34de8cb046b21dcd48b")
    IBMQ.load_account()

app.secret_key = 'super secret key'
app.config.from_object(__name__)
app.debug = True
setup_quantum_connection()
app.run()