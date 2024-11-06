# https://github.com/quantum-italy-developer/qiskit_app_carbon_design/blob/ef83f56fcf9af8437a34044642fc110068920c05/welcome.py
# Copyright 2015 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import flask
import flask_login
import json
import requests

# New QISKit libraries
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit import IBMQ
#from qiskit.backends.ibmq import least_busy

# Visualization libraries
import numpy as np


from spheres import applyOperators


# Custom libraries
from emoticons import generateCircuit, executeCircuit

# Read config.json file
with open('config.json') as f:
    credentials_file = json.load(f)

# QISKit authentication
print('loading credentials')
IBMQ.enable_account(credentials_file['IBMQ']['apikey'])
#IBMQ.load_account() #to check
print('printing backends')
provider = IBMQ.get_provider(hub='ibm-q')
print("========")
provider.backends()
print('finished printing')

app = flask.Flask(__name__)
jsonify = flask.jsonify

# # === Start Login ====
# app.secret_key = 'italy ibm q application pythonQ'
# login_manager = flask_login.LoginManager()
# login_manager.init_app(app)
# # Mock database
# users = {credentials_file['user']['mail']: {'password': credentials_file['user']['password']}}
# class User(flask_login.UserMixin):
#     pass
#
# @login_manager.user_loader
# def user_loader(email):
#     if email not in users:
#         return
#
#     user = User()
#     user.id = email
#     return user
#
# @login_manager.request_loader
# def request_loader(request):
#     email = request.form.get('email')
#     if email not in users:
#         return
#
#     user = User()
#     user.id = email
#
#     # DO NOT ever store passwords in plaintext and always compare password
#     # hashes using constant-time comparison!
#     user.is_authenticated = request.form['password'] == users[email]['password']
#
#     return user
#
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if flask.request.method == 'GET':
#         return app.send_static_file('login.html')
#
#     email = flask.request.form['email']
#     if email in users.keys():
#         if flask.request.form['password'] == users[email]['password']:
#             user = User()
#             user.id = email
#             flask_login.login_user(user)
#             return app.send_static_file('dashboard.html') # Redirect to landing page
#
#     return '''<p>Invalid login, try again</p>'''
#
# @login_manager.unauthorized_handler
# def unauthorized_handler():
#     return app.send_static_file('login.html')
#
# # === End Login ===

# @app.route('/', methods=['GET', 'POST'])
# @flask_login.login_required
# def Welcome():
#     return app.send_static_file('login.html')

@app.route('/', methods=['GET', 'POST'])
# @flask_login.login_required
def Welcome():
    return app.send_static_file('dashboard.html')

@app.route('/dashboard',methods=['GET', 'POST'])
# @flask_login.login_required
def Dashboard():
    return app.send_static_file('dashboard.html')

@app.route('/spheres', methods=['GET', 'POST'])
# @flask_login.login_required
def Spheres():
    return app.send_static_file('spheres.html')

@app.route('/docs', methods=['GET', 'POST'])
# @flask_login.login_required
def Docs():
    return app.send_static_file('docs.html')

@app.route('/team', methods=['GET', 'POST'])
# @flask_login.login_required
def Team():
    return app.send_static_file('team.html')

@app.route('/publications', methods=['GET', 'POST'])
# @flask_login.login_required
def Publications():
    return app.send_static_file('publications.html')

@app.route('/emoticons', methods=['GET', 'POST'])
# @flask_login.login_required
def Emoticons():
    return app.send_static_file('emoticons.html')

@app.route('/generateCircuit', methods=['POST'])
def genCirc():
    return jsonify(generateCircuit())

#@app.route('/executeCircuit', methods=['POST'])
#def exeCirc():

@app.route('/applyOperators', methods=["POST"])
def appOper():
    print('received POST applyOp')
    res = applyOperators()
    return jsonify(res)

@app.route('/visualize3d', methods=['GET','POST'])
def sphere():
    return app.send_static_file('visualize3d.html')

port = os.getenv('PORT', '5000')
if __name__ == "__main__":
    # app.debug = True
    app.run(host='0.0.0.0', port=int(port),threaded=True)
