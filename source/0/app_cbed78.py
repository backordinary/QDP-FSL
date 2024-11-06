# https://github.com/Aemerse/Quanage-Server/blob/bf1dfdee54bc71e7ed5dc7df1b202548613e3fee/app.py
"""Imports"""
from flasgger import Swagger
from flask import Flask
from qiskit import IBMQ
from routes import configure_routes

app = Flask(__name__)
swagger = Swagger(app)
configure_routes(app)
IBMQ.save_account('57fcd139a2ca2e77337f1f5dc2a5a39a2aa0c862981fcb33a2469bc63f47a22a3c8df9f074157f984ee96896ffeb3886f44a137a0b967922f05b91b4ae494a6b')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
