# https://github.com/phuongbc20/template-qfaas/blob/56d4d2257a67f3249f2200e6a83976c8ec7f7cf8/template/qiskit/function/handler.py
from qiskit import *

def handle(event, context):
    return {
        "statusCode": 200,
        "body": "Hello from OpenFaaS!"
    }
