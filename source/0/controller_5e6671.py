# https://github.com/CS6620-S21/Asynchronous-quantum-job-submission-framework/blob/dad0447e100e0619251312e503f919328c839e99/poc/controller.py

from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from qiskit.qobj.qasm_qobj import QasmQobj as QasmQobj

import pickle
import json
import numpy


class Controller:

    def submit(self, qasm_dict={}):
        print("Recieved qasm_dict: ", qasm_dict)

        # json_data = json.dumps(qasm_dict)

        # print("Type after json.dumps: ", type(json_data))

        qasm_ojb = QasmQobj.from_dict(qasm_dict)
        backend_sim = BasicAer.get_backend('qasm_simulator')

        result = backend_sim.run(qasm_ojb).result()
        print("Result: ", result)

        result_dict = result.to_dict()
        print("result_dict: ", result_dict)

        result_json = json.dumps(result_dict, cls=QobjEncoder)
        print("result_json \n: ", result_json)
        return result_json


class QobjEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.int32):
            return obj.item()
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return (obj.real, obj.imag)
        return json.JSONEncoder.default(self, obj)
