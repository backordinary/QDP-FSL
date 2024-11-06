# https://github.com/CS6620-S21/Asynchronous-quantum-job-submission-framework/blob/dad0447e100e0619251312e503f919328c839e99/poc/client.py
from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from qiskit.qobj.qasm_qobj import QasmQobj as QasmQobj

import pickle
import json

import numpy


class QobjEncoder(json.JSONEncoder):
    def default(self, obj):
        print("**************************")
        if isinstance(obj, numpy.int32):
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            return obj.item()
        if isinstance(obj, numpy.ndarray):
            print('++++++++++', obj)
            return obj.tolist()
        if isinstance(obj, complex):
            print('----------------', obj)
            return (obj.real, obj.imag)
        return json.JSONEncoder.default(self, obj)


qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
backend_sim = BasicAer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc, backend_sim)
qasm_obj = assemble(transpiled_qc)

# print(data)
qasm_obj_dic = qasm_obj.to_dict([True])
print(qasm_obj_dic)
# print("qasm_obj_dic: \n", qasm_obj_dic)
# print("type of qasm_obj_dic: ", type(qasm_obj_dic))

# Convert qasm_obj_dic to json for sending over

qasm_obj_json = json.dumps(qasm_obj_dic, cls=QobjEncoder)
print(qasm_obj_dic)
# print("type of qasm_obj_json: ", type(qasm_obj_json))

# Logic to get json serialized and load it and run it.


load = json.loads(qasm_obj_json)

load_qasm = QasmQobj.from_dict(load)
# print("-------------------", load_qasm)

# with open('data.json', 'wb+') as outfile:
#     json.dump(data, outfile)

result = backend_sim.run(load_qasm).result()
print(result)
print("type of Result: ----------", type(result))
print("DATA--------------", result.data(qc))

result_dict = result.to_dict()
print("result_dict: ", result_dict)
print("Tpe of result_dict: ", type(result_dict))

result_json = json.dumps(result_dict, cls=QobjEncoder)
