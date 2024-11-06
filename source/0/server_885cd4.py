# https://github.com/anaghasethuraman01/Delivery-Routing-System-using-Quantum-Computing/blob/d597e57680cd8b13b6eb2bba5cd034ecdbd36748/DeliveryRoutingSystem/backend/server.py
import flask
from flask_pymongo import PyMongo
import numpy as np
import json
from bson import json_util
import utils
from flask_cors import CORS
from numpyencoder import NumpyEncoder
from qiskit import IBMQ
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from base64 import encodebytes

app = flask.Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = "mongodb+srv://Project295:drs295@cluster0.slztl.mongodb.net/drs?retryWrites=true&w=majority"
mongo = PyMongo(app)
db = mongo.db

@app.route("/users")
def members():
     user_list = list(db.users.find())
     return json.dumps(user_list, default=json_util.default)
@app.route('/getConnectionDetails',methods=['GET'])
def getConnectionDetails():
     IBMQ.save_account('fbed16e0f3ddc79e1c4375a90c6f6f6c4bc5130f29bc3fe2ccd80d753e4167be53d2eabb76363fb26f6114d898910b9c4068cee2f59ef51c08522d4643f2bd59',overwrite=True)
     
     provider = IBMQ.load_account()
     available_cloud_backends = provider.backends() 
     str = []
     print('\n Cloud backends:')
     for i in available_cloud_backends: 
          print((i.name())) 
          str.append(i.name())
     #  str = available_cloud_backends[0]
     print(str)
     res = {}
     res = {
          "list" : str,
          "status" : "Success"
     }     
     #print(res)
     #return res
     myResponse = flask.make_response('Response')
     myResponse.access_control_allow_origin = 'http://localhost:3000'
     # myResponse.access_control_allow_origin = 'http://localhost:3000/home'
     myResponse.data = json.dumps(res, cls=NumpyEncoder)
     return myResponse
     
     
@app.route('/getRoute/<n>/<k>/<algo>', methods=['GET'])
def getRoute(n, k, algo):
     xc, yc, x_quantum, quantum_cost, nodeMap, qubit_needed = utils.getRoute(int(n),int(k),algo)
     res = {}
     if(algo == 'cplex') :
          res = {
               'xc': xc,
               'yc': yc,
               'nodeMap': nodeMap,
               'classical_cost': quantum_cost
          }
     else :
          print('Printing beatutified array:')
          for ar in x_quantum:
               print(ar)
                              
          res = {
               "xc": xc,
               "yc": yc,
               "x_quantum": x_quantum,
               "quantum_cost": quantum_cost,
               "nodeMap": nodeMap,
               "qubit_needed": qubit_needed
          }
     myResponse = flask.make_response('Response')
     myResponse.access_control_allow_origin = 'http://localhost:3000'
     # myResponse.access_control_allow_origin = 'http://localhost:3000/home'
     myResponse.data = json.dumps(res, cls=NumpyEncoder)
     return myResponse

def node_sequence(arr,n,visited,row,node_sequence):
     for col in range(n):
          if arr[row][col] == 1:
               node_sequence = [i]

     
if __name__ == "__main__":
    app.run(debug=True)
