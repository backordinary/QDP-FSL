# https://github.com/AbhinavAkkiraju/QuantumLogistics/blob/0bbc2e6196c2ba879ad3a3faddc6d23e34866c68/app.py
import re
from geographiclib.geodesic import Geodesic
from global_land_mask import globe
from numpy import float64
from qiskit import IBMQ
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
from geopy.geocoders import Nominatim
import flask
from flask import request, jsonify
import math
from flask_cors import CORS, cross_origin
IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
gcomp = provider.get_backend('ibm_washington')
qins = QuantumInstance(backend=gcomp, shots=200)
meo = MinimumEigenOptimizer(QAOA(COBYLA(maxiter=100), quantum_instance=qins))

app = flask.Flask(__name__)
geolocator = Nominatim(user_agent='app')
cors = CORS(app, resources={"/": {"origins": "*"}})

lat1 = 0
long1 = -122.205335

lat2 = 47.0
long2 = -122.618962

ton = 1
ep = 33
cp = 33
sp = 33

@app.route('/', methods=['GET'])
@cross_origin()
def getInfo():
    global lat1
    global long1
    global lat2
    global long2
    global ton 
    global ep
    global cp
    global sp
    if 'city1' in request.args:
        lat1 = geolocator.geocode((request.args['city1'])).latitude
        long1 = geolocator.geocode((request.args['city1'])).longitude
        print(lat1, long1)
    else:
        return "Error"
    if 'city2' in request.args:
        lat2 = geolocator.geocode((request.args['city2'])).latitude
        long2 = geolocator.geocode((request.args['city2'])).longitude
        print(lat2, long2)
    else:
        return "Error"
    if 'ton' in request.args:
        ton = float64(request.args['ton'])
    else:
        return "Error"
    if 'ep' in request.args:
        ep = float64(request.args['ep'])
    else:
        return "Error"
    if 'cp' in request.args:
        cp = float64(request.args['cp'])
    else:
        return "Error"
    if 'sp' in request.args:
        sp = float64(request.args['sp'])
    else:
        return "Error"
    
    geod = Geodesic.WGS84
    l = geod.InverseLine(lat1, long1, lat2, long2)
    ds = 1000
    n = int(math.ceil(l.s13 / ds))
    segments = []
    curSeg = []
    curSegState = False

    kmWater = 0
    kmLand = 0

    for i in range(n + 1):
        s = min(ds * i, l.s13)
        g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        if i == 0:
            curSegState = globe.is_land(g['lat2'], g['lon2'])
            curSeg = [g['s12'], g['s12']]
        if curSegState != globe.is_land(g['lat2'], g['lon2']):
            curSeg.append(curSegState)
            segments.append([(curSeg[1] - curSeg[0]) / 1000, curSeg[2]])
            curSegState = globe.is_land(g['lat2'], g['lon2'])
            curSeg = [g['s12'], g['s12']]
        elif curSeg[1] < g['s12']:
            curSeg[1] = g['s12']
        if i == n and len(segments) == 0:
            segments.append([curSeg[1] / 1000, True])
    for i in segments:
        if i[1]:
            kmLand += i[0]
        elif not i[1]:
            kmWater += i[0]

    qp = QuadraticProgram()

    m = (ep * (0.16 * kmLand * ton) + cp * (22 * kmLand * ton) * (1 / (sp * 15)))
    w = kmWater
    k = (ep * (0.18 * kmLand * ton) + cp * (62 * kmLand * ton) * (1 / (sp * 36)))
    air = abs(kmLand - k - m + w)

    KmMUpperBound = int(kmLand)
    KmAUpperBound = int(kmLand + kmWater)
    KmKUpperBound = int(kmLand)
    KmBUpperBound = int(kmWater)
    if ton > 400000:
        KmBUpperBound = 0
    if ton > 36:
        KmKUpperBound = 0
    if kmLand < 280 or ton > 12500:
        KmMUpperBound = 0
    if kmLand + kmWater < 280 or ton > 112:
        KmAUpperBound = 0
    print("Land: " + str(kmLand), "Water: " + str(kmWater))
    # KmK = Truck
    # KmM = Train
    # KmB = boat
    # KmA = KmA
    qp.integer_var(name="KmK", lowerbound = 0, upperbound = KmKUpperBound)
    qp.integer_var(name="KmM", lowerbound = 0, upperbound = KmMUpperBound)
    qp.integer_var(name="KmB", upperbound = KmBUpperBound)
    qp.integer_var(name="KmA", upperbound = KmAUpperBound)
    qp.linear_constraint({"KmK": 1, "KmM": 1}, "<=", int(kmLand))
    qp.linear_constraint({"KmK": 1, "KmM": 1, "KmB": 1, "KmA": 1}, "=", int(kmLand + kmWater))

    print(w, k, m, air)
    if kmWater > 0:
        qp.minimize(linear={"KmB": w, "KmK": k, "KmM": m, "KmA": air})
    else:
        qp.minimize(linear={"KmK": k, "KmM": m, "KmA": (kmLand - k - m + w)})
    solution = meo.solve(qp)
    result = re.search('\[(.*)\]', str(solution)).group(1).replace(" ", "")[:-1].split(".")
    for i in range(0, len(result)):
        result[i] = int(result[i])
    print(result)

    time = 0
    carbonFootPrint = 0
    price = 0
    if len(result) == 4:
        carbonFootPrint += 62 * result[0] * ton
        carbonFootPrint += 8 * result[2] * ton
        carbonFootPrint += 22 * result[1] * ton
        carbonFootPrint += 602 * result[3] * ton

        price += 0.18 * result[0] * ton
        price += 0.29 * result[2] * ton
        price += 0.16 * result[1] * ton
        price += 0.98 * result[3] * ton

        conversion_factor = 0.62137119

        time += (result[0] / 36) / conversion_factor
        time += (result[2] / 21) / conversion_factor
        time += (result[1] / 15) / conversion_factor
        time += (result[3] / 575) / conversion_factor
        return jsonify({"truck": result[0], "train": result[1], "boat": result[2], "aircraft": result[3], "co2": carbonFootPrint, "price": price, "time": time})
    else:
        carbonFootPrint += 62 * result[0] * ton
        carbonFootPrint += 22 * result[1] * ton
        carbonFootPrint += 602 * result[2] * ton

        price += 0.18 * result[0] * ton
        price += 0.16 * result[1] * ton
        price += 0.98 * result[2] * ton

        conversion_factor = 0.62137119
        time += (result[0] / 36) / conversion_factor
        time += (result[1] / 15) / conversion_factor
        time += (result[2] / 575) / conversion_factor
        return jsonify({"truck": result[0], "train": result[1], "boat": 0, "aircraft": result[2], "co2": carbonFootPrint, "price": price, "time": time})
app.run(host='192.168.1.23', port=80)