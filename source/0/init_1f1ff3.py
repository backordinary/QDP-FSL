# https://github.com/valeria0428/numerico2021/blob/1eae8454bd80c09c9abfe1b589feff7478bd105c/__init__.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from qiskit import (
    IBMQ,
    QuantumCircuit,
    execute,
    ClassicalRegister,
    QuantumRegister
)
import sys
import qiskit
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
from base64 import b64encode
import io

from methods.ejemplo import prueba
from methods.biseccion import biseccion as bise
from methods.raices import raizMultiples as rmul
from methods.reglaFalsa import reglaFalsa as rf
from methods.incrementales import incrementales as incre
from methods.newton import newton as new
from methods.eliminacionSimple import Eliminacion as gaussSimple
from methods.pivoteoParcial import Eliminacion as pivoteoParcial
from methods.pivoteoTotal import Eliminacion as pivoteoTotal
from methods.secante import secante
from methods.biseccion import biseccion
from methods.puntoFijo import punto_fijo as pf
from methods.lagrange import lagrange as lag
from methods.newtonInterpolacion import polinomio_newton as newI
from methods.puntoFijo import punto_fijo
from methods.lagrange import lagrange
from methods.newtonInterpolacion import polinomio_newton
from methods.incrementales import incrementales
from methods.newton import newton
from methods.eliminacionSimple import Eliminacion as gaussSimple
from methods.pivoteoParcial import Eliminacion as pivoteoParcial
from methods.pivoteoTotal import Eliminacion as pivoteoTotal

app = Flask(__name__)
CORS(app)


def createImg(lines):
    print("Generating image")
    img_str = ""
    max_length = 0

    for line in lines:
        length = len(line)
        if length > max_length:
            max_length = length
        img_str += line

    img = Image.new("RGB", (max_length*10, len(lines)*19),
                    color=(255, 255, 255))
    writer = ImageDraw.Draw(img)

    console_font = ImageFont.truetype("clacon.ttf", 22, encoding="unic")

    writer.multiline_text((10, 10), u''+img_str,
                          fill=(0, 0, 0), font=console_font)

    print("Converting image to base64..")
    img_bytes_array = io.BytesIO()
    img.save(img_bytes_array, format="png")
    img_bytes_array = img_bytes_array.getvalue()

    return "data:image/png;base64,{}".format(b64encode(img_bytes_array).decode('utf-8'))


@app.route("/biseccion")
def biseccion():
    xi = np.double(float(request.args.get('xi')))
    xs = np.double(float(request.args.get('xs')))
    tolerancia = np.double(float(request.args.get('tolerancia')))
    niter = np.double(float(request.args.get('niter')))

    return jsonify(bise(xi, xs, tolerancia, niter))


@app.route("/incrementales")
def incrementales():
    x0 = np.double(float(request.args.get('x0')))
    delta = np.double(float(request.args.get('delta')))
    niter = np.double(float(request.args.get('niter')))
    return jsonify(incre(x0, delta, niter))


@app.route("/newton")
def newton():
    x0 = np.double(float(request.args.get('x0')))
    tolerancia = np.double(float(request.args.get('tolerancia')))
    niter = np.double(float(request.args.get('niter')))
    return jsonify(new(x0, tolerancia, niter))


@app.route("/puntoFijo")
def punto_fijo():
    tolerancia = np.double(float(request.args.get('tolerancia')))
    xa = np.double(float(request.args.get('xa')))
    niter = np.double(float(request.args.get('niter')))
    return jsonify(pf(tolerancia, xa, niter))

@app.route("/raiz")
def raizMultiples():
    x0 = np.double(float(request.args.get('x0')))
    tolerancia = np.double(float(request.args.get('tolerancia')))
    niter = np.double(float(request.args.get('niter')))

    return jsonify(rmul(x0, tolerancia, niter))


@app.route("/falsa")
def reglaFalsa():
    xi = np.double(float(request.args.get('xi')))
    xs = np.double(float(request.args.get('xs')))
    tolerancia = np.double(float(request.args.get('tolerancia')))
    niter = np.double(float(request.args.get('niter')))

    return jsonify(rf(xi, xs, tolerancia, niter))


@app.route("/secante")
def _secante():
    x0 = np.double(float(request.args.get('x0')))
    x1 = np.double(float(request.args.get('x1')))
    tolerancia = np.double(float(request.args.get('tolerancia')))
    niter = np.double(float(request.args.get('niter')))
    return jsonify(secante(x0, x1, tolerancia, niter))


@app.route("/eliminacionSimple")
def _gaussSimple():
    A = [[2, -3, 4, 1], [-4, 2, 1, -2], [1, 3, -5, 3], [-3, -1, 1, -1]]
    b = [10, -10, 32, -21]
    n = 4
    return jsonify(gaussSimple(A, b, n))


@app.route("/pivoteoParcial")
def _pivoteoParcial():
    A = [[2, -3, 4, 1], [-4, 2, 1, -2], [1, 3, -5, 3], [-3, -1, 1, -1]]
    b = [10, -10, 32, -21]
    n = 4
    return jsonify(pivoteoParcial(A, b, n))


@app.route("/pivoteoTotal")
def _pivoteoTotal():
    A = [[2, -3, 4, 1], [-4, 2, 1, -2], [1, 3, -5, 3], [-3, -1, 1, -1]]
    b = [10, -10, 32, -21]
    n = 4
    return jsonify(pivoteoTotal(A, b, n))


@app.route("/newtonInter")
def newtonInter():
    n = np.double(float(request.args.get('n')))
    x = np.double(float(request.args.get('x')))
    y = np.double(float(request.args.get('y')))
    return jsonify(newI(new(n, x, y), n-1))


@app.route("/lagrange")
def lagrange():
    valor = np.double(float(request.args.get('valor')))
    x = np.double(float(request.args.get('x')))
    y = np.double(float(request.args.get('y')))
    return jsonify(lag(valor, x, y))


@app.route("/")
def hello():
    return "quiskit_version: {}\npython_version: {}\nquiskit_libraries_version: {} <br> <h3>Services</h3><ul><li>fourier</li><li>sumador</li>".format(qiskit.__version__, sys.version, qiskit.__qiskit_version__)


@app.route("/fourier")
def fourier():
    IBMQ.save_account(
        "1ba6efe6f9c85bbdbe55aa57be4085614f79aa3c626875a997e704379ce1064a55e51325337bc41a69408d5fef5f6e2a9410cf23f3325d03b8f4bce00d057c5a")

    print("Connect with IBM")
    provider = IBMQ.load_account()
    simulator = provider.get_backend('ibmq_qasm_simulator')
    circuit = QuantumCircuit(3, 2)

    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    print("Executing circuit in IBM machine")
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print("Saving the circuit temp file")
    hashcode = random.random()
    circuit.draw(filename="circuit{}.temp".format(hashcode))

    print("Serializind data")
    tempCircuit = open("circuit{}.temp".format(hashcode), "r", encoding="utf8")
    circuitLines = tempCircuit.readlines()

    image = createImg(circuitLines)

    tempCircuit.close()
    os.remove("circuit{}.temp".format(hashcode))

    response = {"result": {
        "data": list(counts.keys()),
        "counts": counts
    }, "draw": image}
    print("Done.")
    return jsonify(response)


@app.route("/sumador")
def sumador():
    IBMQ.save_account(
        "1ba6efe6f9c85bbdbe55aa57be4085614f79aa3c626875a997e704379ce1064a55e51325337bc41a69408d5fef5f6e2a9410cf23f3325d03b8f4bce00d057c5a")

    print("Connect with IBM")
    provider = IBMQ.load_account()
    simulator = provider.get_backend("ibmq_qasm_simulator")

    hashcode = random.random()

    # Numbers sending for the user
    first = request.args.get('n1')
    second = request.args.get('n2')

    l = len(first)
    l2 = len(second)

    if l > l2:
        n = l
    else:
        n = l2

    a = QuantumRegister(n)  # First number
    b = QuantumRegister(n+1)  # Second number
    c = QuantumRegister(n)  # Carry bits
    cl = ClassicalRegister(n+1)  # Classical output
    circuit = QuantumCircuit(a, b, c, cl)

    print("Save circuit in first fase")
    circuit.draw(filename="circuit{}_fase1.temp".format(hashcode))

    # Setea los registros con los valores ingresados
    for i in range(l):
        if first[i] == "1":
            circuit.x(a[l - (i+1)])  # Flip the qubit from 0 to 1

    for i in range(l2):
        if second[i] == "1":
            circuit.x(b[l2 - (i+1)])

    print("Save circuit in second fase")
    circuit.draw(filename="circuit{}_fase2.temp".format(hashcode))

    for i in range(n-1):
        circuit.ccx(a[i], b[i], c[i+1])
        circuit.cx(a[i], b[i])
        circuit.ccx(c[i], b[i], c[i+1])

    circuit.ccx(a[n-1], b[n-1], b[n])
    circuit.cx(a[n-1], b[n-1])
    circuit.ccx(c[n-1], b[n-1], b[n])

    circuit.cx(c[n-1], b[n-1])

    for i in range(n-1):
        circuit.ccx(c[(n-2)-i], b[(n-2)-i], c[(n-1)-i])
        circuit.cx(a[(n-2)-i], b[(n-2)-i])
        circuit.ccx(a[(n-2)-i], b[(n-2)-i], c[(n-1)-i])
        # These two operations act as a sum gate; if a control bit is at
        # the 1> state then the target bit b[(n-2)-i] is flipped
        circuit.cx(c[(n-2)-i], b[(n-2)-i])
        circuit.cx(a[(n-2)-i], b[(n-2)-i])

    for i in range(n+1):
        circuit.measure(b[i], cl[i])

    print("Send to IBM machine")
    job2 = execute(circuit, simulator, shots=1000)
    result = job2.result()
    counts = result.get_counts(circuit)

    print("Save circuit in third fase")
    circuit.draw(filename="circuit{}_fase3.temp".format(hashcode))

    print("Serializing data")
    fase1 = open("circuit{}_fase1.temp".format(hashcode), "r", encoding="utf8")
    fase2 = open("circuit{}_fase2.temp".format(hashcode), "r", encoding="utf8")
    fase3 = open("circuit{}_fase3.temp".format(hashcode), "r", encoding="utf8")

    response = {
        "result": {
            "data": list(counts.keys()),
            "counts": counts
        },
        "draws": [createImg(fase1.readlines()), createImg(fase2.readlines()), createImg(fase3.readlines())]
    }

    fase1.close()
    fase2.close()
    fase3.close()

    os.remove("circuit{}_fase1.temp".format(hashcode))
    os.remove("circuit{}_fase2.temp".format(hashcode))
    os.remove("circuit{}_fase3.temp".format(hashcode))

    print("Done.")

    return jsonify(response)


if __name__ == '__main__':
    app.run(threaded=True, debug=False)
