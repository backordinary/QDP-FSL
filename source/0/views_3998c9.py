# https://github.com/sheryylli/adsmidterm/blob/0e61cff1fdc7bb4bd1a4f3a9f0be858e27c2f2e8/grovers/views.py
from django.shortcuts import render
import json
#create ur views here
from qiskit import IBMQ
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import PhaseOracle
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.tools.monitor import job_monitor

IBMQ.enable_account('2c7a7ecc86e2ce1698d3c15bf742a5ac234cd0baa886913ea8e1185ce90c4721f3bba34ca79f085b191f60f5273ebc629d114d79c9ba35bfedfeda8be4110a1e')
provider = IBMQ.get_provider(hub='ibm-q')

def home(request):
    return render(request, 'index.html', {})

def grovers(request):

    print(request.body)

    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode);

    device = body['device']
    backend = provider.get_backend(device)

    expression = body['expression']
    oracle = PhaseOracle(expression)

    problem = AmplificationProblem(oracle=oracle, is_good_state=oracle.evaluate_bitstring)
    grover = Grover(quantum_instance=backend)

    result = grover.amplify(problem)

    counts = result.circuit_results

    response = JsonResponse({'result': str(counts)})
    return response