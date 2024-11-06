# https://github.com/Quantum-Computing-Philippines/ibm-quantum-coin-flipper/blob/62172565ee954ff270c7e6ddc404fcb69cec91a7/qcf-backend/qrng/views.py
import json

from django.http import JsonResponse
from django.shortcuts import HttpResponse, render
from qiskit import (IBMQ, ClassicalRegister, QuantumCircuit, QuantumRegister,
                    execute)
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy

def home(request):
    return render(request, 'index.html', {})
    
# Performs Quantum Coin Flip
def results(request):
    try:
        IBMQ.disable_account()
        s = str(request.body)
        apikey =  s.split('apikey=')[1]
        apikey =  apikey.split('&backendkey', 1)[0]
        backendkey = s.split('backendkey=')[1]
        backendkey = backendkey[:-1]
        IBMQ.enable_account(f'{apikey}')
        provider = IBMQ.get_provider(hub='ibm-q')
        device = backendkey
        backend = provider.get_backend(device)

        # Execute coin flipper
        q = QuantumRegister(1)
        c = ClassicalRegister(1)
        circuit = QuantumCircuit(q, c)
        circuit.h(q) 
        circuit.measure(q, c) 

        job = execute(circuit, backend, shots=1024)
        job_monitor(job)
        counts = job.result().get_counts()

        print('RESULT: ', counts, '\n')

        job_id = job.job_id()
        result = int(counts.most_frequent(), 2)
        

        if result == 0:
            response = JsonResponse({'result':'tails', 'value': result, 'job_id': job_id})
        if result == 1:
            response = JsonResponse({'result':'heads', 'value': result, 'job_id': job_id})
        IBMQ.disable_account()
        return response

    except:
        s = str(request.body)
        apikey =  s.split('apikey=')[1]
        apikey =  apikey.split('&backendkey', 1)[0]
        backendkey = s.split('backendkey=')[1]
        backendkey = backendkey[:-1]
        IBMQ.enable_account(f'{apikey}')
        provider = IBMQ.get_provider(hub='ibm-q')
        device = backendkey
        backend = provider.get_backend(device)

        # Execute coin flipper
        q = QuantumRegister(1)
        c = ClassicalRegister(1)
        circuit = QuantumCircuit(q, c)
        circuit.h(q) 
        circuit.measure(q, c) 

        job = execute(circuit, backend, shots=1024)
        job_monitor(job)
        counts = job.result().get_counts()

        print('RESULT: ', counts, '\n')

        job_id = job.job_id()

        # transform to get the most frequent count of either 0 or 1
        result = int(counts.most_frequent(), 2)

        if result == 0:
            response = JsonResponse({'result':'tails', 'value': result, 'job_id': job_id})
        if result == 1:
            response = JsonResponse({'result':'heads', 'value': result, 'job_id': job_id})
        IBMQ.disable_account()
        return response    



# Call all available IBM Quantum Backends
def backends(request):
    try:
        IBMQ.disable_account()
        s = str(request.body)
        apikey =  s.split('=')[1]
        apikey = apikey[:-1]
        print(apikey)
        IBMQ.enable_account(f'{apikey}')
        provider = IBMQ.get_provider(hub='ibm-q')
        allBackends = ([backend.name() for backend in IBMQ.providers()[0].backends()])
        IBMQ.disable_account()
        return JsonResponse(allBackends, safe=False)
    except:
        s = str(request.body)
        apikey =  s.split('=')[1]
        apikey = apikey[:-1]
        print(apikey)
        IBMQ.enable_account(f'{apikey}')
        provider = IBMQ.get_provider(hub='ibm-q')
        allBackends = ([backend.name() for backend in IBMQ.providers()[0].backends()])
        IBMQ.disable_account()
        return JsonResponse(allBackends, safe=False)
    

# Call least busy backend
def leastbusy(request):
    try:
        IBMQ.disable_account()
        s = str(request.body)
        apikey =  s.split('=')[1]
        apikey = apikey[:-1]
        print(apikey)
        IBMQ.enable_account(f'{apikey}')
        provider = IBMQ.get_provider(hub='ibm-q')
        allBackends = ([backend.name() for backend in IBMQ.providers()[0].backends()])
        backend = least_busy(provider.backends(simulator=False))
        backend = least_busy(provider.backends(simulator=False))
        backend_leastbusy = str(backend.name())
        IBMQ.disable_account()
        return JsonResponse(backend_leastbusy, safe=False)
    except:
        s = str(request.body)
        apikey =  s.split('=')[1]
        apikey = apikey[:-1]
        print(apikey)
        IBMQ.enable_account(f'{apikey}')
        provider = IBMQ.get_provider(hub='ibm-q')
        allBackends = ([backend.name() for backend in IBMQ.providers()[0].backends()])
        backend = least_busy(provider.backends(simulator=False))
        backend = least_busy(provider.backends(simulator=False))
        backend_leastbusy = str(backend.name())
        IBMQ.disable_account()
        return JsonResponse(backend_leastbusy, safe=False)
