# https://github.com/Siddharthgolecha/Quantum-Compressor/blob/e76cbe3939032d79cfb0f5677233bed6a4e40d60/main.py
from flask import Flask, render_template, request
import numpy as np
from qiskit import QuantumCircuit, Aer

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
  
@app.route('/', methods=['GET','POST'])
def computing():
    out = ""
    if request.method == 'POST':
        data = request.json.get('data')
        chars = np.array([ord("Z")-ord(i) for i in data.upper()])
        angles = 2*np.arcsin(np.sqrt(chars/25))
        n = len(angles)
        qc = QuantumCircuit(n,n)
        for i in range(len(angles)):
            qc.ry(angles[i],i)
        qc.measure(range(n),range(n))
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(qc, shots=1024)
        counts = job.result().get_counts()
        arr = [0 for _ in range(n)]
        for state, count in counts.items():
            state = state[::-1]
            for i in range(n):
                arr[i] += int(state[i])*count/1024
        for i in range(n):
            out += chr(ord("A")+25-int(round(arr[i]*25,0)))
        return out
    else:  
        return render_template('index.html')      
  
# main driver function
if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)