# https://github.com/danchepkwony/qchacks-python/blob/589ebbcfbd777c1ebcabebf194c99b43678bcfa6/QiskitWorker.py
from qiskit import *
from qiskit.providers.ibmq import least_busy
    
IBMQ.save_account('daffffa40c20d3f39cbcd68720561cec5e7cffa25c8f863423ffbf18e3c79b8fa07253d502fa37210e4177c176a282d56e54ba665541c2f272ac07eca13580b8')
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')

class QiskitWorkout():
    def __init__(self, message):
        self.message = message

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()
    
    def run(self):
        note = self.message
        circuit = QuantumCircuit(4, 3)
        circuit.h(3)
        circuit.z(3)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.barrier()

        note = note[::-1]
        if note[0] == '0':
            circuit.i(0)
        else:
            circuit.cx(0, 3)
            
        if note[1] == '0':
            circuit.i(1)
        else:
            circuit.cx(1, 3)
            
        if note[2] == '0':
            circuit.i(2)
        else:
            circuit.cx(2, 3)
            
        circuit.barrier()
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        circuit.measure(2, 2)

        result = execute(circuit, backend = backend, shots = 1000).result().get_counts()
        return result
