# https://github.com/grig-ar/Projects/blob/b36e8d45956d29ac5287e9571cc83ebf0b04fabe/4/Python/Quantum%20Tester/main/quantum/circuit/ShorCircuit.py
import qiskit as q
from main.quantum.circuit.ICircuit import ICircuit
from math import pi as PI
from main.quantum.circuit.ElementsConstructor import ElementsConstructor


class ShorCircuit(ICircuit):

    def add_circuit(self, circuit: q.QuantumCircuit):
        if circuit.num_qubits != 5:
            raise q.exceptions.QiskitError('num_qubits must be 5 for this circuit')
        constructor = ElementsConstructor()

        constructor.add_h_gate_to_all(circuit)
        circuit.h(2)
        circuit.h(0)
        circuit.cx(2, 3)
        circuit.crz(PI / 2, 0, 1)
        circuit.cx(2, 4)
        circuit.h(1)
        circuit.crz(PI / 4, 0, 2)
        circuit.crz(PI / 2, 1, 2)
        circuit.h(2)
        constructor.add_measure(circuit)

    @staticmethod
    def get_inspire_circuit(backend_name: str):
        if backend_name == 'Starmon-5':
            return '''
            version 1.0
            
            qubits 5
        
            {H q[0] | H q[1] | H q[2] }
            H q[0]
        
            CNOT q[2], q[3]
        
            RZ q[2], 0.7854
            CNOT q[0], q[2]
            RZ q[2], -0.7854
            CNOT q[0], q[2]
        
            CNOT q[2], q[4]
            H q[1]
        
            RZ q[2], 0.3927
            CNOT q[0], q[2]
            RZ q[2], -0.3927
            CNOT q[0], q[2]
        
            RZ q[2], 0.3927
            CNOT q[1], q[2]
            RZ q[2], -0.3927
            CNOT q[1], q[2]
        
            H q[2]
        
            measure_z q[0]
            measure_z q[1]
            measure_z q[2]
            '''
        else:
            return '''
            version 1.0
        
            qubits 5
        
            {H q[0] | H q[1] | H q[2] }
            H q[0]
        
            CNOT q[2], q[3]
        
            RZ q[1], 0.7854
            CNOT q[0], q[1]
            RZ q[1], -0.7854
            CNOT q[0], q[1]
        
            CNOT q[2], q[4]
            H q[1]
        
            RZ q[2], 0.3927
            CNOT q[0], q[2]
            RZ q[2], -0.3927
            CNOT q[0], q[2]
        
            RZ q[2], 0.3927
            CNOT q[1], q[2]
            RZ q[2], -0.3927
            CNOT q[1], q[2]
        
            H q[2]
        
            measure_z q[0]
            measure_z q[1]
            measure_z q[2]
            '''
