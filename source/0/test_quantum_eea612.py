# https://github.com/Jb-Za/QuantumTeleportationProtocol/blob/c600aaeffc022db3c4a81caad22b1843c5d4b149/test_quantum.py
import unittest
import QuantumTeleportation
from unittest.mock import Mock
from qiskit import *
import matplotlib.pyplot as plt


class TestQuantum(unittest.TestCase):
    def test_string_to_binary(self):
        helloWorld = [1001000 , 1100101, 1101100, 1101100, 1101111, 100000, 1110111, 1101111, 1110010, 1101100, 1100100]
        result = QuantumTeleportation.string_to_binary("Hello world")
        #print("\nstring to binary expected output: " , helloWorld)
        #print("string to binary given output: " , result)
        self.assertEqual(result ,helloWorld)

    def test_binary_to_string(self):
        helloWorld = [1001000 , 1100101, 1101100, 1101100, 1101111, 100000, 1110111, 1101111, 1110010, 1101100, 1100100]
        result = QuantumTeleportation.binary_to_string(helloWorld)
        #print("\nbinary to string expected output: Hello world" )
        #print("binary to string given output: " , result)
        self.assertEqual(result, "Hello world")

    
    def test_Send_Message(self):
        result = QuantumTeleportation.Send_Message(0)
        print("\nsend individual bit expected output: 0" )
        print("send individual bit given output: " , result)
        self.assertEqual(result ,0)
        
        #self.assertEqual()

    def test_executeCircuits(self):
        eight_bits = [False , False ,False ,False ,False ,False ,False ,False ]
        circuit = QuantumCircuit(24 , 8)
        if eight_bits[0] == False:
            circuit.x(0)
        if eight_bits[1] == False:
            circuit.x(1)
        if eight_bits[2] == False:
            circuit.x(2)
        if eight_bits[3] == False:
            circuit.x(3)
        if eight_bits[4] == False:
            circuit.x(4)
        if eight_bits[5] == False:
            circuit.x(5)
        if eight_bits[6] == False:
            circuit.x(6)
        if eight_bits[7] == False:
            circuit.x(7)
        circuit.barrier()
        # alice
        circuit.h(0 + 8)
        circuit.cx(0+8 , 0+16)
        circuit.cx(0 , 0+8)
        circuit.h(0)
        circuit.barrier()   
        #bob
        circuit.cx(0+8 , 0+16)
        circuit.cz(0 , 0+16)
        circuit.barrier()
        # alice
        circuit.h(1 + 8)
        circuit.cx(1+8 , 1+16)
        circuit.cx(1 , 1+8)
        circuit.h(1)
        circuit.barrier()   
        #bob
        circuit.cx(1+8 , 1+16)
        circuit.cz(1 , 1+16)
        circuit.barrier()
        # alice
        circuit.h(2 + 8)
        circuit.cx(2+8 , 2+16)
        circuit.cx(2 , 2+8)
        circuit.h(2)
        circuit.barrier()   
        #bob
        circuit.cx(2+8 , 2+16)
        circuit.cz(2 , 2+16)
        circuit.barrier()
        # alice
        circuit.h(3 + 8)
        circuit.cx(3+8 , 3+16)
        circuit.cx(3 , 3+8)
        circuit.h(3)
        circuit.barrier()   
        #bob
        circuit.cx(3+8 , 3+16)
        circuit.cz(3 , 3+16)
        circuit.barrier()
        # alice
        circuit.h(4 + 8)
        circuit.cx(4+8 , 4+16)
        circuit.cx(4 , 4+8)
        circuit.h(4)
        circuit.barrier()   
        #bob
        circuit.cx(4+8 , 4+16)
        circuit.cz(4 , 4+16)
        circuit.barrier()
        # alice
        circuit.h(5 + 8)
        circuit.cx(5+8 , 5+16)
        circuit.cx(5 , 5+8)
        circuit.h(5)
        circuit.barrier()   
        #bob
        circuit.cx(5+8 , 5+16)
        circuit.cz(5 , 5+16)
        circuit.barrier()
        # alice
        circuit.h(6 + 8)
        circuit.cx(6+8 , 6+16)
        circuit.cx(6 , 6+8)
        circuit.h(6)
        circuit.barrier()   
        #bob
        circuit.cx(6+8 , 6+16)
        circuit.cz(6 , 6+16)
        circuit.barrier()
        # alice
        circuit.h(7 + 8)
        circuit.cx(7+8 , 7+16)
        circuit.cx(7 , 7+8)
        circuit.h(7)
        circuit.barrier()   
        #bob
        circuit.cx(7+8 , 7+16)
        circuit.cz(7 , 7+16)
        circuit.barrier()
        circuit.measure(0 + 16, 0 ) 
        circuit.measure(1 + 16, 1 ) 
        circuit.measure(2 + 16, 2 ) 
        circuit.measure(3 + 16, 3 ) 
        circuit.measure(4 + 16, 4 ) 
        circuit.measure(5 + 16, 5 ) 
        circuit.measure(6 + 16, 6 ) 
        circuit.measure(7 + 16, 7 ) 

        #circuit.draw(output='mpl', fold= -1)
        #plt.show()
        
        result = QuantumTeleportation.executeCircuits([circuit, circuit])
        print("\nsend eight bits expected output: ", eight_bits, eight_bits  )
        print("send eight bits given output: " , result[0])
        print("this is sending two circuits at once, so the output is 16 bits")
        self.assertEqual(result[0] , [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) # given the fact that it is a quantum circuit, there tends to be individual incorrect values sometimes. this is normal





if __name__ == "__main__":
    unittest.main()