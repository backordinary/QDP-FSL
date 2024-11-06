# https://github.com/shikharsingh3/Qosf_Screening_Task_2/blob/3572f73b12f0756b3c18fd0ccd833f24327baae3/.python/UnitTests.py
"""Task 2 Unit Tests"""

from Task_2 import BellStateCode
import unittest
from qiskit import QuantumCircuit

"""The measurement results are checked using different error gates for the two input qubit states. 
"""

class TestCalc(unittest.TestCase):

 """
 The following tests check whether measurement results of first and fourth qubit of the 6 qubit register are in line 
 with the expected measurement result of the Bell State ((|00〉 + |11〉)/sqrt(2)), i.e., 00 or 11. 
 """

 def test_Z0X0(self):
   """Z0X0 means Z and X gate is applied on the first and the fourth qubit of the 6 qubit 
   register respectively.
   """
   e1 = QuantumCircuit(3)
   e1.z(0)
   e2 = QuantumCircuit(3)
   e2.x(0)
   memory = BellStateCode(e1, e2) 
   for bit in range(0, len(memory)-1):
     self.assertTrue(memory[bit] == '00' or memory[bit] =='11')

 def test_Z2X1(self):
   """Z2X1 means Z and X gate is applied on the third and fifth qubit of the 6 qubit
   register respectively.
   """
   e1 = QuantumCircuit(3)
   e1.z(2)
   e2 = QuantumCircuit(3)
   e2.x(1)
   memory = BellStateCode(e1, e2) 
   for bit in range(0, len(memory)-1):
     self.assertTrue(memory[bit] == '00' or memory[bit] =='11') 
  
 def test_IdX2(self):
   """IdX2 means identity and X gate is applied on the first three qubits and the sixth qubit of the 6 qubit
   register respectively.
   """
   e1 = QuantumCircuit(3)
   e2 = QuantumCircuit(3)
   e2.x(2)
   memory = BellStateCode(e1, e2) 
   for bit in range(0, len(memory)-1):
     self.assertTrue(memory[bit] == '00' or memory[bit] =='11')
  
 def test_X2Z1(self):
   """X2Z1 means X and Z gate is applied on the third and second qubit of the 6 qubit
   register respectively.
   """
   e1 = QuantumCircuit(3)
   e1.x(2)
   e2 = QuantumCircuit(3)
   e2.z(1)
   memory = BellStateCode(e1, e2) 
   for bit in range(0, len(memory)-1):
     self.assertTrue(memory[bit] == '00' or memory[bit] =='11')

 def test_Z0Id(self):
   """Z0Id means Z and Identity gate is applied on the first and last three qubits of the 6 qubit
   register respectively.
   """
   e1 = QuantumCircuit(3)
   e1.z(0)
   e2 = QuantumCircuit(3)
   memory = BellStateCode(e1, e2) 
   for bit in range(0, len(memory)-1):
     self.assertTrue(memory[bit] == '00' or memory[bit] =='11')

if __name__ == '__main__':
 unittest.main()
