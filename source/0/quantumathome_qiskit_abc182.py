# https://github.com/cbarange/workshop_api_cyberquantum/blob/8b2d968b45bd66d743d9c5ab1e465f569c64dfff/quantumathome_qiskit.py
#!/bin/python3
# -*- coding: utf-8 -*-
"""
pip install pyinstaller
pyinstaller -F quantumathome_naive.py
pyinstaller --hidden-import qiskit --onefile quantumathome_qiskit.py
"""


import math
import random


import json
import sys

import time
import requests
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor




def shor_qiskit(number):
  N = int(number)
  backend = Aer.get_backend('aer_simulator')
  quantum_instance = QuantumInstance(backend, shots=1024)
  shor = Shor(quantum_instance=quantum_instance)
  result = shor.factor(N)
  return result
  

if __name__ == '__main__':
  
  array = [9,15,21,33,39,51,57,69,87,93,111,123,129,141,159,177,15,25,35,55,65,85,95,115,145,155,185,205,215,235,265,295,21,35,49,77,91,119,133,161,203,217,259,287,301,329,371,413,33,55,77,121,143,187,209,253,319,341,407,451,473,517,583,649,39,65,91,143,169,221,247,299,377,403,481,533,559,611,689,767,51,85,119,187,221,289,323,391,493,527,629,697,731,799,901,1003,57,95,133,209,247,323,361,437,551,589,703,779,817,893,1007,1121,69,115,161,253,299,391,437,529,667,713,851,943,989,1081,1219,1357,87,145,203,319,377,493,551,667,841,899,1073,1189,1247,1363,1537,1711,93,155,217,341,403,527,589,713,899,961,1147,1271,1333,1457,1643,1829,111,185,259,407,481,629,703,851,1073,1147,1369,1517,1591,1739,1961,2183,123,205,287,451,533,697,779,943,1189,1271,1517,1681,1763,1927,2173,2419,129,215,301,473,559,731,817,989,1247,1333,1591,1763,1849,2021,2279,2537,141,235,329,517,611,799,893,1081,1363,1457,1739,1927,2021,2209,2491,2773,159,265,371,583,689,901,1007,1219,1537,1643,1961,2173,2279,2491,2809,3127,177,295,413,649,767,1003,1121,1357,1711,1829,2183,2419,2537,2773,3127,3481]
  
  for number in array:
    if number%2==0:
      continue
    
    
    # - qiskit -


    # - naive -
    start_time = time.time()
    res = shor_qiskit(number)
    factors = map(str,res.factors)
    compute_time = '{:.12f}'.format((time.time() - start_time))
    
    # a quoi resemble res 
    
    if len(res.factors)!=0:
      result = ''
      if type(res.factors[0]) == list:
        result = res.factors[0][0]
      else:
        result = res.factors[0]  
    
      response=requests.put(f'https://workshop-cyberquantum-api.herokuapp.com/numbers/found/{number}/{compute_time}/naive/{sys.argv[1]}').text
      print(f"Factor found : {result} x {round(number/result)} = {number}, compute in {compute_time} seconds. {sys.argv[1]} you win {result} points, total:{response}") 
      time.sleep(1.235)
      # logging.info(f"Factor found : {res} x {round(number/res)} = {number}, compute in {compute_time} seconds")    

  # print(json.dumps(res))
  
    
  
    
    
  
