# https://github.com/stingraze/quantum-web-crawler-controller/blob/8c9c56723396abbc46d40052f1a29a6788b2f61b/quantum-controlled-web-crawler-controller.py
#Some code from "Quantum Machine Learning with Python: Using Cirq from Google Research and IBM Qiskit" by  Santanu Pattanayak. 
#Highly modified code by Tsubasa Kato at Inspire Search Corporation 12/9/2022 0:40PM
import os
from qiskit import QuantumCircuit, execute, Aer
import time
import subprocess

from io import TextIOWrapper, TextIOBase, StringIO
from subprocess import PIPE, Popen, call
from tempfile import TemporaryFile
from sarge import run, Capture

np.seterr(divide='ignore')
counter = 0
#Most of code from: https://stackoverflow.com/questions/67972186/live-output-from-python-subprocess-that-calls-python-script
def run_command(command):
   os.environ['PYTHONUNBUFFERED'] = '1'
   process = Popen(command, shell=False, stdout=PIPE, env=os.environ) # Shell doesn't quite matter for this issue
   while True:
      output = process.stdout.readline()
      if process.poll() is not None:
         break
      if output:
         print(output.decode('utf-8'))
   rc = process.poll()
   return rc

def quantum_compute():
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(1,1) 
    circuit.h(0)

    circuit.measure(range(1),range(1))
    job = execute(circuit, simulator, shots=100)

    result = job.result()
    counts = result.get_counts(circuit)

    print(circuit.draw(output='text'))
    time.sleep(7)
    count_items = counts.items()
    counter_items = 0
    val_zero = 0
    val_one = 0

    for key, value in counts.items():
        print(key, "->", value)
        if (counter_items == 0):
            val_one = value          
        if (counter_items == 1):
            val_zero = value
            
        counter_items = counter_items + 1

    if(val_zero > val_one):
        print("zero is greater")
        print("=======")
        print("1:" + str(val_one))
        print("0:" + str(val_zero))
        print("\nTotal count for 0 and 1 are:", counts)
        #Runs Crawler in 3 threads
        run_command(['python3', "multithread-crawler.py", "3"])

    if (val_zero < val_one):
        print("one is greater")
        print("=======")
        print("1:" + str(val_one))
        print("0:" + str(val_zero))
        print("\nTotal count for 0 and 1 are:", counts)
        #Runs Crawler in 4 threads
        run_command(['python3', "multithread-crawler.py", "4"])

    if (val_zero == val_one):
        print("both are equal")
        print("=======")
        print("1:" + str(val_one))
        print("0:" + str(val_zero))
        print("\nTotal count for 0 and 1 are:", counts)
        #Runs Crawler in 5 threads
        run_command(['python3', "multithread-crawler.py", "5"])
start = time.time_ns() / (10 ** 9)
#If running quantum_compute() in a loop, uncomment the lines below.
#You will need to implement more code to make it useful.
#while (counter < 10):
#    counter = counter + 1
#    print(str(counter) + "\n") 
quantum_compute()
stop = time.time_ns() / (10 ** 9)
#If loop above is enabled, code below will run.
#print("Computed in Quantum Simulator " + str(counter) + " times")
benchmark_time = stop - start
print(benchmark_time)