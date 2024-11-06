# https://github.com/skushagra/quantum/blob/911161315a44563c97fdd1ee4c2ffacc5030ce78/python_files/random_integer_gen.py
import qiskit
from qiskit import *
from qiskit.tools.visualization import plot_histogram
import random
from numpy import pi
# --
def gen_rand(circ, f):
    fs = f
    exec = execute(circuit, backend=Aer.get_backend('qasm_simulator'), shots=1).result()
    counts = exec.get_counts()
    rand_int = int(random.randint(0,50))
    for i in counts:
        if i == '0' and len(fs)<rand_int:
            fs += i
            gen_rand(circ, fs)
        elif i == '1' and len(fs)<rand_int:
            fs += '1'
            gen_rand(circ, fs)
        else:
            print(int(fs, 2))
            break
# --
circuit = QuantumCircuit(1,1)
circuit.rx(pi/2,0)
circuit.measure(0,0)
gen_rand(circuit, f='0')
