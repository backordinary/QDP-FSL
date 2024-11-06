# https://github.com/agrawalnishant2303/VLSI-testing-using-Grover-s-Algorithm/blob/8776c4d60b5564c77157acfda608c4caa3d200f4/Final%20Code/FinalCode.py
import numpy as np
from qiskit import BasicAer
from qiskit.visualization import plot_histogram
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle, TruthTableOracle
def AND(a,b,c):
     return '({} + -{}) * ({} + -{}) * (-{} + -{} + {})'.format(a,c,b,c,a,b,c)
def OR(a,b,c):
    return "(-{} + {}) * (-{} + {}) * ({} + {} + -{})".format(a,c,b,c,a,b,c)
def NAND(a,b,c):
    return "({} + {}) * ({} + {}) * (-{} + -{} + -{})".format(a,c,b,c,a,b,c)
def NOR(a,b,c):
    return "(-{} + -{}) * (-{} + -{}) * ({} + {} + {})".format(a,c,b,c,a,b,c)
def NOT(a,b):
    return "({} + {}) * (-{} + -{})".format(a,b,a,b)
def XOR(a,b,c):
    return "(-{} + {} + {}) * ({} + -{} + {}) * (-{} + -{} + -{}) * ({} + {} + -{}))".format(a,b,c,a,b,c,a,b,c,a,b,c)
import numpy as np
import csv
my_set = set()
boolean_expression = " "
with open('Circuit Description.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if(row[0] == 'NOT'):
            my_set.add(row[1])
            my_set.add(row[3])
        else:
            my_set.add(row[1])
            my_set.add(row[2])
            my_set.add(row[3])
    ff_out = row[3]
with open('Circuit Description.csv', 'r') as file:
    reader = csv.reader(file)
    items = []
    for row in reader:
        items
        if(row[0] == 'AND'):
            boolean_expression += AND(row[1],row[2],row[3])
        elif(row[0] == 'OR'):
            boolean_expression += OR(row[1],row[2],row[3])
        elif(row[0] == 'NAND'):
            boolean_expression += NAND(row[1],row[2],row[3])
        elif(row[0] == 'NOR'):
            boolean_expression += NOR(row[1],row[2],row[3])
        elif(row[0] == 'XOR'):
            boolean_expression += XOR(row[1],row[2],row[3])
        elif(row[0] == 'NOT'):
            boolean_expression += NOT(row[1],row[3])
        boolean_expression += ' * '
def faulty_ckt(variable,value):
    boolean_expression1 = "({}*)".format(variable)
    with open('Circuit Description.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if(row[1]==variable):
                boolean_expression1 += ' * '
                row[1] = "{}*".format(row[1])
                variable=row[3]
                row[3] = "{}*".format(row[3])
                if(row[0] == 'AND'):
                    boolean_expression1 += AND(row[1],row[2],row[3])
                elif(row[0] == 'OR'):
                    boolean_expression1 += OR(row[1],row[2],row[3])
                elif(row[0] == 'NAND'):
                    boolean_expression1 += NAND(row[1],row[2],row[3])
                elif(row[0] == 'NOR'):
                    boolean_expression1 += NOR(row[1],row[2],row[3])
                elif(row[0] == 'XOR'):
                    boolean_expression1 += XOR(row[1],row[2],row[3])
                elif(row[0] == 'NOT'):
                    boolean_expression1 += NOT(row[1],row[3])
            elif(row[2]==variable):
                boolean_expression1 += ' * '
                row[2] = "{}*".format(row[2])
                variable=row[3]
                row[3] = "{}*".format(row[3])
                if(row[0] == 'AND'):
                    boolean_expression1 += AND(row[1],row[2],row[3])
                elif(row[0] == 'OR'):
                    boolean_expression1 += OR(row[1],row[2],row[3])
                elif(row[0] == 'NAND'):
                    boolean_expression1 += NAND(row[1],row[2],row[3])
                elif(row[0] == 'NOR'):
                    boolean_expression1 += NOR(row[1],row[2],row[3])
                elif(row[0] == 'XOR'):
                    boolean_expression1 += XOR(row[1],row[2],row[3])
                elif(row[0] == 'NOT'):
                    boolean_expression1 += NOT(row[1],row[3])
            if(row[0] == 'NOT'):
                my_set.add(row[1])
                my_set.add(row[3])
            else:
                my_set.add(row[1])
                my_set.add(row[2])
                my_set.add(row[3])
        return boolean_expression1,row[3]

faulty_boolean_exp,faulty_out = faulty_ckt('E',1)

my_set.add('out')
boolean_expression1 = " * "
boolean_expression1 += XOR(ff_out,faulty_out,'out')
faulty_boolean_exp += boolean_expression1
my_list = list(my_set)
my_list.sort()
my_list
a = {k: v for v, k in enumerate(my_list,start = 1)}
search_key = '*'
res = dict(filter(lambda item: search_key in item[0], a.items()))
keys_values = res.items()
new_d = {str(key): str(value) for key, value in keys_values}
for key in new_d.keys():
    faulty_boolean_exp = faulty_boolean_exp.replace(key, new_d[key])
boolean_expression += faulty_boolean_exp
keys_values = a.items()
new_d = {str(key): str(value) for key, value in keys_values}
new_d
for key in new_d.keys():
    boolean_expression = boolean_expression.replace(key, new_d[key])
import re
res = re.findall(r'\(.*?\)', boolean_expression)
new_str = ""
for i in range(len(res)):
    parse = re.findall(r'[-+]?[0-9]+', res[i])
    for j in range(len(parse)):
        new_str += '{} '.format(parse[j])
    new_str += "0\n"
input_3sat = '''
c example DIMACS-CNF 3-SAT
p cnf {} {}
'''.format(len(my_set),len(res))
input_3sat += new_str
oracle = LogicalExpressionOracle(input_3sat)
grover = Grover(oracle)
from qiskit import IBMQ
IBMQ.save_account('API_token')
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q',group='open')
sim = provider.get_backend('simulator_mps')
quantum_instance = QuantumInstance(sim, shots=8192)
result = grover.run(quantum_instance)
print(result['assignment'])
from qiskit.compiler import transpile
grover_compiled = transpile(result['circuit'],backend = sim, optimization_level =3)
print('gates = ',grover_compiled.count_ops())
print('depth = ',grover_compiled.depth())
result['top_measurement']
qc = result['circuit']
qc.draw()