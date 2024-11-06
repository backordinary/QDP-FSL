# https://github.com/a1426/Rail-Circuit/blob/4b5cf8d79f930d448132635cb7d87676caf9edfa/src/source_gen.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from random import randint, choice
import matplotlib.pyplot as plt
import gate_finder
from collections import defaultdict
import os
def random_component(obj):
    return choice([obj.x, obj.y, obj.z,obj.h, obj.i, obj.s, obj.sdg, obj.t, obj.tdg])
component_list=["x","y","z","h","i","s","sdg","t","tdg"]

class Simple_Square_Gates:
    def __init__(self,size):
        self.folder_sizes = defaultdict(int)
        self.circuit=QuantumCircuit(size)
        self.history= defaultdict(int)
        self.gates={}
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
        method=random_component(self.circuit)
        pos=randint(0,size-1)
        self.history[pos]+=1
        method(pos)
        self.gates[f'{pos}-{self.history[pos]}']=method.__name__
    def export(self):
        self.circuit.draw(output="mpl")
        plt.savefig("generated_circuits/test.png")
    def generate_folders(self,path):
        if os.path.exists(path):
            gate_finder.clear(path)
        else:
            os.makedirs(path)
        gate_finder.isolate_gates("generated_circuits/test.png",path)
        for file_path in os.listdir(path):
            file_name=os.path.splitext(file_path)[0]
            current_gate=self.gates[file_name]
            name=str(self.folder_sizes[current_gate])
            os.rename(os.path.join("img_save",file_path),os.path.join("gates",current_gate,name+".png"))
            self.folder_sizes[current_gate]+=1
def initialize_folders():
    for gate_type in component_list:
        try:
            os.makedirs(os.path.join("gates",gate_type))
        except FileExistsError:
            pass
def generate(size, folder_size=None):
    c1=Simple_Square_Gates(size)
    if folder_size: c1.folder_sizes = folder_size
    c1.export()
    c1.generate_folders(f"img_save")
    return c1.folder_sizes
