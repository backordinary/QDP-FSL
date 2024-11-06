# https://github.com/JonasReichhardt/qCharta/blob/ce0f6af2f11cdccc1134ef4e9ff6674718d33f60/src/main.py
from qiskit import QuantumCircuit, transpile
from qiskit.test.mock.backends import FakeBrooklyn
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, PassManager

import os
from qCharta import qCharta
from sabre import Sabre
from coupling import coupling
from support_funcs import get_circuit_cost, get_layout_description_comment, check_equivalence

inputdir = "..\\benchmarks\\"
outputdir = "..\\mapped\\"
coupling_map = CouplingMap(coupling)
seed = 100

def main():
    # get all qasm files of directory
    files = os.listdir(inputdir)
    files = list(filter(lambda path: path.endswith(".qasm"),files))

    quantum_circuits = {}
    print("Parsing .qasm files")
    for filename in files:
        quantum_circuits[filename]=QuantumCircuit.from_qasm_file(path=inputdir+filename)

    mean = 0
    lastRun = []

    for i in range(25):
        result = qCharta_benchmark(quantum_circuits)

        lastCost = result[1]
        mean = mean + lastCost
    
    print("mean over 25 runs:"+str(mean/25))

def reference_benchmark(quantum_circuits):
    reference_results = []
    reference_cost = 0

    for name, circuit in quantum_circuits.items():
        transpiled = transpile(circuit, coupling_map=coupling_map,routing_method="basic")

        cost = get_circuit_cost(transpiled)
        reference_results.append(cost)
        reference_cost = reference_cost+cost
            
        transpiled.qasm(filename=outputdir+"reference\\"+name)
    
    return [reference_cost,reference_results]
    
def qCharta_benchmark(quantum_circuits, check_eq = False):
    own_results = []
    own_cost = 0

    for name, circuit in quantum_circuits.items():
        # create transpiler with coupling map
        transpiler = qCharta(coupling_map, seed, "heuristic")

        # create pass manager and append transformation pass
        pass_manager = PassManager()
        pass_manager.append(transpiler)

        # run transformation pass
        mapped_qc = pass_manager.run(circuit)

        cost = get_circuit_cost(mapped_qc)
        own_results.append(cost)
        own_cost = own_cost+cost

        if check_eq:
            # check if result is equivalent to original ciruit
            if check_equivalence(circuit,mapped_qc):
                print(name +": result = original")
            else:
                print(name +": result is not equivalent")

        filecontent = mapped_qc.qasm()
        filecontent = filecontent.replace('\n', '\n' + get_layout_description_comment(transpiler.initial_layout, circuit_to_dag(mapped_qc)) + '\n', 1)

        with open(outputdir+"qCharta\\"+name, "w+") as file:
            file.write(filecontent)
        file.close()

    return [own_cost,own_results]

def sabre_benchmark(quantum_circuits, mapping_strategy):
    results = []
    sum_cost = 0
    
    for name, circuit in quantum_circuits.items():
        transpiler = Sabre(coupling_map, layout_strategy=mapping_strategy)
        
        # create pass manager and append transformation pass
        pass_manager = PassManager()
        pass_manager.append(transpiler)

        # run transformation pass
        mapped_qc = pass_manager.run(circuit)

        cost = get_circuit_cost(mapped_qc)
        results.append(cost)
        sum_cost = sum_cost+cost
    
    return [sum_cost,results]   

if __name__=="__main__":
    main()