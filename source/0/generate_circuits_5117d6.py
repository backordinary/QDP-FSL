# https://github.com/ejwilson3/approximate-circuits/blob/0d397aca627f169848ed15b775f00ccd8027ffb5/generate_circuits.py
import numpy as np
import qiskit
import sys
# my numpy and qiskit make too many warnings
# import warnings
# warnings.filterwarnings('ignore')

if len(sys.argv) < 2:
    print("No QASM file")
    exit()
if len(sys.argv) < 3:
    output = "output"
else:
    output = sys.argv[2]

filename = sys.argv[1]
circ = qiskit.QuantumCircuit.from_qasm_file(filename)

nq = circ.num_qubits
matrix = qiskit.quantum_info.Operator(circ).data
if nq < 4:
    import qsearch
    qsproj = qsearch.Project(output)
    qsproj.add_compilation(output, matrix, write_location=(output +"/" + output))
    qsproj.run()
else:
    import qfast
    from os import mkdir
    try:
        mkdir(output)
    except FileExistsError:
        pass
    qasm_iteration = 0
    
    def print_potentials(gate_list):
        global qasm_iteration
        tool = "QSearchTool"
        combiner = "NaiveCombiner"
        instantiater = qfast.Instantiater(tool)
        qasm_list = instantiater.instantiate(gate_list)
        combiner = qfast.plugins.get_combiner(combiner)()
        qasm_out = combiner.combine(qasm_list)
        newfile = open(output + "/" + output + str(qasm_iteration) + ".qasm", "w")
        qasm_iteration += 1
        newfile.write(str(qasm_out))
        newfile.close()

    kwdict = {}
    kwdict["partial_solution_callback"] = print_potentials
    final_circ = str(qfast.synthesize(matrix, intermediate_solution_callback=None, model_options=kwdict))
    final_file = open(output + "/" + output + str(qasm_iteration) + ".qasm", "w")
    final_file.write(final_circ)
    final_file.close()
