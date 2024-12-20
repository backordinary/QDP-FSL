# https://github.com/teaguetomesh/c2qa-modular-qc/blob/660632a58d9bb369283e0c76b07c61de7b70c7ae/test.py
import networkx as nx
import qiskit
import random

from qiskit_helper_functions.benchmarks import generate_circ

import arquin

if __name__ == "__main__":
    num_modules = 3
    module_size = 4
    # Last qubit of module i is connected to first qubit of module i+1
    global_edges = [[[i, module_size - 1], [(i + 1) % num_modules, 0]] for i in range(num_modules)]
    module_graph = nx.cycle_graph(module_size)
    device = arquin.Device(
        global_edges=global_edges, module_graphs=[module_graph for _ in range(num_modules)]
    )
    device.plot(save_dir="figures")

    seed = random.randint(0,int(1e8))
    seed = 14737345
    print("seed = {}".format(seed))
    circuit = generate_circ(
        num_qubits=device.size,
        depth=device.size,
        circuit_type="random",
        reg_name="q",
        connected_only=True,
        seed=seed,
    )

    coupling_map = arquin.converters.edges_to_coupling_map(device.fine_graph.edges)
    transpiled_circuit = qiskit.compiler.transpile(
        circuit, coupling_map=coupling_map, layout_method="sabre", routing_method="sabre"
    )
    print("Qiskit depth %d --> %d" % (circuit.depth(), transpiled_circuit.depth()))

    compiler = arquin.ModularCompiler(
        circuit=circuit, circuit_name="random", device=device, device_name="ring"
    )
    compiler.run(visualize=True)
