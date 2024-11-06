# https://github.com/jamesgopsill/ICED21-Quantum-Design/blob/bbb3d60639f0dbb81aa18165647eb4a0769d1a26/qasm/run.py
import circuits # import local file
from qiskit import transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
import itertools
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# Uncomment a circuit
	circuit = circuits.no_overlap()
	#circuit = circuits.no_1_1_2_tiles()
	#circuit = circuits.no_1_1_1_tile()
	#circuit = circuits.combined_constraints_v1()
	#circuit = circuits.constraint_free()
	#circuit = circuits.combined_constraints_v2()

	# Draw the quantum circuit
	circuit.draw("mpl", plot_barriers=False)
	
	# Create a simulator instance
	sim = QasmSimulator()

	# Create compiled
	compiled = transpile(circuit, sim)

	# Run the study
	study = sim.run(compiled, shots=1024)
	results = study.result()
	counts = results.get_counts(compiled)

	# Generate all the binary sequences for n bits
	binary_sequences = ["".join(seq) for seq in itertools.product("01", repeat=4)]

	complete_results = {}
	for s in binary_sequences:
		complete_results[s] = 0.

	for k, v in counts.items():
		complete_results[k] += v

	plot_histogram(complete_results)
	plt.tight_layout()

	# Show the plots
	plt.show()
