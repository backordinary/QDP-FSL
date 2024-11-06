# https://github.com/jamesgopsill/ICED21-Quantum-Design/blob/bbb3d60639f0dbb81aa18165647eb4a0769d1a26/statevectors/run.py
import circuits # import local file
from qiskit import transpile
from qiskit.providers.aer import StatevectorSimulator
from qiskit.visualization import plot_histogram
import itertools
import matplotlib.pyplot as plt

if __name__ == "__main__":

	#circuit = circuits.no_overlap()
	#circuit = circuits.no_1_1_2_tiles()
	#circuit = circuits.no_1_1_1_tile()
	circuit = circuits.combined_constraints_v1()
	#circuit = circuits.constraint_free()
	#circuit = circuits.combined_constraints_v2()

	# Draw the quantum circuit
	circuit.draw("mpl", plot_barriers=False)

	# Use the StatevectorSimulator

	sim = StatevectorSimulator()

	circuit = transpile(circuit, sim)

	result = sim.run(circuit).result()

	# Retrieve the probabilities (N.b. it won't return null values)

	sequence_probabilities = result.get_counts(circuit)

	# Generate all the binary sequences for n bits
	binary_sequences = ["".join(seq) for seq in itertools.product("01", repeat=4)]

	# Map (combine) the probabilities as we're only interested in the 4 bits sequences
	final_results = {}
	for s in binary_sequences:
		final_results[s] = 0.

	for k, v in sequence_probabilities.items():
		reverse = k[::-1]
		final_results[reverse[0:4]] += v
		
	for k, v in final_results.items():
		print(k, v)

	# Plot the histogram
	plot_histogram(final_results)
	plt.tight_layout()

	# Show the plots
	plt.show()