# https://github.com/mdschatz/qiskit/blob/929dd2dc5a68b69a06ea474998e9f973614bfec2/grovers_search.py
import ast, click
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compiler, Aer
from qiskit.tools import visualization


# From Section 6.1 of Intro to Quantum Computing, Without the Physics
def input_state(circuit, f_in, f_out, n):
	"""(n+1)-qubit input state for Grover search."""
	for j in range(n):
		circuit.h(f_in[j])
	circuit.x(f_out)
	circuit.h(f_out)


# From Section 6.2 of Intro to Quantum Computing, Without the Physics
def black_box_u_f(circuit, f_in, f_out, aux, n, exactly_1_3_sat_formula):
	num_clauses = len(exactly_1_3_sat_formula)
	if (num_clauses > 3):
		raise ValueError("We_only_allow_at_most_3_clauses")
	for (k, clause) in enumerate(exactly_1_3_sat_formula):
		for literal in clause:
			if literal > 0:
				circuit.cx(f_in[literal - 1], aux[k])
			else:
				circuit.x(f_in[-literal - 1])
				circuit.cx(f_in[-literal - 1], aux[k])

		circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
		circuit.ccx(f_in[2], aux[num_clauses], aux[k])

		circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
		for literal in clause:
			if literal < 0:
				circuit.x(f_in[-literal - 1])
	if num_clauses == 1:
		circuit.cx(aux[0], f_out[0])
	elif num_clauses == 2:
		circuit.ccx(aux[0], aux[1], f_out[0])
	elif num_clauses == 3:
		circuit.ccx(aux[0], aux[1], aux[num_clauses])
		circuit.ccx(aux[2], aux[num_clauses], f_out[0])
		circuit.ccx(aux[0], aux[1], aux[num_clauses])

	for (k, clause) in enumerate(exactly_1_3_sat_formula):
		for literal in clause:
			if literal > 0:
				circuit.cx(f_in[literal - 1], aux[k])
			else:
				circuit.x(f_in[-literal - 1])
				circuit.cx(f_in[-literal - 1], aux[k])
		circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
		circuit.ccx(f_in[2], aux[num_clauses], aux[k])
		circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
		for literal in clause:
			if literal < 0:
				circuit.x(f_in[-literal-1])


# From Section 6.3 of Intro to Quantum Computing, Without the Physics
def inversion_about_average(circuit, f_in, n):
	for j in range(n):
		circuit.h(f_in[j])
	for j in range(n):
		circuit.x(f_in[j])
	n_controlled_Z(circuit, [f_in[j] for j in range(n - 1)], f_in[n-1])
	for j in range(n):
		circuit.x(f_in[j])
	for j in range(n):
		circuit.h(f_in[j])


# From Section 6.3 of Intro to Quantum Computing, Without the Physics
def n_controlled_Z(circuit, controls, target):
	if (len(controls) > 2):
		raise ValueError("The_controlled_Z_with_more_than_2_controls_is_not_implemented")
	elif (len(controls) == 1):
		circuit.h(target)
		circuit.cx(controls[0], target)
		circuit.h(target)
	elif (len(controls) == 2):
		circuit.h(target)
		circuit.ccx(controls[0], controls[1], target)
		circuit.h(target)


# From Section 6.4 of Intro to Quantum Computing, Without the Physics
def grovers_search_alg(n, exactly_1_3_sat_formula, k_iter=2, exp_name="Grover"):
	assert k_iter == 2, "Implementation has only been verified on k_iter==2"

	f_in = QuantumRegister(n)
	f_out = QuantumRegister(n)
	aux = QuantumRegister(len(exactly_1_3_sat_formula) + 1)
	ans = ClassicalRegister(n)
	qc = QuantumCircuit(f_in, f_out, aux, ans, name=exp_name)

	input_state(qc, f_in, f_out, n)

	# Apply two full iterations
	black_box_u_f(qc, f_in, f_out, aux, n, exactly_1_3_sat_formula)
	inversion_about_average(qc, f_in, n)
	black_box_u_f(qc, f_in, f_out, aux, n, exactly_1_3_sat_formula)
	inversion_about_average(qc, f_in, n)
	# Measure the output register in the computational basis
	for j in range(n):
		qc.measure(f_in[j], ans[j])
	return qc


class Exactly_1_3_SAT_Formula(click.Option):
	def type_cast_value(self, ctx, value):
		try:
			opt = ast.literal_eval(value)
			assert isinstance(opt, list)
			for i in opt:
				assert isinstance(i, list) and all(isinstance(x, int) for x in i)
			return opt
		except:
			raise click.BadParameter(value)


@click.command()
@click.option("--n_literals", type=int, help="Number of literals in 1-3 SAT formula.", default=3)
@click.option("--exactly_1_3_sat_formula", cls=Exactly_1_3_SAT_Formula, help="Exact 1_3_SAT formula.", default="[[1, -2, 3], [-1, 2, 3], [-1, -2, -3]]")
def main(n_literals, exactly_1_3_sat_formula):
	assert n_literals <= 3 and len(exactly_1_3_sat_formula) <= 3, ("The number of literals and clauses must each satisfy n <= 3")
	qc = grovers_search_alg(n_literals, exactly_1_3_sat_formula)

	# Create an instance of the local quantum simulator
	quantum_simulator = Aer.get_backend('qasm_simulator')
	# Compile the object intO quantum object code
	# that can be executed on the simulator
	qobj = compiler.assemble(qc, quantum_simulator, shots=2048)
	# Execute and store the results. Note taht this could take some time
	job = quantum_simulator.run(qobj)
	result = job.result()
	# Get counts
	counts = result.get_counts("Grover")

	print('Observed measurement')
	print('string | count')
	for key in sorted(counts):
		print(' {:>5s}    {:d}'.format(key, counts[key]))

	# Plot histogram
	figure = visualization.plot_histogram(counts)
	print()
	figure.savefig('groverhist.png')
	print('Histogram saved as groverhist.png')


if __name__ == "__main__":
	main()