# https://github.com/ejwilson3/approximate-circuits/blob/34e42f349631a5f0793f031aa1053bcd61720cca/run_circuits.py
import qiskit
import sys
from qiskit.providers.aer import noise

# Some numpy and qiskit combinations send out a ridiculous number of deprecation warnings
# import warnings
# warnings.filterwarnings('ignore')

if len(sys.argv) < 4:
    print("Missing one of: reference QASM, synthesis directory, or number of synthesis files to read.")
    exit()

################################################################################
# Most editing will need to be done here
simulated = True
shots = 1000
# optimization level
ol = 1
qiskit.IBMQ.load_account()
provider = qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
device = provider.get_backend('ibmq_5_yorktown')
################################################################################

reference = qiskit.QuantumCircuit.from_qasm_file(sys.argv[1])
syn_dir = sys.argv[2]
n_files = int(sys.argv[3])
nq = reference.num_qubits

################################################################################
# If the reference circuit is measured, comment this section.
cr = qiskit.ClassicalRegister(nq)
reference.add_register(cr)
for j in range(nq):
    reference.measure(j, cr[j])
################################################################################

properties = device.properties()
coupling_map = device.configuration().coupling_map
noise_model = noise.NoiseModel.from_backend(properties)
basis_gates = noise_model.basis_gates
simulator = qiskit.Aer.get_backend('qasm_simulator')

# Generate the file names
base = syn_dir + "/" + syn_dir
tail = ".qasm"

# Store the circuits themselves
circuits = []
# Store the CNOT depth of each circuit
count = []
fullcount = 0
refcount = 0
# Store the completed circuits; the keys are the CNOT depth.
completed = {}
# Store the results of the circuits; the keys are the CNOT depth.
result_dict = {}

skipped = 0
for i in range(0, n_files+1):
    filename = base + str(i) + tail
    circ = qiskit.QuantumCircuit.from_qasm_file(filename)
    if(circ.num_qubits != nq):
        # This circuit has too few (or too many) qubits and won't be comparable by most metrics. QFast will sometimes create smaller circuits.
        skipped += 1
        continue
    circ2 = qiskit.QuantumCircuit(nq)
    circuits.append(circ2)
    circuits[i-skipped].extend(circ)
    # This circuits from QSearch and QFast don't come with measurement
    cr = qiskit.ClassicalRegister(nq)
    circuits[i-skipped].add_register(cr)
    for j in range(nq):
        circuits[i-skipped].measure(j, cr[j])

# This is the circuit normally output by the synthesizer.
full_circuit = circuits.pop()

# Transpile the approximate circuits. It may fail here, if the transpilation
# times out. You can try again, or lower the optimization level if this happens
# too often.    
for idx, circ in enumerate(circuits):
    tcirc = qiskit.compiler.transpile(circ, backend=device, basis_gates=basis_gates, optimization_level=ol)
    qobj = qiskit.compiler.assemble(tcirc, shots=shots)
    circuits[idx] = qobj

tcirc = qiskit.compiler.transpile(full_circuit, backend=device, optimization_level=ol, basis_gates=basis_gates)
qobj1 = qiskit.compiler.assemble(tcirc, shots=shots)

tcirc = qiskit.compiler.transpile(reference, backend=device, optimization_level=ol, basis_gates=basis_gates)
qobj2 = qiskit.compiler.assemble(tcirc, shots=shots)

# Actually counting the number of CNOTs; transpilation can add some.
for circ in circuits:
    n_cx = 0
    for gate in circ.to_dict()['experiments'][0]['instructions']:
        if(len(gate['qubits']) > 1):
            n_cx += 1
    count.append(n_cx)

for gate in qobj1.to_dict()['experiments'][0]['instructions']:
    if(len(gate['qubits']) > 1):
        fullcount += 1

for gate in qobj2.to_dict()['experiments'][0]['instructions']:
    if(len(gate['qubits']) > 1):
        refcount += 1


for key in range(min(count), max(count) + 1):
    completed[key] = []
    
# Here actually run the circuits
if simulated:
    for i, depth in enumerate(count):
        completed[depth].append(simulator.run(circuits[i], noise_model=noise_model))
    full_completed = simulator.run(qobj1, noise_model=noise_model)
    ref_completed = simulator.run(qobj2, noise_model=noise_model)
else:
    for i, depth in enumerate(count):
        completed[depth].append(device.run(circuits[i]))
    full_completed = device.run(qobj1)
    ref_completed = device.run(qobj2)

# Store the job IDs if run on a physical machine.
if not simulated:
    j_dict = {}
    for key in completed:
        j_dict[key] = []
        for item in completed[key]:
            j_dict[key].append(item.job_id())
    newfile = open(syn_dir + "/" + syn_dir + ".backupJobIDs.txt", "w")
    newfile.write(str(j_dict) + "\n" + str(full_completed.job_id()) + "\n" + str(ref_completed.job_id()) + "\n")
    newfile.close()

# This will be run on the simulator to compare. This is how the JS score is generated; it is compared to the results of this circuit run without noise.
perfect = simulator.run(qobj2)

# Put the counts in their own dictionaries
for key in completed:
    result_dict[key] = []
for key in completed:
    for res in completed[key]:
        if res != []:
            result_dict[key].append(res.result().get_counts())
result_full = full_completed.result().get_counts()
result_ref = ref_completed.result().get_counts()
result_perf = perfect.result().get_counts()

# Save the counts to be used later.
newfile = open(syn_dir + "/" + syn_dir + ".counts.txt", "w")
newfile.write(str(result_dict) + "\n" + str(fullcount) + "\n" + \
              str(result_full) + "\n" + str(refcount) + "\n" + str(result_ref) \
               + "\n" + str(result_perf))
newfile.close()
