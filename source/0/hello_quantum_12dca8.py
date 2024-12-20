# https://github.com/ctuning/ck-qiskit/blob/9a606a1ad9497d142486b788074827a3e6aeab11/program/qiskit-backends/hello_quantum.py
"""
Example used in the readme. In this example a Bell state is made
"""
import sys
import os
from pprint import pprint
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram, QISKitError, available_backends, register

try:
    import Qconfig
    register(Qconfig.APItoken, Qconfig.config["url"], verify=False,
                      hub=Qconfig.config["hub"],
                      group=Qconfig.config["group"],
                      project=Qconfig.config["project"])
except:
    offline = True
    print("""WARNING: There's no connection with IBMQuantumExperience servers.
             cannot test I/O intesive tasks, will only test CPU intensive tasks
             running the jobs in the local simulator""")

# Running this block before registering quietly returns a list of local-only simulators
#
print("The backends available for use are:")
backends = available_backends()
pprint(backends)
print("\n")

if 'CK_IBM_BACKEND' in os.environ:
   backend = os.environ['CK_IBM_BACKEND']
if backend not in backends:
   print("Your choice '%s' was not available, so picking a random one for you..." % backend)
   backend = backends[0]
   print("Picked '%s' backend!" % backend)
try:
    # Create a QuantumProgram object instance.
    Q_program = QuantumProgram()

    # Create a Quantum Register called "qr" with 2 qubits.
    qr = Q_program.create_quantum_register("qr", 2)
    # Create a Classical Register called "cr" with 2 bits.
    cr = Q_program.create_classical_register("cr", 2)
    # Create a Quantum Circuit called "qc". involving the Quantum Register "qr"
    # and the Classical Register "cr".
    qc = Q_program.create_circuit("bell", [qr], [cr])

    # Add the H gate in the Qubit 0, putting this qubit in superposition.
    qc.h(qr[0])
    # Add the CX gate on control qubit 0 and target qubit 1, putting 
    # the qubits in a Bell state
    qc.cx(qr[0], qr[1])

    # Add a Measure gate to see the state.
    qc.measure(qr, cr)

    # Compile and execute the Quantum Program in the local_qasm_simulator.
    result = Q_program.execute(["bell"], backend=backend, shots=1024, seed=1)

    # Show the results.
    print(result)
    print(result.get_data("bell"))

except QISKitError as ex:
    print('There was an error in the circuit!. Error = {}'.format(ex))
