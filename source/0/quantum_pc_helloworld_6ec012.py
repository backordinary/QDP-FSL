# https://github.com/frainfreeze/studying/blob/05f71390417acef431cbd434bde8765fb5c1abe0/home/old/projects/quantum_pc_helloWorld.py
import sys
import os
from pprint import pprint

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram, QISKitError, RegisterSizeError

# - [QISKit](https://github.com/QISKit/qiskit-sdk-py) -> Quantum Software Development Kit
# - [QISKit Documentation](https://www.qiskit.org/documentation/) -> docs for Python SDK version for working with OpenQASM and the IBM Q experience (QX)
# - [Beginners Guide](https://quantumexperience.ng.bluemix.net/qx/tutorial?sectionId=beginners-guide&page=introduction) | [Full User Guide](https://quantumexperience.ng.bluemix.net/qx/tutorial?sectionId=full-user-guide&page=introduction) -> IBM Q Experience Documentation 

# Create a QuantumProgram object instance.
Q_program = QuantumProgram()
try:
    import Qconfig
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"], verify=False,
                      hub=Qconfig.config["hub"],
                      group=Qconfig.config["group"],
                      project=Qconfig.config["project"])
except:
    offline = True
    print("""WARNING: There's no connection with IBMQuantumExperience servers.
             cannot test I/O intesive tasks, will only test CPU intensive tasks
             running the jobs in the local simulator""")

print("The backends available for use are:")
pprint(Q_program.available_backends())
print("\n")
backend = 'local_qasm_simulator'
try:
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
except RegisterSizeError as ex:
    print('Error in the number of registers!. Error = {}'.format(ex))
