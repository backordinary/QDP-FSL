# https://github.com/Harkirat9711/Qiskit_Demo/blob/5d023299013b4a93928cbeafd948cb7faeae5c7a/helloquantum.py
""" ASCII encoding of "Hello, World!" requires over 100 bits, and
therefore over 100 qubits. Current quantum devices are not yet large
enough for the job.


However, two ASCII characters require only 16 (qu)bits. Writing an
emoticon like ;) can therefore be done using ibmqx5 backend.""


##;)='0011101100101001’


8)='0011100000101001’. 

The Demonstration will be pictured as a Diagram"""

from qiskit import QuantumProgram
import Qconfig

qp = QuantumProgram()
qp.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# set up registers and program
qr = qp.create_quantum_register('qr', 16)
cr = qp.create_classical_register('cr', 16)
qc= qp.create_circuit('Circuit', [qr], [cr])

Q_SPECS = {
    'circuits': [{
        'name': 'Circuit',
        'quantum_registers': [{
            'name': 'qr',
            'size': 16
        }],
        'classical_registers': [{
            'name': 'cr',
            'size': 16
        }]}],
}

def show_image(img_name):
    
    return Image(filename=os.path.join("..", "images", "intro_img", img_name))


# rightmost eight (qu)bits have ')' = 00101001''

qp = QuantumProgram(specs=Q_SPECS)

# get the circuit by Name
circuit = qp.get_circuit('Circuit')

# get the Quantum Register by Name
quantum_r = qp.get_quantum_register('qr')

# get the Classical Register by Name
classical_r = qp.get_classical_register('cr')

circuit.x(quantum_r[0])
circuit.x(quantum_r[3])
circuit.x(quantum_r[5])



# second eight (qu)bits have superposition of
# '8' = 00111000
# ';' = 00111011
# these differ only on the rightmost two bits

circuit.h(quantum_r[9])
circuit.cx(quantum_r[9],quantum_r[8])
circuit.x(quantum_r[11])
circuit.x(quantum_r[12])
circuit.x(quantum_r[13])



# measure
for j in range(16):
   circuit.measure(quantum_r[j], classical_r[j])

qp.get_circuit_names()

QASM_source = qp.get_qasm('Circuit')

print(QASM_source)
import os
import shutil
from qiskit.tools.visualization import latex_drawer
import pdf2image

def circuitImage(circuit, basis="u1,u2,u3,cx"):
    """Obtain the circuit in image format
    Note: Requires pdflatex installed (to compile Latex)
    Note: Required pdf2image Python package (to display pdf as image)
    """
    filename='circuit'
    tmpdir='tmp/'
    shutil.rmtree(tmpdir)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    latex_drawer(circuit, tmpdir+filename+".tex", basis=basis)
    os.system("pdflatex -output-directory {} {}".format(tmpdir, filename+".tex"))
    images = pdf2image.convert_from_path(tmpdir+filename+".pdf")
    #shutil.rmtree(tmpdir)
    return images[0]




# run and get results

def circuitImage(circuit, basis="u1,u2,u3,cx,x,y,z,h,s,t,rx,ry,rz"):
    """Obtain the circuit in image format
    Note: Requires pdflatex installed (to compile Latex)
    Note: Required pdf2image Python package (to display pdf as image)
    """
    filename='circuit'
    tmpdir='tmp/'
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    latex_drawer(circuit, tmpdir+filename+".tex", basis=basis)
    os.system("pdflatex -output-directory {} {}".format(tmpdir, filename+".tex"))
    images = pdf2image.convert_from_path(tmpdir+filename+".pdf")
    return images[0]

print("Running Emoji: Hello World")
print("Please wait till the time the editor opens")
print("Please Close the Editor and wait for the Circuit Diagram PDF ")
results = qp.execute(["Circuit"], backend='local_qasm_simulator', max_credits=3, wait=10, timeout=240, shots=10)
stats = results.get_counts("Circuit")

import matplotlib.pyplot as plt
plt.rc('font', family='monospace')
for bitString in stats:
    char = chr(int( bitString[0:8] ,2)) # get string of the leftmost 8 bits and convert to an ASCII character
    char += chr(int( bitString[8:16] ,2)) # do the same for string of rightmost 8 bits, and add it to the previous character
    prob = stats[bitString] / 1024 # fraction of shots for which this result occurred
    # create plot with all characters on top of each other with alpha given by how often it turned up in the output
    plt.annotate( char, (0.5,0.5), va="center", ha="center", color = (0,0,0, prob ), size = 300)
    if (prob>0.05): # list prob and char for the dominant results (occurred for more than 5% of shots)
        print(str(prob)+"\t"+char)
plt.axis('off')
plt.show()

print("Creating The IBM quantum Circuit")
print("Soon you will have the circuit in PDF")
print("Logic By Harkirat")

basis="cx,x,y,z,h,s,t"
circuitImage(circuit, basis)
filename='circuit'
tmpdir='tmp/'
os.system("open {}/{}".format(tmpdir, filename+".pdf"))
backend = 'local_qasm_simulator' 
circuits = ['Circuit']  # Group of circuits to execute
qobj=qp.compile(circuits, backend) # Compile your program
result = qp.run(qobj, wait=2, timeout=240)







