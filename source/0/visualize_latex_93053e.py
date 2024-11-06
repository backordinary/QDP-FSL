# https://github.com/parasol4791/quantumComp/blob/101be82cc0cf0baa617cae947484534eb71e22fd/visualize_latex.py
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import array_to_latex

import pylatexenc  # this is the package they say is needed for displaying Latex images, plus Pillow.  Alas, it does not help on PyCharm

sim = Aer.get_backend('qasm_simulator')

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)
qc.save_unitary()  # saves matrix
result = sim.run(qc).result()
unitary = result.get_unitary()
print(unitary)
u = array_to_latex(unitary, source=False, prefix="\\text{Circuit = } ")  # source=True means Latex text output
print(u)
qc.draw('latex')


# Conversion from Latex to PDF, auto displaying PDF in a browser
from pdflatex import PDFLaTeX
import subprocess
import os

dir_ = '/home/dvk11/Documents/latex_file'
fname_tex = dir_ + '.tex'
fname_pdf = dir_ + '.pdf'
# .tex file is created
with open(fname_tex, 'w') as f:
    f.write(u)
# Convert .tex to PDF
# So far, Qiskit inputs did not work with pdf converted - it thinks the files are corrupted. Either a different converter is needed, or some packages are missing.
subprocess.run(['pdflatex', '-interaction=nonstopmode', fname_tex])
if not os.path.exists(fname_pdf):
    print('No PDF file')
# Display PDF file in web browser
import webbrowser
webbrowser.open(fname_tex)
