# https://github.com/lenn-arts/qiskit2tex/blob/fbfeae6a7d8b31470dd99e678bdc1c5e7416b21e/qiskit2tex.py
#!/usr/bin/python
#
# QISKIT2TEX
# turn quantum circuits defined in qiskit (.py) files into tex code (.tex) files,
# ready to be included in already setup latex documents (needs an import of the necessary packages in the preambel.tex)
#
# Author: Lennart Schulze
# Version: 2020-12-03

### imports
import sys # to enable command line arguments
import os # to work with path names
import importlib  # to import source python file
from qiskit import QuantumCircuit # to render to qasm

### read source file
input_file_string = sys.argv[1] # file location (without .py ending) passed as argument in command line (arg0 is this script)
# need (empty) __init__.py in source code folder
exec(open(input_file_string).read())  # import circuit (circuit should always be named "qc")

### output as tex
begin = "\\begin{equation*}" # marks begin of useful code of output
end = "\\end{equation*}" # marks end of useful code of output
tex_code = qc.draw("latex_source") # generate overloaded output
tex_code = tex_code[tex_code.find(begin):tex_code.find(end)+len(end)] # cut to useful part based on markers
# print(tex_code)

### save tex in file
if len(sys.argv) > 2:
    output_dir = sys.argv[2]
else:
    output_dir = os.path.dirname(os.path.realpath(__file__))
output_file_string = os.path.join(output_dir, f"{input_file_string.split('.py')[0]}.tex")
f = open(output_file_string, "w") # create or overwrite existing file
f.write(tex_code) # insert tex code
f.close()

# exit()
