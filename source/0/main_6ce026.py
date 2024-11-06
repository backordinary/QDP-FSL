# https://github.com/BankNatchapol/When-God-Gives-You-a-Life-Quote/blob/cc7e317f3d3873439aaa788c30d09fa4f0d1acfa/main.py
from qiskit import *
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider

from Quote2Img import convert
import config

import pandas as pd
import json

with open("files/quote_list", "r") as fp:
  randomquotes = json.load(fp)

q = QuantumRegister(8,'q')
c = ClassicalRegister(8,'c')
circuit = QuantumCircuit(q,c)
circuit.h(q) 
circuit.measure(q,c)

provider = AzureQuantumProvider(
  resource_id=config.resource_id,
  location=config.location
)

backend_name = 'quantinuum.hqs-lt-s1-sim'
backend = provider.get_backend(backend_name)
circuit = transpile(circuit, backend)

# cost = backend.estimate_cost(circuit, shots=2)

# print(f"Estimated cost: ${cost.estimated_total}")

job = backend.run(circuit, shots=2)
job_id = job.id()
print("Job id", job_id)
job_monitor(job)

result = job.result()
counts = result.get_counts()
number = int(list(counts.keys())[0] + list(counts.keys())[1], 2)
randomquote = randomquotes[number]

if len(randomquote) > 200:
  font_size = 32
else:
  font_size = 64

img=convert(
	quote=randomquote,
	fg="white",
	image="files/god.png",
	border_color="black",
	font_size=font_size,
	font_file=None,
	width=1080,
	height=1080)

img.show()
