# https://github.com/quantum-kittens/quantum-prompt-generator/blob/4bfc8178530d776dd4f0d84583b8203b13ca4871/app.py
import re
from collections import defaultdict, deque

from flask import Flask, Markup, render_template, request
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy

from promptgen import *

###############################################################################
# To run this on an actual quantum device, uncomment the following lines, and
# comment out `backend = BasicAer.get_backend("qasm_simulator")`. Then
# uncomment the lines in the `get class` section of the character generator
# as described further down.
###############################################################################

# IBMQ.save_account("ibmq-token-goes-here")
# provider = IBMQ.load_account()
# backend = least_busy(
#     provider.backends(
#         filters=lambda x: x.configuration().n_qubits >= 5
#         and not x.configuration().simulator
#         and x.status().operational == True
#     )
# )

backend = BasicAer.get_backend("qasm_simulator")


app = Flask(__name__)


# Prompt Generator
pg = PromptGenerator


@app.route("/")
def index():
    #return render_template("index.html")
    ## get prompt
    prompt = pg().generate_prompt()

    return render_template(
        "index.html",
        backstory=prompt,
        answered=True,
        )


if __name__ == "__main__":
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
