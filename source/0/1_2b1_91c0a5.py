# https://github.com/tokyotech-toyota/quantum_computing/blob/f60b1c7197a1f15d91d07c9b435a0ec44847f10b/1%2B1/1%2B1.py
#!/usr/bin/env -S python3 -W ignore
# coding: utf-8
################################################################################
from __future__ import print_function
# import tbvaccine as tb; tb.add_hook(isolate=False, show_vars=False)
from qiskit import (
    IBMQ,
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    Aer,
    execute,
    transpile
)
################################################################################


##################################################
################### Token auth ###################
##################################################
token_path = "../IBMQ_token"
with open(token_path, "r") as file:
    token = file.read()
IBMQ.save_account(token, overwrite=True)


################################################################################
##################################### Main #####################################
################################################################################
def main():
    ##################################################
    ################## Make circuit ##################
    ##################################################
    circ = QuantumCircuit(4, 4)
    circ.h(0)
    circ.h(1)
    circ.barrier()
    circ.ccx(0, 1, 2)
    circ.cx(0, 3)
    circ.cx(1, 3)
    circ.barrier()
    circ.measure([0,1,2,3], [3,2,1,0])

    ##################################################
    ################# Print circuit ##################
    ##################################################
    circ_trans = transpile(circ)
    # circ_trans.draw(output="mpl", filename="./circuit.png", plot_barriers=True)
    style = {"dpi":200, "showindex":True, "cregbundle":True, "margin":[1.5,1,0.5,1]}
    circ_trans.draw(output="mpl", filename="./circuit.png",
                    style=style, initial_state=True, plot_barriers=True, fold=20)

    ##################################################
    #################### Run job #####################
    ##################################################
    backend = Aer.get_backend("qasm_simulator")
    shots = 4096
    job = execute(circ_trans, backend=backend, shots=shots)

    ##################################################
    ################## Print result ##################
    ##################################################
    result = job.result()
    result_dict = result.get_counts(circ_trans)
    ########## Export result ##########
    with open("result.ssv", "w") as file:
        file.write("# qubit probability\n")
        for i in result_dict:
            file.write("{0} {1:.8E}\n".format(i, result_dict[i]/shots))

            
################################################################################
################################################################################
################################################################################
if __name__ == "__main__":
    main()
