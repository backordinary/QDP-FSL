# https://github.com/lucasponteslpa/quantum-classifier/blob/3f97717ebe61ba93c02ce655d72805df88e085e5/exp.py
import qiskit
from minimum import MinimumQClassifier
from dataset import ProcessData
from general_preparation import QClassifier
from qiskit.visualization import plot_histogram, circuit_drawer
from pdb import set_trace
import numpy as np

def print_state(cq):
    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(cq, backend)
    result = job.result()
    print(result.get_statevector())

def print_res(cq):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    results = qiskit.execute(cq, backend=backend, shots=1024).result()
    answer = results.get_counts()
    print(answer)

def run_classifier(params):
    dataexp = ProcessData()
    if(params["show_data"]):
        dataexp.show_data()
    if(params["circuit"]=="general"):

        if(params["circuit"]==4):
            qclass = QClassifier(dataexp.norm[48:52,:],dataexp.Y[48:52], dataexp.norm[32,:])
        elif(params["circuit"]==8):
            qclass = QClassifier(dataexp.norm[46:54,:],dataexp.Y[46:54], dataexp.norm[32,:])
        elif(params["circuit"]==16):
            qclass = QClassifier(dataexp.norm[42:58,:],dataexp.Y[42:58], dataexp.norm[32,:])
        elif(params["circuit"]==64):
            qclass = QClassifier(np.append(dataexp.norm[0:32,:],dataexp.norm[55:87,:]).reshape(64,2),np.append(dataexp.Y[0:32],dataexp.Y[55:87]), dataexp.norm[32,:])
        else:
            qclass = QClassifier(np.array([[0.0,1.0],[0.789,0.615]]), np.array([0,1]), np.array([-0.549,0.836]))
        
        qclass.preparation()
        if(params["draw"]):
            circuit_drawer(qclass.circuito,filename='qclass.tex',output='latex_source')
        #print(qclass.circuito)
        print_res(qclass.circuito)
    #set_trace()
    else:

        mini = MinimumQClassifier()
        print(mini.circuito)
        if(params["draw"]):
            circuit_drawer(mini.circuito,filename='minimum.tex',output='latex_source')
        print_res(mini.circuito)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--circuit', type=str, help='Define what circuit will be used')
    parser.add_argument('--num_exemples', type=int,default=2, help='Number of training exemples')
    parser.add_argument('--test_index', type=str, help='Index of test data')
    parser.add_argument('--show_data', type=bool, default=False, help='Plot the data distribution')
    parser.add_argument('--draw', type=bool, default=False, help='Write a tex file with the circuit scheme')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)
    run_classifier(params)
