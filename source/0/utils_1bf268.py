# https://github.com/snuphya/GST_qiskit/blob/9ad98227e7e476158802e32e7663c8f64d59585d/utils/utils.py
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICPHASE
import pygsti
from qiskit import IBMQ
import numpy as np
from qiskit import *
import shutil
import os

def clear_and_mk_files(GST_type, max_max_length):
    shutil.rmtree(GST_type)
    print("{} folder is removed".format(GST_type))
    
    os.mkdir(GST_type)
    if GST_type == "smq1Q_XYI":
        exp_design = smq1Q_XYI.get_gst_experiment_design(max_max_length)
        pygsti.io.write_empty_protocol_data(exp_design, GST_type, clobber_ok=True)
    elif GST_type == "smq2Q_XYICPHASE":
        exp_design = smq2Q_XYICPHASE.get_gst_experiment_design(max_max_length)
        pygsti.io.write_empty_protocol_data(exp_design, GST_type, clobber_ok=True)
    else:
        NotImplementedError("{} is not implemented".format(GST_type))
    print("{} folder with empty dataset.txt is reconstructed".format(GST_type))

def get_exp_design(max_max_length, GST_type):
    if GST_type == "smq1Q_XYI":
        exp_design = smq1Q_XYI.get_gst_experiment_design(max_max_length = max_max_length)
    elif GST_type == "smq2Q_XYICPHASE":
        exp_design = smq2Q_XYICPHASE.get_gst_experiment_design(max_max_length = max_max_length)
    else:
        NotImplementedError("{} is not implemented".format(GST_type))
    return exp_design

def get_circs_from_exp_design(exp_design, n_qubit):
    circs = []
    # layers 라는 이름이 부적합한듯.
    for layers in exp_design.all_circuits_needing_data:
        circs.append(get_circ_from_layers(layers, n_qubit))
    return circs


def get_circ_from_layers(layers, n_qubit):
    circ = QuantumCircuit(n_qubit, n_qubit)
    for gates in layers.layertup:
        if gates==None:
            circ.id()
        # Be careful of rotation direction
        elif gates.name=='Gxpi2':
            circ.rx(np.pi/2, gates.qubits)
        elif gates.name=='Gypi2':
            circ.ry(np.pi/2, gates.qubits)
        elif gates.name=='Gcphase':
            circ.cz(gates.qubits[0], gates.qubits[1])
        else:
            NotImplementedError("## Gate {} is not implemented in get circ_from layer".format(gates.name))
    circ.measure_all()
    return circ

def save_dataset(exp_design, meas_dict, meas_key, GST_type):
    with open('{}/data/dataset.txt'.format(GST_type), 'w') as f:
        
        f.write("## Columns = ")
        for idx, k in enumerate(meas_key):
            f.write("{} count".format(k), )
            if idx != len(meas_key)-1:
                f.write(", ")
        f.write('\n')
        for idx, layers in enumerate(exp_design.all_circuits_needing_data):
            line = layers.str
            for key in meas_key:
                line += '\t'
                line += str(meas_dict[idx][key]) if key in meas_dict[idx] else str(0)
            f.write(line + '\n')