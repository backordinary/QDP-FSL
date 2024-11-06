# https://github.com/dclarino/ngc-dd/blob/90fd3f911f3eca1664009f9c522386750e26f9ee/src/read_real.py
from qiskit import *
from mqt.qcec import *

def readRealFile(filename):
    # Using readline()
    file1 = open(filename, 'r')
    keep_reading = 1
    count = 0
    qc_args = { "state" : "header" }
    while keep_reading:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty
        # end of file is reached
        if not line:
            keep_reading = 0
            break
        elif not line[0] == "#":
            if (qc_args["state"] == "header"):
                qc_args = readHeaderLine(line, qc_args)
            elif(qc_args["state"] == "body"):
                qc_args = readBodyLine(line, qc_args)
                
        #print("Line{}: {}".format(count, line.strip()))
        
    file1.close()

    return qc_args

def readHeaderLine(line,qc_args):
    if(line[0] != "."):
        raise Exception("not properly formatted header!")
    elif(".begin" in line):
        qc_args["state"] = "body"
    else:
        if (".variables" in line):
            var_list = line.split()[1:]
            qc_args["variables"] = {}
            count = 0;
            for variable in var_list:
                qc_args["variables"][variable] = count
                count+=1
    return qc_args

def readBodyLine(line,qc_args):
    if(line[0] == 't'):
        num_vars = int(line[1])

        split_line   = line.split()
        if(len(split_line) > 2):
            i_vars       = split_line[1:len(split_line)-1]
            o_vars       = [split_line[len(split_line)-1]]
        else:
            i_vars       = [split_line[1]]
            o_vars       = [split_line[1]]

        i_vars_int    = []
        o_vars_int    = []
        count        = 0
        for i_var in i_vars:
            i_vars_int.append(qc_args["variables"][i_var])
        for o_var in o_vars:
            o_vars_int.append(qc_args["variables"][o_var])

        if not "gates" in qc_args :
            qc_args["gates"] = []
        if(num_vars == 1):
            qc_args["gates"].append({"type":"CNOT", "i_vars_int":i_vars_int, "o_vars_int":o_vars_int})
        else:
            qc_args["gates"].append({"type":"MCT", "i_vars_int":i_vars_int, "o_vars_int":o_vars_int})

    return qc_args
