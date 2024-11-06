# https://github.com/Acdimy/CVBE/blob/903ca82abe1a135debcde08aa208db1f965254c5/V2/dev/main.py
from cvbe import *
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit import *
import dd.autoref as _bdd

def init_cir(circ, bdd):
    qubits_num = get_total_qubit_num(circ)
    vars_num = qubits_num
    var_list = []
    for i in range(3*vars_num):
        var_list.append('i'+str(i))
        bdd.declare('i'+str(i))
    gates = circ.data
    gates_layer_list = [[]]
    head_ptr = [0]*qubits_num
    curr_head = 0
    # for k in range(len(gates)):
    #     g = gates[k]
    #     gates_layer_list.append([g])
    for k in range(len(gates)):
        g = gates[k]
        q = [q.index for q in g[1]]
        if len(q) == 1:
            head_ptr[q[0]] += 1
        else:
            head_ptr[q[0]] = head_ptr[q[1]] = max(head_ptr[q[0]], head_ptr[q[1]]) + 1
        if head_ptr[q[0]] > curr_head:
            gates_layer_list.append([g])
            curr_head += 1
        else:
            gates_layer_list[head_ptr[q[0]]].append(g)
    return gates_layer_list,vars_num,var_list,bdd

def gen_cvbe_from_layer(gates_list, vars_num, var_list, curr_output, bdd):
    circ_cvbe = None
    uninvolved_q = [1]*vars_num
    bias_out = 2 if curr_output == 1 else 1
    for g in gates_list:
        nam = g[0].name
        q = [q.index for q in g[1]]
        if nam == 'cx': # qiskit feature
            tmp = q[0]
            q[0] = q[1]
            q[1] = tmp
        for idx in q:
            uninvolved_q[idx] = 0
        U = Operator(g[0]).data
        termlist = []
        for r in range(U.shape[0]):
            for c in range(U.shape[1]):
                if U[r][c] != 0:
                    ref = genRef(r,c,len(q), var_list, vars_num, q, curr_output, bdd)
                    termlist.append(Term(U[r][c], ref, bdd))
        cvbe = CVBE(termlist, bdd=bdd)
        if circ_cvbe == None:
            circ_cvbe = cvbe
        else:
            circ_cvbe = circ_cvbe.tensor(cvbe)
    identity_list = []
    for i in range(len(uninvolved_q)):
        if uninvolved_q[i] != 0:
            var_in = var_list[curr_output*vars_num+i]
            var_out = var_list[bias_out*vars_num+i]
            identity_list.append(bdd.add_expr(var_in + '<=>' + var_out))
    if len(identity_list) != 0:
        identity_cvbe = CVBE([Term(1, dd_ref_and(identity_list), bdd)], bdd=bdd)
        if circ_cvbe == None:
            circ_cvbe = identity_cvbe
        else:
            circ_cvbe = circ_cvbe.tensor(identity_cvbe)
    return circ_cvbe

std_layer = 0
def gen_cvbe(circ, bdd):
    gates_layer_list,vars_num,var_list,bdd = init_cir(circ, bdd)
    # print(gates_layer_list)
    if len(gates_layer_list) % 2 != std_layer:
        gates_layer_list.append([])
    # print(len(gates_layer_list))
    curr_output = 0
    circ_cvbe = None
    for ii,layer in enumerate(gates_layer_list[1:]):
        print(ii, 'layer')
        if ii != 0 and ii % 2 == 0:
            # print(len(circ_cvbe.term_list))
            circ_cvbe.regularize()
        cvbe = gen_cvbe_from_layer(layer, vars_num, var_list, curr_output, bdd)
        internal_vars = var_list[curr_output*vars_num:(curr_output+1)*vars_num]
        curr_output = 2 if curr_output == 1 else 1
        if circ_cvbe == None:
            circ_cvbe = cvbe
        else:
            circ_cvbe = circ_cvbe.sequential(cvbe, internal_vars)
        # print([circ_cvbe.term_list[i].ref.support for i in range(len(circ_cvbe.term_list))])
    circ_cvbe.regularize()
    return circ_cvbe

if __name__ == '__main__':
    bdd = _bdd.BDD()
    bdd.configure(reordering=True)
    path = '../benchmark/'

    # 一般测试
    name0 = 'RandomClifford_q15_0_0.qasm'
    name1 = 'qft_test1.qasm'
    name2 = 'qft_test2.qasm'
    qubits = 12
    circ0 = QuantumCircuit().from_qasm_file(path+name0)
    circ1 = QuantumCircuit().from_qasm_file(path+name1)
    circ2 = QuantumCircuit().from_qasm_file(path+name2)
    
    cvbe0_time_1 = time.time()
    cvbe0 = gen_cvbe(circ0, bdd)
    print(len(cvbe0.term_list))
    cvbe0_time_2 = time.time()
    print("C0 construction: %.6f"%(cvbe0_time_2-cvbe0_time_1))

    # cvbe1_time_1 = time.time()
    # cvbe1 = gen_cvbe(circ1, bdd)
    # # print([cvbe1.term_list[i].ref.support for i in range(len(cvbe1.term_list))])
    # # print([bdd.to_expr(cvbe1.term_list[i].ref) for i in range(len(cvbe1.term_list))])
    # cvbe1_time_2 = time.time()
    # print("C1 construction: %.6f"%(cvbe1_time_2-cvbe1_time_1))

    # cvbe2_time_1 = time.time()
    # cvbe2 = gen_cvbe(circ2, bdd)
    # cvbe2_time_2 = time.time()
    # print("C2 construction: %.6f"%(cvbe2_time_2-cvbe2_time_1))

    # merge_time1 = time.time()
    # # print([bdd.to_expr(cvbe2.term_list[i].ref) for i in range(len(cvbe2.term_list))])
    # cvbe2.switch(qubits, 1, 2, 2, 3)
    # # print([cvbe2.term_list[i].ref.support for i in range(len(cvbe2.term_list))])
    # vars = ['i'+str(i) for i in range(128)]
    # cvbe = cvbe1.sequential(cvbe2, vars[qubits:2*qubits])
    # cvbe.regularize()
    # cvbe.switch(qubits, 1, 1, 3, 2)
    # # print([bdd.to_expr(cvbe.term_list[i].ref) for i in range(len(cvbe.term_list))])
    # merge_time2 = time.time()
    # print("Merge time: %.6f"%(merge_time2-merge_time1))

    # equal_time1 = time.time()
    # res = cvbe0.equals(cvbe)
    # equal_time2 = time.time()
    # print(res)
    
    # print("Equivalence checking time: %.6f"%(equal_time2-equal_time1))

