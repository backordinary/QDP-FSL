# https://github.com/Holly-Jiang/qmap_sdp/blob/d952752d4d39ecc941a68ff88e554ba95f409953/qmapping/old_SDP_test.py
import copy
import string
from os import listdir

import cvxpy as cp
import math
import numpy as np
# 目标：
# 整个电路的fidelity最小
# 单量子比特门的error，双量子门的error
# 单量子比特门的执行时间，双量子门的执行时间
# \sum_i^{N}math.log(E(g_i))+\sun_i^{N}(g_t/T)

# qubit 限制：
# 块内的条件编织：一个逻辑qubit映射到一个物理qubit，
# 物理连接的限制，只能在相邻的位置交换 x_{mn}==1表示q_m,q_n相邻，能够进行交换
# 按照门的顺序执行
# 门能够执行
# 门不能执行-》需要调整，有些什么限制，门执行的条件，<q_i,q_j>分别映射到不相邻的<Q_m,Q_n>，
# 通过限制<Q_m,Q_n>必须要相邻，并且要按照门的顺序满足，
# 然后能找到一条路径
# 如何表示qubit的交换:x_{mn}交换q_m,q_n
# A.增加交换x_{mn}，使得门能够执行，以及目标函数最小
# 在每一个门前面都需要插入所有的交换可能性，然后搜索所有的交换排列能够达到A
from cvxpy import Expression, Variable, installed_solvers
from cvxpy.atoms.affine.binary_operators import MulExpression
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.qasm.node import Gate
from qiskit.transpiler import CouplingMap

from readOpenQASM import configuration, degree_adjcent_matrix, read_open_qasm


class SwapInfo:
    def __init__(self, edge, swaps, var1, var2):
        self.edge = edge
        self.swaps = swaps
        self.var1 = var1
        self.var2 = var2


def build_dist_table_tabu(n: int, graph: list):
    dist = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = bfsForTabu(i, j, graph)
            else:
                dist[i][i] = ShortPath()
    return dist


def bfsForTabu(startindex: int, goal: int, graph: list):
    reslut = ShortPath()
    solutions = []
    edges = []
    queue = []
    v = []
    v.append(startindex)
    queue.append(v)
    length = 0
    successors = []
    while len(queue) > 0:
        v = queue[0]
        queue.pop(0)
        current = v[len(v) - 1]
        if current == goal:
            length = len(v)
            solutions.append(v)
            break
        else:
            successors.clear()
            it = graph
            for e in it:
                if e[0] == current and not (e[1] in v) and not (e[1] in successors):
                    successors.append(e[1])
                if e[1] == current and not (e[0] in v) and not (e[0] in successors):
                    successors.append(e[0])
            # successors = sorted(successors)
            su = iter(successors)
            for succ in su:
                v2 = list(v)
                v2.append(succ)
                queue.append(v2)
    # while len(queue) > 0:
    while len(queue) > 0 and len(queue[0]) == length:
        if queue[0][len(queue[0]) - 1] == goal:
            solutions.append(queue[0])
        queue.pop(0)
    for i in range(len(solutions)):
        v1 = solutions[i]
        e1 = []
        for j in range(len(v1) - 1):
            source = v1[j]
            target = v1[j + 1]
            e1.append([source, target])
        edges.append(e1)
    reslut.distance = (length - 1) * 3
    reslut.paths = edges
    return reslut


class ShortPath:
    def __init__(self):
        self.paths = []
        self.distance = 0


# qubit 限制：
# 块内的条件编织：一个逻辑qubit映射到一个物理qubit，
# 物理连接的限制，只能在相邻的位置交换 x_{mn}==1表示q_m,q_n相邻，能够进行交换
# 按照门的顺序执行
# 门能够执行
# 门不能执行-》需要调整，有些什么限制，门执行的条件，<q_i,q_j>分别映射到不相邻的<Q_m,Q_n>，
# 通过限制<Q_m,Q_n>必须要相邻，并且要按照门的顺序满足，
# 然后能找到一条路径
# 如何表示qubit的交换:x_{mn}交换q_m,q_n
# A.增加交换x_{mn}，使得门能够执行，以及目标函数最小
# 在每一个门前面都需要插入所有的交换可能性，然后搜索所有的交换排列能够达到A

class GateInfo:
    def __init__(self, name: string, qargs: list, var=None):
        self.name = name
        self.qargs = qargs
        self.var = var


def get_the_nonzero_index(row):
    res = list()
    for i in range(len(row)):
        if isinstance(row[i], int) and row[i] != 0:
            res.append(i)
        elif isinstance(row[i], Variable):
            res.append(i)
    return res


# N:每一行交换的次数
def matrix_norm_n_add(s, X: list):
    # swap  the row of s[0] and s[1] and r1 (r2) is the indexes of nonzeo entry,
    # we let each element of r1 and r2  add the variable x_i and the indexes of r2 (resp. r1) in row s[0] (resp. s[1]) add x_j
    v1, v2 = generate_boolean_variables(2)
    for i in range(len(X[s[0]])):
        row1 = (X[s[0]][i] + v2) / 2
        row2 = (X[s[1]][i] + v2) / 2
        X[s[0]][i] = ((X[s[0]][i] + v1) / 2) + row2
        X[s[1]][i] = ((X[s[1]][i] + v1) / 2) + row1
    return X


def matrix_add(s, X: list):
    for i in range(len(X[s[0]])):
        row1 = X[s[0]][i]
        X[s[0]][i] = X[s[1]][i]
        X[s[1]][i] = row1
    return X


def MatrixMulti(S: list, X: list):
    M = [[0 for i in range(len(S))] for j in range(len(S))]
    for i in range(len(S)):
        for j in range(len(X)):
            for k in range(len(S[i])):
                if (isinstance(S[i][k], int) and S[i][k] == 0) or (isinstance(X[k][j], int) and X[k][j] == 0):
                    pass
                elif isinstance(S[i][k], int) and S[i][k] == 1:
                    M[i][j] += X[k][j]
                elif isinstance(X[k][j], int) and X[k][j] == 1:
                    M[i][j] += S[i][k]
                else:
                    M[i][j] += S[i][k] * X[k][j]
    return M


def build_coupling_constraint(coupling: list, n: list):
    constraints = [[-1] * range(n) for i in range(n)]
    for c in coupling:
        constraints[c[0]][c[1]] = 1
    return constraints


def build_unique_constraints(X: list, constraints, param):
    i = 0
    while i < len(X):
        j = 0
        row = 0
        col = 0
        # variables are integer, if we want to X[i] !=X[i+1], the abs of them is >0, then >=1
        # cvxpy limit the strict > <
        while j < len(X):
            row += X[i][j]
            col += X[j][i]
            j += 1
        if (isinstance(row, int) and row > 1) or (not isinstance(row, int)):
            constraints.extend([row <= param])
        if (isinstance(col, int) and col > 1) or (not isinstance(row, int)):
            constraints.extend([col <= param])
        i += 1
    return constraints


# the node in the shortest path
# def generate_candidate_set(dist, node: DAGOpNode, X: list):
#     res = list()
#     # map the logical qubits to physical qubits
#     # qubits = list()
#     # for q in dag.qubits:
#     #     qubits.append(X[q])
#     print(X[node.qargs[0].index][0], X[node.qargs[1].index][0])
#     path = dist[X[node.qargs[0].index][0]][X[node.qargs[1].index][0]].paths
#     print(path)
#     for c in path:
#         for k in c:
#             res.append(k)
#         # if c[0] in qubits or c[1] in qubits:
#         #     res.append(c)
#     return res

def generate_candidate_set(coupling_map: list, dag: DAGCircuit, X):
    res = list()
    # map the logical qubits to physical qubits
    qubits = set()
    for q in dag.qubits:
        qubits.add(L2PP(X, q.index))
    for c in coupling_map:
        if c[0] in qubits or c[1] in qubits:
            flag = False
            for k in res:
                if (c[0] == k[0] and c[1] == k[1]) or (c[0] == k[1] and c[1] == k[0]):
                    flag = True
                    break
            if not flag:
                res.append(c)
    return res


# initial mapping
# give the logical qubit, return the mapped physical qubit
def L2P(X: list, q: int):
    for i in range(len(X)):
        if isinstance(X[i][q], int) and X[i][q] != 0:
            return i
        elif not isinstance(X[i][q], int):
            return None
    pass


# give the logical qubit, return the mapped physical qubit
def L2PP(X: list, q: int, param=None, constraints=None):
    phy_var = list()
    var = list()
    for i in range(len(X)):
        if isinstance(X[i][q], int) and X[i][q] != 0:
            if not constraints is None:
                var.append(cp.Variable(boolean=True))
                constraints.append(var[0] == param)
            return [i], var
        elif not isinstance(X[i][q], int):
            phy_var.append(i)
            v = cp.Variable(boolean=True)
            constraints.append(X[i][q] == v)
            var.append(v)
    return phy_var, var


def P2L(X: list, q: int):
    for i in range(len(X)):
        if X[q][i] == 1:
            return i


def generate_shortest_path_list(dist: list, gate: list, coupling_map: list):
    res = list()
    # 超过总长度>5的路径prune
    limit = 6
    for c in coupling_map:
        # move the qubit gate[0] to the qubit c[0]
        m1 = dist[gate[0]][c[0]]
        m2 = dist[gate[1]][c[1]]
        m3 = dist[gate[0]][c[1]]
        m4 = dist[gate[1]][c[0]]

        if m1.distance + m2.distance < m3.distance + m4.distance:
            if m1.distance + m2.distance > limit:
                continue
            for p in m1.paths:
                for q in m2.paths:
                    s = list()
                    s.extend(p)
                    s.extend(q)
                    res.append([[c[0], c[1]], s])

        else:
            if m3.distance + m4.distance > limit:
                continue
            for p in m3.paths:
                for q in m4.paths:
                    s = list()
                    s.extend(p)
                    s.extend(q)
                    res.append([[c[1], c[0]], s])
    return res


def matrix_deepcopy(X):
    res = list()
    for i in X:
        r = list()
        for j in i:
            r.append(j)
        res.append(r)
    return res


def gateInfo_deepcopy(all_gates):
    res = list()
    for g in all_gates:
        g1 = GateInfo(g.name, g.qargs, g.var)
        res.append(g1)
    return res


def generate_multi_phy_qubit(X, qargs, param, constraints):
    if len(qargs) == 2:
        q1, v1 = L2PP(X, qargs[0].index, param, constraints)
        q2, v2 = L2PP(X, qargs[1].index, param, constraints)
        res = list()
        sum1 = 0
        for i in range(len(q1)):
            sum1 += v1[i]
            for j in range(len(q2)):
                if q1[i] != q2[j]:
                    # physical qubit list, control var1 list, target var2 list
                    res.append([[q1[i], q2[j]], v1[i], v2[j]])
    else:
        # physical qubit list, var list
        res = L2PP(X, qargs[0].index)

    return res


def generate_multi_edges(P, qargs, constraint: list, param):
    vars = list()
    for i in range(len(P)):
        X = P[i][0]
        for q in qargs:
            for j in range(len(X)):
                if X[j][q] != 0:
                    v = cp.Variable(boolean=True)
                    vars.append(v)
                    constraint.append(X[j][q] == v)
    sum = 0
    for v in vars:
        sum += v
    constraint.append(sum == param)
    return vars


def build_constraint(P: list, coupling_map: list, dag: DAGCircuit, param):
    # param denote the constant 1
    # X: index is logical qubit, and the value is the corresponding physical qubit, default to -1
    # Y: index is physical qubit, and the value is the corresponding logical qubit, default to -1
    # 假设有物理比特一样数目的逻辑比特
    vars = list()
    # The qubits of a gate in the list "all_gate" are the physical qubits
    constraints = []
    # for all i in range(X), X[i]!=-1 and
    # if X[i]==j for all k!=i and k in range(X) X[k]!=j
    # build_unique_constraints(X, constraints) todo
    dist = build_dist_table_tabu(len(P[0][0]), coupling_map)

    # physical coupling constraint:
    # coupling_const = build_coupling_constraint(coupling=coupling_map, n=len(X))

    # C = generate_candidate_set(coupling_map, dag, X)
    # 编码门的顺序 在当前映射下门能执行门，再看下一个门
    # iterate the gate ordered by the topological order
    # if the two mapped physical qubit adjcent, the gate can be executed
    # todo prune
    # the neighbor of all physical node of dag
    for node in dag.topological_op_nodes():
        R = list()
        v1s = list()
        print(node.name, node.qargs)
        print(len(P))
        # generate the node.qargs mapping probability
        ve1 = list()
        ve2 = list()
        for p in range(len(P)):
            print("The %s-th mapping:" % p)
            X = P[p][0]
            all_gates = P[p][1]
            # the mapping result of each path P
            if len(node.qargs) == 2:
                edges = generate_multi_phy_qubit(X, node.qargs, param, constraints)
                for edge in edges:
                    ve1.append(edge[1])
                    ve2.append(edge[2])
                    # generate the shortest path list [[[],[],[]][[],[],[]]]
                    d = dist[edge[0][0]][edge[0][1]]
                    if d.distance == 3:
                        v1s.append(edge[1] * edge[2])
                        X = matrix_deepcopy(P[p][0])
                        all_gates = gateInfo_deepcopy(P[p][1])
                        all_gates.append(GateInfo(node.op.name, [edge[0][0], edge[0][1]], edge[1] * edge[2]))
                        R.append([X, all_gates])
                        break
                    C = generate_shortest_path_list(dist, edge[0], coupling_map)
                    # 增加SWAP门集S，使得所有的门都可以执行
                    # 在每一个门前都有所有的交换候选集C，长度最长是耦合图的最长路径
                    # todo prune
                    # 用一个矩阵M表示如果交换(x,y)则将矩阵M_{xy}==第几次交换，矩阵能交换的为0不能交换的-1
                    # 禁忌策略：最近交换过的边不要再交换回去，todo 如果遇到禁忌策略回退
                    # 最坏情况：最长路径l*边条数e个变量-》每次选择邻接
                    # S[j]：一条路径的变换包括[目标位置，]
                    # c[2]: the number of Swaps
                    # c[3]: the number of identity matrices
                    S = generate_variables_matrix(len(X), C, edge[1] * edge[2])
                    for j in range(len(S)):
                        # variables sum(v1s)==1
                        v1s.append(S[j].var1)
                        X = matrix_deepcopy(P[p][0])
                        all_gates = gateInfo_deepcopy(P[p][1])
                        vars.append(S[j].var1)
                        vars.append(S[j].var2)
                        s = S[j].swaps
                        for i in range(len(s)):
                            # print("swap: ", s[i])
                            # print("X: ", X)
                            X = matrix_add(s[i], X)
                            # insert the swap gate into the gate sequence
                            s1 = GateInfo(node.op.name, [C[j][1][i][0], C[j][1][i][1]], S[j].var1)
                            all_gates.append(s1)
                            s2 = GateInfo(node.op.name, [C[j][1][i][1], C[j][1][i][0]], S[j].var1)
                            all_gates.append(s2)
                            all_gates.append(s1)

                        build_unique_constraints(X, constraints, param)
                        # one of the path: the qubits node.qargs mapped to the adjacent qubits S[j].edge,
                        # Only one path of all paths is true sum(v1s)==1
                        constraints.append(
                            X[S[j].edge[0]][node.qargs[0].index] + X[S[j].edge[1]][node.qargs[1].index] == S[j].var1)
                        all_gates.append(GateInfo(node.op.name, [S[j].edge[0], S[j].edge[1]], S[j].var1))
                        R.append(X)
            else:
                nodes = L2PP(X, node.qargs[0].index, param, constraints)
                all_gates.append(GateInfo(node.op.name, nodes[0], nodes[1]))
        if len(node.qargs) == 2:
            v1 = 0
            for v in ve1:
                v1 += v
            constraints.append(v1 == param)
            vars.extend(ve1)
            v1 = 0
            for v in ve2:
                v1 += v
            constraints.append(v1 == param)
            vars.extend(ve2)
            v1 = 0
            for v in v1s:
                v1 += v
            constraints.append(v1 == param)
            P = R

        # 判断边edge是否在coupling_map中存在 若edge 存在coupling的邻接矩阵A中，则A[edge[0]][edge[1]]==1
        # insert enough swap gates to the edge in coupling map
        # set_constraints, v3 = build_set_inter_constraint(coupling_map, len(X))
        # e = np.hstack(edge)
        # constraints.extend([e - np.hstack(set_constraints) == 0])
        # constraints.extend([v3 == 1])

        # g = GateInfo(node.op.name, edge)
        # all_gates.append(g)
    return constraints, vars, P


# 构造coupling的邻接矩阵有边相邻则为1
# # todo consider using the fidelity of edge
# def build_set_inter_constraint(M: list, n: int):
#     res = np.zeros((n, n))
#     for i in range(len(M)):
#         res[M[i][0]][M[i][1]] = 1
#     return res


# [[0, 1], [1, 0], [1, 2], [1, 4], ...]
# generate the variables list  v1
# res: [0*x1+1*x2+1*x3+1*x4...,1*x1+0*x2+2*x3+4*x4]
# x1+x2+x3+x4+...=1, for  x1,x2,...\in {0, 1}
def build_set_inter_constraint(M: list, n: int):
    v1 = generate_boolean_variables(len(M))
    res = [0, 0]
    res2 = 0
    for i in range(len(M)):
        res[0] = res[0] + M[i][0] * v1[i]
        res[1] = res[1] + M[i][1] * v1[i]
        res2 += v1[i]
    return res, res2


# name: gate name
# qubits: operation qubits
# gates: the gate errors and time of architecture
# 门的错误如何编码到目标函数上
#
def edge_info(g, gates: list):
    for c in gates:
        if c.gate == g.name and set(g.qargs) == set(c.qubits):
            return c.parameters
    pass


# 根据门的名字，作用qubit，需要的qubit信息name{gate_length,gate_error}
# 架构名字
def gate_info(g: GateInfo, name: string, arch: string):
    conf, prop = configuration(arch)
    for gate in prop.gates:
        if gate.gate == g.name:
            if len(set(gate.qubits).intersection(set(g.qargs)))==len(g.qargs):
                for p in gate.parameters:
                    if p.name == name:
                        return p.value
                    elif p.name == name:
                        return p.value


# 将插入SWAP的矩阵转化为SWAP依次插入的位置list
def matrix2list(M: list):
    l = len(M)
    res = list()
    flag = False
    for i in range(l * l):
        for j in range(l):
            for k in range(l):
                if i == M[j][k]:
                    res.append([j, k])
                    flag = True
                    break
            if flag == True:
                flag = False
                break
    return res


# variable 才是SDP能够求解出来的，
# 遍历dag，V是变量的矩阵列表
# 计算在dag中门的执行过程中插入门后生成的电路的fidelity
# 有一个硬性限制，矩阵中的值数量不超过目标电路的最长路径
# obj=\sum_g g.e-\sum_q q.t/q.T
# def objective_function(dag: DAGCircuit, V: list, map: list):
#     count = 0
#     obj = 1
#     l = len(V[count])
#     for node in dag.topological_op_nodes():
#         M = matrix2list(V[count])
#         count += 1
#         # 插入的SWAP门误差
#         for i in M:
#             param = edge_info(node.name, [i[0], i[1]], map)
#             obj *= param[1].value
#         # original 门的误差
#         if len(node.qargs) == 2:
#             c = node.qargs[0].index
#             t = node.qargs[1].index
#             param = edge_info(node.name, [c, t], map)
#             obj *= param[1].value
#     return obj


# gates: all gates
# params: all information (gate errors and gate execution time) of gate in the architecture
# return: the fidelity of all gates in topologic
# 门的误差再乘以变量，所以要分开记录插入的变换矩阵和对应的变量
def objective_function(P: list, params: list, cst: list, vars: list, param):
    res = list()
    for p in P:
        V = 1
        for g in p[1]:
            if g.name == 'cx':
                info = edge_info(g, params)
                V += info[0].value * g.var
            else:
                # sum1=0
                # for q in range(len(g.qargs)):
                pass

        res.append(V)
    sum = 0
    vsum = 0
    V = generate_boolean_variables(len(res))
    for i in range(len(V)):
        sum += V[i] * res[i]
        vsum += V[i]
    cst.append(vsum == param)
    vars.extend(V)
    return sum


# Map logical qubits to physical qubits according to the degree of nodes
# in interaction graph and coupling graph
# todo 用整体的interaction graph 定位映射的总体位置还是
# 仅仅用当前的interaction定位对当前结构最有利的映射
# 首先确定一个节点的映射，然后根据出现的先后顺序，边的权重确定先后映射次序
# 出现先后比出现次数更重要
# 中心节点有必要吗 中心能保证去其他地方的路径短，但是根本不需要去更远的地方，更侧重的还是去更多的地方，因为连接密集
# 度最大已经能保证他去其他地方的路径多
# 深度递归的分配1 2 3 分配3 2 1
# 分为已分配节点A和未分配节点B，未分配节点需要递归的找直到找到与已分配节点有关联的边再迭代的分配
# Q: 不连通的图怎么处理 todo
# A: 节点多的子图先进行映射 节点少的节点映射到哪里 todo
# A: 远离一映射部分进行映射？ 挨着一映射部分进行映射，分支节点的映射由联通部分映射决定
# 每个subdag都进行初始映射还是使用上一个subdag的映射 todo
# 第i个映射 利用第i-1块映射的结果，得到第i块的的方案，第i+1使用独立映射，然后怎么影响第i块的方案 这样的话需要先对每个subdag映射 todo
# 再从头到尾进行映射，使用i-1 的映射作为初始映射，done
# 能不能提取出来第i+1个映射的特点来影响第i个映射的抉择 todo
def generate_degree_mapping(M_P, assign, ini, e1, e2, D_P, D_L):
    if not is_logical_not_location(ini, e1):
        return
    delta = 10
    index = -1
    p2 = L2P(ini, e2)
    k = 0
    adj = list()
    while (k < len(M_P[p2])):
        if M_P[p2][k] != 0:
            adj.append(k)
            if is_not_location(ini, k) and is_logical_not_location(ini, e1):
                if math.fabs(D_P[k] - D_L[e1]) < delta:
                    index = k
                    delta = math.fabs(D_P[k] - D_L[e1])
                else:
                    print("not math.fabs(D_P[j] - D_L[e1]) < delte")
        k += 1
    if index == -1:
        print("------", index)
        for i in range(len(M_P)):
            if is_not_location(ini, i) and not i in adj:
                adj.append(i)
        for i in adj:
            for j in range(len(M_P)):
                if M_P[i][j] == 1:
                    if is_not_location(ini, j) and is_logical_not_location(ini, e1):
                        index = j
    if index == -1:
        print("------", index)
    ini[index][e1] = 1
    assign.add(e1)


def is_logical_not_location(ini: list, j: int):
    for l in range(len(ini)):
        if ini[l][j] == 1:
            return False
    return True


def is_not_location(ini: list, j: int):
    for l in range(len(ini)):
        if ini[j][l] == 1:
            return False
    return True


def map_dag_degree(dag: DAGCircuit, assign, queue, M_P, ini, D_P, D_L):
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            continue
        if len(node.qargs) == 2:
            queue.extend([node.qargs[0].index, node.qargs[1].index])
            if {node.qargs[0].index, node.qargs[1].index}.isdisjoint(assign):
                continue

        while len(queue) > 0:
            e1 = queue.pop()
            e2 = queue.pop()
            print(e1, e2, assign)
            # if e1 and e2 are not in assigin, they are entered into the queue
            # if e1 and e2 are in assigin, skip them
            # if either e1 or e2 in assigin, we map the unmapped node
            if e1 not in assign and e2 not in assign:
                pass  # todo
            elif e1 in assign and e2 in assign:
                continue
            else:
                if e1 not in assign:
                    generate_degree_mapping(M_P, assign, ini, e1, e2, D_P, D_L)
                else:
                    generate_degree_mapping(M_P, assign, ini, e2, e1, D_P, D_L)
    while len(queue) > 0:
        e1 = queue.pop()
        e2 = queue.pop()
        print(e1, e2, assign)
        # if e1 and e2 are not in assigin, they are entered into the queue
        # if e1 and e2 are in assigin, skip them
        # if either e1 or e2 in assigin, we map the unmapped node
        if e1 not in assign and e2 not in assign:
            pass  # todo
        elif e1 in assign and e2 in assign:
            continue
        else:
            if e1 not in assign:
                generate_degree_mapping(M_P, assign, ini, e1, e2, D_P, D_L)
            else:
                generate_degree_mapping(M_P, assign, ini, e2, e1, D_P, D_L)


def map_single_qubit(M_P, q, ini, assign):
    phy_qubits = list()
    for i in range(len(ini)):
        flag = True
        for j in range(len(ini[i])):
            if ini[i][j] == 1:
                flag = False
                break
        if flag:
            phy_qubits.append(i)

    for i in range(len(M_P)):
        for j in range(len(M_P[i])):
            if i in phy_qubits and M_P[i][j] == 0 and is_not_location(ini, j) and is_logical_not_location(ini, q):
                ini[j][q] = 1
            elif j in phy_qubits and M_P[i][j] == 0 and is_not_location(ini, i) and is_logical_not_location(ini, q):
                ini[i][q] = 1
    if L2PP(ini, q) == None:
        for i in range(len(ini)):
            if (not i in phy_qubits) and is_not_location(ini, i) and is_logical_not_location(ini, q):
                ini[i][q] = 1
    assign.add(q)


def map_IG_degree(IG: list, assign, assign1, queue, M_P, ini, D_P, D_L):
    for ig in IG:
        if len(ig) == 1:
            if is_logical_not_location(ini, ig[0]):
                map_single_qubit(M_P, ig[0], ini, assign)
            continue
        if ({ig[0], ig[1]}.isdisjoint(assign) and {ig[0], ig[1]}.isdisjoint(assign1)):
            queue.extend(ig)
            continue
        while len(queue) > 0:
            e1 = queue.pop()
            e2 = queue.pop()
            # if e1 and e2 are not in assigin, they are entered into the queue
            # if e1 and e2 are in assigin, skip them
            # if either e1 or e2 in assigin, we map the unmapped node
            if e1 not in assign and e2 not in assign:
                pass  # todo
            elif e1 in assign and e2 in assign:
                continue
            else:
                if e1 not in assign:
                    generate_degree_mapping(M_P, assign, ini, e1, e2, D_P, D_L)
                else:
                    generate_degree_mapping(M_P, assign, ini, e2, e1, D_P, D_L)


def generate_ini_mapping_by_degree(coupling_map, dag: DAGCircuit, IG: list, n: int):
    ini = [[0 for i in range(n)] for j in range(n)]
    assign = set()
    D_P, M_P = degree_adjcent_matrix(coupling_map, n)
    edge = list()
    # find the latest degree node
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            edge.append([node.qargs[0].index, node.qargs[1].index])
    D_L, M_L = degree_adjcent_matrix(edge, n)
    d_max = 0
    d_p_index = -1,
    for j in range(len(D_P)):
        if D_P[j] > d_max:
            d_p_index = j
    d_max = 0
    d_l_index = -1
    for j in range(len(D_L)):
        if D_L[j] > d_max:
            d_l_index = j
    ini[d_p_index][d_l_index] = 1
    assign.add(d_l_index)
    queue = list()
    map_dag_degree(dag, assign, queue, M_P, ini, D_P, D_L)
    # the unmapped node:
    # map them to  the neighbor of the mapped node with minimal degree
    d = 10
    index = -1
    for j in assign:
        if D_L[j] < d:
            d = D_L[j]
            index = j
    for j in range(len(M_P[L2P(ini, index)])):
        if M_P[L2P(ini, index)][j] != 0 and is_not_location(ini, j):
            map_IG_degree(IG, {j}, assign, queue, M_P, ini, D_P, D_L)
    return ini


# gate basis: ['id', 'rz', 'sx', 'x', 'cx', 'reset']
def SDP_tools(X: list, dag: DAGCircuit, IG: list, arch: string):
    conf, prop = configuration(arch)
    cm = CouplingMap(conf.coupling_map)
    # g_l = gate_info(g, "gate_length", arch)
    phy_deg, am = degree_adjcent_matrix(conf.coupling_map, len(cm.physical_qubits))
    # variables of the initial mapping
    if len(X) == 0:
        X = generate_ini_mapping_by_degree(conf.coupling_map, dag, IG, len(cm.physical_qubits))
        # X = generate_int_variables(len(cm.physical_qubits))
    # vars: the variables of Swaps and identity matrices
    # v1: the variables of Swaps
    param = cp.Parameter()
    cst, vars, P = build_constraint([[X, []]], conf.coupling_map, dag, param)

    obj = objective_function(P, prop.gates, cst, vars, param=param)
    param.value = 1
    prob = cp.Problem(cp.Minimize(obj), cst)
    print(prob)

    print("----------", prob.objective.is_dcp())
    prob.solve(solver=cp.GLPK_MI, verbose=True)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var:")
    # for g in all_gates:
    #     print(g.var, end=', ')
    pass


def generate_swap_matrices(n: int, coupling_map: list):
    res = list()
    for i in coupling_map:
        matrix = np.eye(n)
        matrix[i[0]][i[1]] = 1
        matrix[i[1]][i[0]] = 1
        matrix[i[1]][i[1]] = 0
        matrix[i[0]][i[0]] = 0
        res.append(matrix)
    return res


def generate_int_variables(n: int):
    va = list()
    for i in range(n):
        va.append(cp.Variable(integer=True))
    return va


def generate_boolean_variables(n: int):
    va = list()
    for i in range(n):
        va.append(cp.Variable(boolean=True))
    return va


# generate the basic transforation matrics,
# the transformation position v1 and the diagnal position v2 are the variables
# record the transformation matrices M and the correspongding edge C
def generate_eye_list(n: int):
    res = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        res[i][i] = 1
    return res


# generate the swap sequence
# todo prune
# C：the candidates list
def generate_variables_matrix(n: int, C: list, v):
    res = list()
    v1 = generate_boolean_variables(len(C))
    v2 = generate_boolean_variables(len(C))
    for j in range(len(C)):
        m = list()
        for i in range(len(C[j][1])):
            matrix = generate_eye_list(n)
            matrix[C[j][1][i][0]][C[j][1][i][1]] = v1[j] * v
            matrix[C[j][1][i][1]][C[j][1][i][0]] = v1[j] * v
            matrix[C[j][1][i][0]][C[j][1][i][0]] = v2[j]
            matrix[C[j][1][i][1]][C[j][1][i][1]] = v2[j]
            m.append(matrix)
        # [the mapped edge, the swap matrix sequence, the swap variables, the identity variables]
        res.append(SwapInfo(C[j][0], m, v1[j], v2[j]))
    return res


if __name__ == '__main__':
    files = listdir("./nam_u/")
    files = sorted(files)
    count = 0
    for path in files:
        print(count, path)
        count += 1
        IG, dags = read_open_qasm("./nam_u/" + path, path)
        arch = "tokyo"
        for i in range(len(dags)):
            SDP_tools([], dags[i], IG, arch)

# Create two scalar optimization variables.
# Solving a problem with different solvers.
# print(installed_solvers())
# x = cp.Variable(boolean=True)
# y = cp.Variable(boolean=True)
# param1=cp.Parameter()
# obj = cp.Minimize(x+y)
# param1.value=1
# constraints = [x == param1,
#            y == param1,
#            x+y <= param1]
# prob = cp.Problem(obj, constraints)
# # Solve with SCIP.
# prob.solve(solver=cp.GLPK_MI)
# print("optimal value with SCIP:", prob.value)
