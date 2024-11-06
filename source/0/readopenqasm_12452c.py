# https://github.com/Holly-Jiang/qmap_sdp/blob/72c793af0ec82fad9e8dabaa6c9a7c698f645d2c/qmapping/readOpenQASM.py
import os
import string
from os import listdir

import retworkx as rx
import networkx as nx
from networkx.classes.graphviews import generic_graph_view
from networkx.drawing.nx_agraph import view_pygraphviz
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode
from qiskit.providers.fake_provider import FakeAlmaden, FakeSydney, FakeManhattan, FakeTokyo
from qiskit.transpiler import CouplingMap
from dagDrawer import dag_drawer


def makedir(filename):
    new_path = os.path.join("./subdags/", filename)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    return new_path


# 判断cx节点的node的两个祖先节点是否有相同的祖先节点
# 相交则有环出现 True
# 不相交则返回False
def check_cycle(subdag: DAGCircuit, node: DAGOpNode) -> bool:
    if node.op.name == 'cx':
        ancestors = set()
        ancs = subdag.predecessors(node)
        for anc in ancs:
            ids = subdag.ancestors(anc)

            if ids.intersection(ancestors):
                return True
            else:
                ancestors.update(ids)
        return False
    return False


def read_open_qasm(path: string, filename) -> QuantumCircuit:
    circuit = QuantumCircuit.from_qasm_file(path)
    dag = circuit_to_dag(circuit)
    dag_drawer(dag, scale=0.7, filename="./dags/" + filename.split(".")[0] + ".png", style="color")
    dag_qubits = set()
    all_qubits = list()
    arch = "manhattan"
    conf, prop = configuration(arch)
    cm = CouplingMap(conf.coupling_map)
    for q in dag.qubits:
        dag_qubits.add(q.index)
    for i in range(len(cm.physical_qubits)):
        if not i in dag_qubits:
            all_qubits.append([i])
    # construct the fisrt interaction graph
    IG = list()
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            print('[%d,%d]'%(node.qargs[0].index-1,node.qargs[1].index-1) ,end=';')
            for ig in IG:
                if node.qargs[0].index in ig and node.qargs[1].index in ig:
                    continue
            IG.append([node.qargs[0].index, node.qargs[1].index])
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            for ig in IG:
                if node.qargs[0].index in ig:
                    continue
            IG.append([node.qargs[0].index])
    IG.extend(all_qubits)
    # dags = circuit_partition(dag, filename)
    # # print(dag.to_networkx())
    # generic_graph_view(dag.to_networkx())
    # print(circuit_drawer(circuit))
    return dag, IG


def count_iterable(i):
    return sum(1 for e in i)


def remove_ancestors(dag: DAGCircuit, revisited_nodes):
    for node in dag.topological_op_nodes():
        if node._node_id in revisited_nodes:
            dag.remove_op_node(node)


# 将电路根据拓扑排序遍历DAG，然后根据电路中是否出现环进行划分
def circuit_partition(dag: DAGCircuit, filename):
    dag_node_count = dag.node_counter
    print("dag_node:", dag_node_count)
    edge_count = 0
    node_count = 0
    sub_node_count = 0
    dags = []
    subdag = DAGCircuit()
    subdag.metadata = dag.metadata
    del_nodes = set()
    revisited_nodes = set()
    for n in range(16):
        del_nodes.add(n)
    for node in dag.topological_op_nodes():
        if isinstance(node, DAGOpNode):
            # print(node.op.copy(), node.qargs, node.cargs)
            revisited_nodes.add(node._node_id)
            duplicate_qubits = set(subdag._wires).intersection(node.qargs)
            for nd in node.qargs:
                if nd not in duplicate_qubits:
                    subdag.qubits.append(nd)
                    subdag._add_wire(nd)
            new_node = subdag.apply_operation_back(node.op.copy(), node.qargs, node.cargs)
            ancs = dag.predecessors(node)
            for anc in ancs:
                if isinstance(anc, DAGInNode) or anc._node_id in del_nodes:
                    pass
                else:
                    edge_count += 1
            node_count += 1
            if check_cycle(subdag, new_node):
                dags.append(subdag)
                sub_node_count += len(revisited_nodes)
                remove_ancestors(dag, revisited_nodes)
                del_nodes.update(revisited_nodes)
                dag_drawer(subdag, scale=0.7, filename=makedir(filename) + "/" + str(len(dags)) + ".png", style="color")
                subdag = DAGCircuit()
                subdag.metadata = dag.metadata
                edge_count = 0
                node_count = 0
                revisited_nodes = set()
                continue

    dag_drawer(subdag, scale=0.7, filename=makedir(filename) + "/" + str(len(dags)) + ".png", style="color")
    dags.append(subdag)
    sub_node_count += len(revisited_nodes)
    print("subdag node:", sub_node_count, dag_node_count - sub_node_count)
    # if dag_node_count - sub_node_count - 48 != 0:
    #     exit(-999999)
    return dags


# 根据设备名称读取配置
def configuration(conf):
    if conf == "sydney":
        conf = FakeSydney().configuration()
        prop = FakeSydney().properties()
        return conf, prop
    elif conf == "manhattan":
        conf = FakeManhattan().configuration()
        prop = FakeManhattan().properties()
        return conf, prop
    elif conf == "tokyo":
        conf = FakeTokyo().configuration()
        prop = FakeTokyo().properties()
        return conf, prop


# 计算节点的度 邻接矩阵
def degree_adjcent_matrix(cm: list, n: int):
    deg = [0 for i in range(n)]
    matrix = [[0 for i in range(n)] for j in range(n)]
    for e in cm:
        matrix[e[0]][e[1]] += 1
        deg[e[0]] += 1
        deg[e[1]] += 1
    return deg, matrix


# 计算coupling_graph的各个节点的度以及各个节点的error rate T1，T2，2-qubit之间的fidelity 对每个度的节点做一个排序
# 计算逻辑节点的度，将逻辑节点映射到物理节点上度最相近的节点上，首先映射一个节点然后根据该节点映射他的邻居，
# 如果存在非连通图，怎么办 TODO
def initial_mapping(dag: DAGCircuit):
    conf, prop = configuration("sydney")
    cm = CouplingMap(conf.coupling_map)
    phy_deg = degree_adjcent_matrix(conf.coupling_map, len(cm.physical_qubits))
    dag.topological_op_nodes()
    pass


if __name__ == '__main__':
    # read_open_qasm("./test.qasm", "111")
    # read_open_qasm("/Users/jiangqianxi/Desktop/github/TSA/tsa/src/main/resources/data/4gt4-v0_72.qasm", "4gt4-v0_72.qasm")
    files = listdir("/Users/jiangqianxi/Desktop/github/TSA/tsa/src/main/resources/data/")
    files = sorted(files)
    count = 0
    for path in files:
        print("the %d-th circuit: %s" % (count, path))
        count += 1
        dags = read_open_qasm("/Users/jiangqianxi/Desktop/github/TSA/tsa/src/main/resources/data/" + path, path)
        pass
        # for dag in dags:
        #     initial_mapping(dag)
