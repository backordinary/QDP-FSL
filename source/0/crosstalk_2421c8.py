# https://github.com/huafei1137/VQE/blob/98529528b5c74d2d999f13dd63ee885e5db32333/crosstalk.py
import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
from qiskit.tools.monitor import job_monitor
# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()
class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
    
def show(root):
        queue= []
        queue.append(root)
        while len(queue) > 0 :
            for i in range(len(queue)):
                cur = queue.pop(0)
                print(cur.name, end = '  ')
                for child in cur.children:
                    queue.append(child)
            print('\n******')
import pprint
from collections import defaultdict


class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
from collections import deque
#using bfs to calculate distance
def bfs(graph, i, j):
    dis = 0
    queue = deque(str(i))
    while queue:
        for i in range(len(queue)):

            cur = queue.popleft()
            if cur == str(j):
                #print("find!!")
                return dis
            else:
                for child in graph[cur]:
                    queue.append(child)
        dis += 1
from collections import deque

def connected_components(graph,mylist,mapping):
    for i in range(len(mylist)):
        mylist[i] = str(mapping[int(mylist[i])])
    
    seen = set()
    # graph = modified(graph,mylist)
    for root in range(len(graph)):
        if str(root) not in seen and str(root) in mylist:
            seen.add(str(root))
            component = []
            queue = deque([str(root)])

            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in seen and neighbor in mylist:
                        seen.add(neighbor)
                        queue.append(neighbor)
            yield component
#generating distance matrix
def distance_matrix(graph):
    arr = []
    counter = -1
    for r in range(len(graph)):
        row = []
        for c in range(len(graph)):
            row.append(counter)
        arr.append(row)
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i == j :
                arr[i][j] = 0
            else:
                arr[i][j] = bfs(graph,i,j)
    return arr
# get distance
def get_distance(dis,component,s,mapping):
    ans = 100
    index = 0
    for j in component:
        if dis[mapping[int(s)]][mapping[int(j)]] < ans :
            ans = dis[mapping[int(s)]][mapping[int(j)]]
            index = j
        
    
    return ans,index
def update_component(component,i):
    for cur in component:
        if i in cur:
            cur.remove(i)
    component[0].append(i)
    for cur in component:
        if cur == []:
            component.remove(cur)
    
def update_mapping(mapping,i,j):
    htol = [0 for k in range(len(mapping))]
    for ff in range(len(mapping)):
        htol[int(mapping[ff])] = ff
    print(f"from hardware to logical mapping is {htol}")
    print(f"we swap hardware qubit{i} and qubit{j}")
    left = htol[int(i)]
    right = htol[int(j)]
    cur = mapping[left]
    mapping[left] = mapping[right]
    mapping[right] = cur
    print(f"from logical to hardware mapping is {mapping}")
    return
def back_mapping(mapping,i,j):
    htol = [0 for k in range(len(mapping))]
    for ff in range(len(mapping)):
        htol[int(mapping[ff])] = ff
    print(f"from hardware to logical mapping is {htol}")
    print(f"we swap hardware qubit{i} and qubit{j}")
    left = htol[int(i)]
    right = htol[int(j)]
    cur = mapping[left]
    mapping[left] = mapping[right]
    mapping[right] = cur
    print(f"from logical to hardware mapping is {mapping}")
    return
def turn_avail(avail,i,j):
    avail[i] = False
    avail[j] = False
# def modifed_path(graph,component,mapping,distancematrix):
#     biggest = component[0]
#     for i in range(len(component)-1):
#         cur = component[i+1]
#         for j in cur :
#             print(j)
#             (dis,index) = get_distance(distancematrix,biggest,j,mapping)
#             print(f"the new distance now is {dis} from qubit {j} to {index}")
            
def find_path(graph,component,mapping,distancematrix):
    
    biggest = component[0]
    swapres = []
    # avail = [True for i in range(len(mapping))]
    print(f"current mapping is {mapping}")
    count = 0
    while count < 10 and len(component) > 1:
        print(f"round {count}")
        avail = [True for i in range(len(mapping))]
        # print(f"debug with {component}")
        # print(f"debug with length should be {len(component)}")
        # print(f"debug with{ range(len(component)-1)}")
        for curi in component[1:]:
            #len(component)
            # print(i+1)
            # print(f"debug with length should be {len(component)}")
            # print(f"debug with{ range(len(component)-1)}")
            # print(f"debug with {i}")
            cur = curi
            for j in cur :
                print(f"logical qubit{j}")
                physicalj = mapping[int(j)]
                print(f"hardware qubit{physicalj}")
                costlist = []
                curdis = get_distance(distancematrix,biggest,j,mapping)
                print(f"before swap the distance is {curdis[0]}")
                if curdis[0] == 1:
                    update_component(component,str(j))
                    print(component)
                else:
                    for k in range(len(avail)):
                        
                        #if avail[k] and distancematrix[int(j)][int(k)] == 1 and str(k) not in cur:
                        if avail[k] and distancematrix[physicalj][int(k)] == 1 and k not in cur:
                            temp = []
                            (pre,n) = get_distance(distancematrix,biggest,j,mapping)
                            print(f"previous distance is {pre}")
                            print(f"we can add swap between physical qubit {mapping[int(j)]}  and physical qubit{k}  ")
                            update_mapping(mapping,physicalj,int(k))
                            # print(f"the new mapping is{mapping}")
                            (ans,index) = (get_distance(distancematrix,biggest,j,mapping))
                            print(f"the new distance now is {ans}")
                            costdis = ans - pre
                            print(f"the cost value is {costdis}")
                            back_mapping(mapping,physicalj,int(k))
                            print("******")
                            temp.append(k)
                            temp.append(costdis)
                            temp.append(ans)
                            costlist.append(temp)        
                    print(costlist)
                    
                    costlist.sort(key = lambda x:x[1])
                    if len(costlist) < 1 or costlist[0][1] > -1 :
                        print(f"all the swaps are not usefull, skip to next qubit")
                    else :

                        for c in costlist:
                            if c[2] == 1:
                                print(f"we can merget to biggest component if we swap qubit {physicalj} and qubit {c[0]}  now it has distance{c[2]}")
                        
                        print(f"debug with costlist{costlist}")
                        pick = costlist[0]
                        #maybe sometimes we do not need to choose if the swap is not good enough if cost function is all >0 we do not need to choose
                        print(f"current mapping is {mapping}")
                        
                        print(f"we will choose swap qubit {physicalj} and qubit {pick[0]}  now it has distance{pick[2]}")
                        update_mapping(mapping,physicalj,int(pick[0]))
                        turn_avail(avail,physicalj,int(pick[0]))
                        print(f"after swap the mapping is {mapping}")
                        tempres = []
                        tempres.append(physicalj)
                        tempres.append(pick[0])
                        swapres.append(tempres)
                        if pick[2] == 1:
                            print(f"update compoennt with{j}")
                            update_component(component, str(j))
                            print(component)
                #update_component()
            #print(cur)
        count += 1
    print(component)
    print(len(component))
    return swapres
def convert_to_tree(graph, root,myl,check,mapping):
    mylist = [str(mapping[int(i)]) for i in myl]
    #print(mylist)
    converttree = Tree(root)
    queue = deque([converttree])
    while queue:
        size = len(queue)
        #print(f"size is {size}")
        for i in range(size):

            cur = queue.popleft()
            # print(f"cur node is {cur.name}")
            for child in graph[str(cur.name)]:
                #print(f"child is {child}")
                if check[int(child)] == False and child in mylist:
                    check[int(child)] = True
                    temp = Tree(child)
                    cur.add_child(temp)
                    queue.append(temp)
    return converttree


def backtrack( root,backmap):
    # queue = deque([root])
    # while queue:
    #     size = len(queue)
    #     print(size)
    #     for i in range(size):
    #         cur = queue.popleft()
    #         cur.name = backmap[int(cur.name)]
    #         for child in root.children:
    #             queue.append(child)
    # return
    queue= []
    queue.append(root)
    while len(queue) > 0 :
        for i in range(len(queue)):
            cur = queue.pop(0)
            # print(cur.name, end = '  ')
            cur.name = backmap[int(cur.name)]
            for child in cur.children:
                queue.append(child)
        # print('\n******')

    # #return converttree
    # if len(graph[root]) > 0:
    #     #print('find')
    #     for child in graph[root]:
    #         print(f'child number is {child}')
    #         if int(child) < len(mylist) and check[int(child)]==False and child in mylist:
    #             check[int(child)] = True
    #             #print('match')
    #             converttree.add_child(convert_to_tree(graph,child,mylist,check))
    # else:
    #     return converttree
    # return converttree
import operator
def dfs(root,res):
    cur = root
    
    if cur is not None:
        print(cur.name)
        res.append(cur.name)
        #temp.append(cur.name)
        if len(cur.children) == 0:
            print("*****")
            res.append(-1)
            #temp.clear()
        for child in cur.children:
            dfs(child,res)
    else:
        #print("****")
        return
def get_root(graph,myl,mapping,distancematrix):
    res = [mapping[int(i)] for i in myl]
    print(res)
    dic ={}
    for i in res:
        print(i)
        count = 0
        for j in graph[str(i)]:
            if int(j) in res:
                count += 1
        dic[i] = count
    
    res = sorted(dic.items(), key=operator.itemgetter(1),reverse=True)
    # print(type(dic))
    # print(type(res))
    # print(type(res))
    max = res[0][1]
    candidate = []
    print(max)
    for i in res:
        if i[1] == max:
            candidate.append(i[0])
    print(f"all candiates are {candidate}")
    center = {}
    if len(candidate) == 1:
        print("only one candiate return")
        return candidate[0]
    else:
        for i in candidate:
            center[i] = get_mid(i,candidate,distancematrix)
        print(center)
        cenres = sorted(center.items(), key=operator.itemgetter(1))
        print(cenres)
        # print(cenres[0][0])
        return cenres[0][0]
        
def get_mid(index,res,distancematrix):
    sum = 0
    #print(res)
    for i in res:
        
        
        sum += distancematrix[int(i)][int(index)]
    return sum


def transfer_to_list(mapping,graph,root,myl):

    # backmap = [0 for i in range(len(mapping))]
    check = [False for i in range(len(graph))]
    
    check[int(root)] = True
    print(mapping)
    # for i in range(len(mapping)):
    #     backmap[int(mapping[i])] = i
    # print(backmap)
    # print(f"the tree list are{myl}")
    myt = convert_to_tree(graph,str(root), myl, check,mapping)
    #show(myt)
    # backtrack(myt,backmap)
    show(myt)
    sumres=[]
    dfs(myt,sumres)
    print(sumres)
    schedule = []
    split = []
    for i in range(len(sumres)):
        if sumres[i] == -1:
            split.append(i)
    print(split)
    startindex=0
    for i in split:
        schedule.append(sumres[startindex:i])
        startindex = i+1
    for i in range(len(schedule)):
        if i > 0:
            schedule[i].insert(0,sumres[0])
    print(schedule)
    for i in schedule:
        i = i.reverse()
    return schedule
def transfer_to_qasm(swap,schedule,graph):
    # f = open("/Users/feihua/pythonfile/VQE/schedule.qasm", "w+")
    f = open("schedule2.qasm", "w+")
    pre = ["OPENQASM 2.0;\n","include \"qelib1.inc\";\n",f"qreg q[{len(graph)}];\n"]
    f.writelines(pre)
    size = len(swap)
    swapcount = 0
    gatecount = 0
    depthcount = 0
    for i in range(len(swap)):
        swapcur = swap[i]
        swapcount += len(swapcur)
        curgatecount = 0
        curdepthmax = 0
        for k in swapcur:
            f.write(f"swap q[{int(k[0])}], q[{int(k[1])}];\n")
        writelist = []
        schedulecur = schedule[i]
        for tt in range(len(schedulecur)):
            cur = schedulecur[tt]
            curmax = 0
            curdepthmax = max(curdepthmax,len(schedulecur[tt]))
            for j in range(len(cur)):
                if j < len(cur)-1:
                    writelist.append(f"cx q[{int(cur[j])}],q[{int(cur[j+1])}];\n")
                    print(f"cx q[{int(cur[j])}],q[{int(cur[j+1])}];")
                    curgatecount += 1
                    #f.write(f"cx q[{int(cur[j])}],q[{int(cur[j+1])}];\n")
        
        f.writelines(writelist)
        f.write(f"rz(pi/2) q[{schedulecur[0][0]}];\n")
        curgatecount = curgatecount*2+1
        print(writelist)
        print(len(writelist))
        for d in reversed(writelist):
            f.write(d)
        gatecount += curgatecount
        depthcount += (curdepthmax-1)*2+1
    f.close()   
    print(f"the total swaps are {swapcount} and the total cx gates are {gatecount} the total depths are {depthcount}")
    return
def generate_string(pauli):
    generate = []
    count = -1
    for p in pauli:
        for c in p:
            count += 1
            if c == 'I':
                continue
            else:
                generate.append(str(count))
            
    return generate
import pdb
def VQE(pauli, couplingname):
    couplingsum={}
    couplingsum['Melborune'] = [('1','2'),('0','1'),('2','3'),('3','4'),('4','8'),('5','7'),('7','6'),('6','9'),('9','10'),('10','11'),('0','11'),('1','10'),('2','9'),('3','6'),('4','7'),('8','5')]
    ccp= [[0, 1], [1, 0], [1, 2], [1, 4], [2, 1], [2, 3], [3, 2], [3, 5], [4, 1], [4, 7], [5, 3], [5, 8], [6, 7], [7, 4], [7, 6], [7, 10], [8, 5], [8, 9], [8, 11], [9, 8], [10, 7], [10, 12], [11, 8], [11, 14], [12, 10], [12, 13], [12, 15], [13, 12], [13, 14], [14,  11], [14, 13], [14, 16], [15, 12], [15, 18], [16, 14], [16, 19], [17, 18], [18, 15], [18, 17], [18, 21], [19, 16], [19, 20], [19, 22], [20, 19], [21, 18], [21, 23], [22, 19], [22, 25], [23, 21], [23, 24], [24, 23], [24, 25], [25, 22], [25, 24], [25, 26], [26, 25]]
    for i in ccp:
        for k in range(len(i)):
            i[k] = str(i[k])
    couplingsum['Montreal'] = ccp
    #print(ccp)
    g = Graph(couplingsum[couplingname], directed=False)
    distancematrix = distance_matrix(g._graph)
    print(distancematrix)
    mapping = [i for i in range(len(distancematrix))]
    round = 0
    swapsum = []
    schedulesum = []
    for p in pauli:
        print("")
        print(f"**************start round {round}***********")
        print("")
        print(f"original pauli is{p}")
        generatelist = generate_string(p)
        if generatelist == []:
            print("emptylist found, move to next layer")
            round += 1
            continue
        print(f"original list is{generatelist}")
        res = list(connected_components(g._graph,generatelist,mapping))
        res.sort(key=lambda x:len(x))
        res.reverse()
        if len(res) <= 1:
            print(f"we only have one biggest component {res} ")
        else:
            print(res)
            print(f"the biggest component in hardware is {res[0]}")
            # pdb.set_trace()
            for i in range(len(res)-1):
                cur = res[i+1]
                print(f"other components in hardware are {cur} ")
        #from hardware to logical
        print("turn in to logical qubits")
        htol = [0 for i in range(len(mapping))]
        for i in range(len(mapping)):
            htol[int(mapping[i])] = i
        print(f"from hardware to logical mapping is {htol}")

        for i in range(len(res)):
            cur = res[i]
            for j in range(len(cur)):
                cur[j] = str(htol[int(cur[j])])
        print(f"the logical qubits component is {res}")

        # (ans,index) = (get_distance(distancematrix,res[0],'5',mapping))
        # print(f"the distance is {ans} from '5' to {index}")
        # print('***************')
        if len(res) > 1:
            swap = find_path(g._graph,res,mapping,distancematrix)
            swapsum.append(swap)
            print(swap)
        else :
            swapsum.append([])
        print(mapping)
        print(f"the biggest component is {res[0]}")
        myl = res[0]
        root = get_root(g._graph,myl,mapping,distancematrix)
        print(f"the root is {root}")
        schedule = transfer_to_list(mapping,g._graph,str(root),myl)
        schedulesum.append(schedule)
        # transfer_to_qasm(swap,schedule,g._graph)
        print(schedule)
        round += 1
    print(f"the total swap are {swapsum}")
    print(len(swapsum))
    for s in swapsum:
        print(f"the swaps are{s}")
    print(f"the total schedule are {schedulesum}")
    for s in schedulesum:
        print(f"the schedule are{s}")
    transfer_to_qasm(swapsum,schedulesum,g._graph)
def generate_pauli_txt(filename): 
    tempresult = []
    pauliresult = []
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        temp = line.split('*')
        tempresult.append(temp[1].strip(' '))
    for p in tempresult:
        pauliresult.append(p.rstrip())
    # print(pauliresult)   
    return pauliresult
def VQE_linear(pauli, couplingname):
    couplingsum={}
    couplingsum['Melborune'] = [('1','2'),('0','1'),('2','3'),('3','4'),('4','8'),('5','7'),('7','6'),('6','9'),('9','10'),('10','11'),('0','11'),('1','10'),('2','9'),('3','6'),('4','7'),('8','5')]
    g = Graph(couplingsum[couplingname], directed=False)
    distancematrix = distance_matrix(g._graph)
    print(distancematrix)
    cxnumber = 0
    depth = 0
    swapnumber = 0
    for p in pauli:
        print(p)
        curcx = 0
        curswap = 0
        cindex = 0
        preindex = -1

        for k in range(len(p)):
            
            if p[k] != "I":
                cindex = k
                if preindex != -1 and cindex - preindex > 1:
                    curswap += cindex-preindex-1
                preindex = k
                if curswap > 0:
                    print(f"find swap number is {curswap}")
                print(p[k])
                curcx = curcx + 1

        print(f"cureent layer cx number is{curcx}")
        cxnumber += curcx*2 + curswap*3*2
        swapnumber += curswap*2
        depth += curcx*2+1
    print(f"total cx number is{cxnumber}")
    print(f"total depths are{depth}")
def runtime(file):
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-pnnl', group='internal', project='default')
    mybackend = provider.get_backend('ibmq_mumbai')
    print(provider)
    print(mybackend)
    provider = IBMQ.get_provider(hub='ibm-q-pnnl', group='internal', project='default')
    rt = provider.runtime
    qc = QuantumCircuit.from_qasm_file(file)
    shots = 20000
    params = provider.runtime.program(program_id="circuit-runner").parameters()
    program_inputs = {
        'circuits': qc,
        'shots': shots,
        'optimization_level': 0,
        #'iterations': 5,
        # 'initial_layout': [0,1,2,3,4,5,6],
        'measurement_error_mitigation': True,
    }
    # params.circuits = circuit
    # params.measurement_error_mitigation = True
    # program_inputs = {
    #     'iterations': 3
    # }
    # Configure backend options
    options = {'backend_name': mybackend.name()}

    # Execute the circuit using the "circuit-runner" program.
    job = provider.runtime.run(program_id="circuit-runner",
                            options=options,
                            inputs=program_inputs)

    # Get runtime job result.
    # job_monitor(job)
    result = job.result(decoder=RunnerResult)

    counts = result.get_counts()
    print("finish the runtime job the results are")
    print(counts)


def main():
    print("Reading file")
    pauli = ['XXYXIYIIXZ','XXZYIIIZII'] # 01237
    pauli2 = ['XXYXIYIIXZ']
    pauli3 = ['XYZZZYIIIX']#0123459
    pauli4 = ['IIII','ZIII','ZIIZ','YYYY']
    pauli5 = generate_pauli_txt('pauli.txt')
    couplingname = 'Melborune'
    couplilarge = 'Montreal'
    #VQE(inputstring,couplilarge)
    # print(pauli5)
    #VQE_linear(inputstring,couplingname)
    # VQE(pauli4,couplingname)
    runtime('schedule3.qasm')
    print("end work")


if __name__ == "__main__":
    main()