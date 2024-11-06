# https://github.com/aurore-chappuis/quantum-teleportation/blob/f623cfdaf2aa2005593e15ad33e58a1aefe4812f/TH.py
from os import stat
from typing import *
import math

from qiskit import *
from qiskit.compiler.transpiler import _parse_backend_properties 
from qiskit.providers.ibmq import *
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

proco = ["ibmq_manila","ibmq_santiago","ibmq_athens","ibmq_belem","ibmq_quito","ibmq_lima","ibmqx2"]

class circuitBuilder(object):
    def __init__(self,minCiruitsize:Tuple[int,int],builder:Callable[[QuantumCircuit,Any],QuantumCircuit],stateValidator:Callable[[Any],bool] = lambda s: True,state = None):
        self._builder = builder
        self._validator = stateValidator
        self.minCiruitsize = minCiruitsize

        if self.isValidState(state):
            self._state = state
        else:
            raise TypeError(f"Invalide state {state}")
        
    def __call__(self,circuit:QuantumCircuit) -> QuantumCircuit:
        return self._builder(circuit,self._state)
    
    def setState(self,state:Any):
        if self.isValidState(state):
            self._state = state
        else :
            raise TypeError(f"Invalide state {state}")
        
    def isValidState(self,state:Any) -> bool:
        return self._validator(state)


class Qubit(object):
    def __init__(self,c1:complex,c2:complex):
        if not math.isclose(abs(c1)**2+abs(c2)**2, 1):
            raise ValueError(f"Invalid Qubit state {c1}|0> + {c2}|1> sum of amplitudes-squared does not equal one. ")
        self.c1 = c1
        self.c2 = c2

    def QubitValidator(s:Any) -> bool:
        return isinstance(s,Qubit)

    def toVector(self)->list:
        return [self.c1,self.c2]
    
    def __str__(self) -> str:
        return f"[{self.c1};{self.c2}]"

def teleportBuilder(circuit:QuantumCircuit,state:Qubit)->QuantumCircuit:

    circuit.initialize(state.toVector(),0)

    circuit.h(1)
    circuit.cnot(1,2)

    circuit.barrier()

    circuit.cnot(0,1)
    circuit.h(0)

    circuit.barrier()

    circuit.cnot(1,2)
    circuit.cz(0,2)

    circuit.barrier()

    circuit.measure(2,0)

    return circuit

import time
import re

class Reprinter(object):
    def __init__(self) -> None:
        self._text = ""
    
    def up(self,nbline):
        for _ in range(nbline):
            sys.stdout.write("\x1b[A")

    def reprint(self,text):
        self.clear()
        self.print(text)

        
    def clear(self):
        self.up(self._text.count("\n"))
        sys.stdout.write(re.sub(r"[^\s]", " ", self._text))
        self.up(self._text.count("\n"))
        self._text = ""

    def print(self,text):
        print(text)
        self._text += text + "\n"

    def clearCache(self):
        self._text = ""

def startJobs(circuit,qComputerFilter,statusLine="|"):
    jobs = {n:{"job":None,"expeResult":None,"status":None} for n in qComputerFilter}
    
    for f in qComputerFilter:
        statusLine += "QUEUED".center(max(len(f)+2,11)," ") + "|"
        qcomputer = provider.get_backend(f)
        while (True):
            joblim = qcomputer.job_limit()
            if joblim.maximum_jobs - joblim.active_jobs >=1:
                break
            time.sleep(2)
        jobs[f]["job"] = execute(circuit,backend = qcomputer,shots=512)
    
    return jobs,statusLine

def monitorJobs(jobsDict,header,delim,printer):
    jobsLeft = 0
    slen = 0
    
    for s in jobsDict.keys():
        jobsLeft += len(jobsDict[s].keys())
        slen = max(slen,len(str(s)))

    while jobsLeft >0:
        time.sleep(10)
        Gstatus = header + "\n" + delim + "\n"
        eta = datetime.now(pytz.UTC)
        for s in jobsDict.keys():
            Gstatus += "|" + str(s).center(slen," ") + "|"
            for f in jobsDict[s].keys():
                status = jobsDict[s][f]["job"].status()
                if status.name == "QUEUED" :
                    queueInfo = jobsDict[s][f]['job'].queue_info()
                    if queueInfo != None and queueInfo.estimated_complete_time > eta:
                        eta = queueInfo.estimated_complete_time
                    Gstatus += f"{status.name}({jobsDict[s][f]['job'].queue_position()})".center(max(len(f)+2,11)," ") + "|"

                else :
                    Gstatus += status.name.center(max(len(f)+2,11)," ") + "|"

                if jobsDict[s][f]["status"] != None :
                    continue
                
                if (status.name in ['DONE', 'CANCELLED', 'ERROR']):
                    jobsLeft -= 1
                    
                    if status.name == 'DONE':
                        counts = jobsDict[s][f]["job"].result().get_counts()
                        
                        jobsDict[s][f]["expeResult"] = {k:counts[k] for k in counts}

                jobsDict[s][f]["status"] = status.name
            Gstatus += "\n"
        Gstatus += delim + f"\nJobs Left : {jobsLeft} ETA : " + prettyDelta(eta - datetime.now(pytz.UTC))
        printer.reprint(Gstatus)

def autoSelectQComputer(filter:"list[str]",builder:circuitBuilder,states:list,maxQueuedJob:int = 10) -> IBMQBackend:
    provider=IBMQ.get_provider('ibm-q')
    simulator=Aer.get_backend('qasm_simulator')
    
    # avoid side effect
    filter = filter.copy()
    
    # filter q computers with too much pending jobs
    for f in filter:
        qcomputer = provider.get_backend(f)
        if qcomputer.status().pending_jobs > maxQueuedJob:
            filter.remove(f)
            continue
        joblim = qcomputer.job_limit()
        if joblim.maximum_jobs - joblim.active_jobs > len(states):
            print(f"[Warning] number of states is superior to the number of available jobs for the backend {f} autoselection may take longer than expected")
    
    # data holder
    jobs = {}
    
    # preparing fancy print
    slen = max([len(str(s)) for s in states])

    filterLen = {f:min(len(f),11) for f in filter}

    header = "|" + "state".center(slen," ") + "|"
    delim = "+" + "-" * slen + "+"

    for f in filter:
        header += f.center(max(len(f)+2,11)," ") + "|"
        delim += "-" * max(len(f)+2,11) + "+"
        
    circuit = QuantumCircuit(*builder.minCiruitsize)
    circuit = builder(circuit)
    display(circuit.draw("mpl"))
    
    printer = Reprinter()
    
    printer.print(header)
    printer.print(delim)
    
    
    
    # creating jobs
    for s in states:

        line = "|" + str(s).center(slen," ") + "|"

        builder.setState(s)
        circuit = QuantumCircuit(*builder.minCiruitsize)
        circuit = builder(circuit)
        
        jobs[s],line = startJobs(circuit,filter,line)
        
        counts = execute(circuit,backend = simulator,shots=512).result().get_counts()
        jobs[s]["simResult"] = {k:counts[k] for k in counts}
            
        printer.print(line)
    printer.print(delim)
        
    jobsLeft = len(states)*len(filter)
    printer.print(f"Jobs Left : {jobsLeft}")
    
    # waiting for all jobs to end and gathering data
    monitorJobs(jobs,header,delim,printer)

    errors = {f:{"count":0,"tot":0} for f in filter}
    
    # processing error
    for s in states:
        for f in filter:
            if jobs[s][f]["status"] == 'DONE':
                errors[f]["tot"] += 512
                for k in jobs[s]["simResult"].keys():
                    errors[f]["count"] += abs(jobs[s][f]["expeResult"][k] - jobs[s]["simResult"][k])
    
    back = None
    error = 2.0
    
    #finding the best q computer
    for (k,v) in errors.items() :
        if (v["tot"]>0):
            e = float(v["count"])/v["tot"]
            if e<error:
                error = e
                back = k
    
    print(error)
    return back


from string import Formatter

def prettyDelta(timed:timedelta) -> str:
    s = int(timedelta.total_seconds())
    m,s = divmod(s,60)
    h,m = divmod(m,60)
    d,h = divmod(h,24)

    string = ""

    if(d >0):
        string += "{0:02}d".format(d)

    if (h>0):
        string += "{0:02}h".format(h)

    if (m>0):
        string += "{0:02}m".format(m)

    if (s>0):
        string += "{0:02}s".format(s)
    return string
            

proco = ["ibmq_manila","ibmq_santiago","ibmq_athens","ibmq_belem","ibmq_quito","ibmqx2"]#,"ibmq_lima"

qcomputer = autoSelectQComputer(proco, circuitBuilder((3,1),teleportBuilder,Qubit.QubitValidator,Qubit(0,1)), [Qubit(1,0),Qubit(0,1)] )


    