# https://github.com/vexandmore/QKDproject/blob/45ed1e53e3ac52d8bb3878afd137a4e25359c16a/src/main/qkaImplementationFINAL.py
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:26:44 2021

@author: Raphael
"""

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.visualization import *
import numpy as np
import random
import json

class Sender:
    def __init__ (self,x, bits, bases):        
        self.length = len(bases)
        self.sendBits = bits
        self.sendBases = bases
  
    def makeSendCircuits(self):
        circuits = []
        for i in range(self.length):
            circuit = QuantumCircuit(1,1)
            if (self.sendBases[i] == 0):
                if (self.sendBits[i] == 0):
                    pass
                else:
                    circuit.x(0)
            else:
                if (self.sendBits[i] == 0):
                    circuit.h(0)
                else:
                    circuit.x(0)
                    circuit.h(0)
            circuits.append(circuit)
        return circuits


class Receiver:
    def __init__(self, x, bases):        
        self.length = len(bases)
        self.receiveBases = bases 
        
    def makeReceiveCircuits(self):
        recCircuits = []
        for i in range(self.length):
            recCircuit = QuantumCircuit(1,1)
            if (self.receiveBases[i] == 1):
                recCircuit.h(0)
                recCircuit.measure(0,0)
            else:
                recCircuit.measure(0,0)
            recCircuits.append(recCircuit)
        return recCircuits

class Contributor(Sender, Receiver):    
            
    def areHashesEqual(K, H):
        """turn K into H_ into binary because H is in binary"""
        H_ = Contributor.decimalToBinary(hash(K))
        return Contributor.checkHash(H_, H)
    
    def removeDecoy(self, ls, pos):
        """returns list without decoys"""
        new_ls = []
        for i in range(len(ls)):
            if i in pos:
                pass
            else:
                new_ls.append(ls[i])
        return new_ls
     
    def checkHash(hash1, hash2):
        temp = (hash1 == hash2)
        for i in range(temp):
            if temp[i] == False:
                return True
        return False
        
    def checkDecoy(self, dS, dC):
        """returns if there has been eavesdropping (False), no eavesdropping (True)"""
        for i in dS:
            if i != self.bi_dS[i]:
                return False
        for i in dC:
            if i != self.bi_dC[i]:
                return False
        else:
            return True
   
    def findDecoy(ls, pos):
        """returns list of decoys found within given list"""
        decoys = []
        for i in range(len(ls)):
            for j in range(len(pos)):
                if pos[j] == i:
                    decoys.append(ls[i])
        return decoys  
    
    def receiveDecoy(self, ls, pos, ba):
        """returns decoy measurements"""
        len_ls = len(ls)
        decoys = Contributor.findDecoy(ls, pos)
        recCircuits = Contributor.createRecCircuits(len_ls, ba)
        return self.measure(len_ls, decoys, recCircuits)
    
    def measure(self, length, sendCircuits, recCircuits):
        mes = []
        for i in range(len(sendCircuits)):
            sendAndReceive = sendCircuits[i] + recCircuits[i]
            result = execute(sendAndReceive, self.backend, shots=1, memory=True).result()
            measured_bit = int(result.get_memory()[0])
            mes.append(measured_bit)
        return mes
        
    def measure2(self, ls, ba):
        sendCircuits = ls[:]
        recCircuits = Contributor.createRecCircuits(len(ba), ba)
        mes = []
        for i in range(len(sendCircuits)):
            sendAndReceive = sendCircuits[i] + recCircuits[i]
            result = execute(sendAndReceive, self.backend, shots=1, memory=True).result()
            measured_bit = int(result.get_memory()[0])
            mes.append(measured_bit)
        return mes
    
    def createRecCircuits(len_ls, ba):
        """just creates measurement gates"""
        receiver = Receiver(len_ls, ba)
        return receiver.makeReceiveCircuits()
        
    def insertDecoys(dls, ls, pos):
        x = ls[:]
        for i in range(len(ls)):
            for j in range(len(pos)):
                if pos[j] == i:
                    x[i:i] = [dls[j]]
        return x
    
    
    def getDecoyPos(len_dls, len_ls):
        pos = random.sample(range(len_ls), len_dls)
        pos.sort()
        return pos
    
    def send(length, bits, bases):
        return Sender(length, bits, bases)
         
    
    def cbits(n):
        rng = np.random.default_rng()
        return rng.integers(0,1,n, endpoint=True)
    
    def decimalToBinary(n):
        if n > 0:
            return bin(n)[2:]
        else:
            return bin(n)[3:]

    def unshuffle(self, shuffled_ls, seed):
        n = len(shuffled_ls)
        p = [i for i in range(1, n + 1)]
        random.Random(seed).shuffle(p)
        zipped_ls = list(zip(shuffled_ls, p))
        zipped_ls.sort(key=lambda x: x[1])
        return [a for (a, b) in zipped_ls]
    
    def __init__(self,securityProperty, backend, bi_dS, bi_dC, Kkey):
        if securityProperty == None:
            self.bi_dS = bi_dS
            self.bi_dC = bi_dC
            self.Kkey = Kkey
            self.backend = backend
        
        else:
            self.securityProperty = securityProperty
            self.backend = backend
            n = 24
            """NEW DEFINITION
            s: shuffled list, s_: encoded shuffled list, s__: encoded shuffled list with decoys"""
        
            self.K = Contributor.cbits(n)
            self.seed = np.random.randint(1,101)
        
            Ktemp = (''.join(map(str, self.K)))
            self.Kkey = int(Ktemp)
            """what is different about the new definition is that now the shuffled list
            is encoded, and not that the encoded list is shuffled"""

            random.Random(self.seed).shuffle(self.K)
            S = self.K[:]
            self.K = self.unshuffle(self.K, self.seed)
        
            self.bS = list(Contributor.cbits(len(S)))
            H_ = Contributor.decimalToBinary(hash(Ktemp))
            H = [int(x) for x in str(H_)]
            self.bH = list(Contributor.cbits(len(H)))
          
            sentS = Contributor.send(len(S), S, self.bS)
            sentH = Contributor.send(len(H), H, self.bH)  
            
            S_ = sentS.makeSendCircuits()
            C = sentH.makeSendCircuits()
            
            """now for the decoys"""
            len_dS = int(len(S_) * securityProperty)
            len_dC = int(len(C) * securityProperty)
        
            self.bi_dS = Contributor.cbits(len_dS)
            self.ba_dS = Contributor.cbits(len_dS)
        
            self.bi_dC = Contributor.cbits(len_dC)
            self.ba_dC = Contributor.cbits(len_dC)
        
            sentdS = Contributor.send(len_dS, self.bi_dS, self.ba_dS)
            sentdC = Contributor.send(len_dC, self.bi_dC, self.ba_dC)
        
            dS = sentdS.makeSendCircuits()
            dC = sentdC.makeSendCircuits()
            """now we have decoys, time to input them into lists, remembering positions
            first get positions"""
            self.pos_dS = Contributor.getDecoyPos(len_dS, len(S_))
            self.pos_dC = Contributor.getDecoyPos(len_dC, len(C))
            """now, insert"""   
            self.S__ = Contributor.insertDecoys(dS, S_, self.pos_dS)
            self.C_ = Contributor.insertDecoys(dC, C, self.pos_dC)
            


def GiveData(strsecurityProperty, backend):
    securityProperty = float(strsecurityProperty)
    user = Contributor(securityProperty, backend, None, None, None)
    #Send the bits across the channel
    strS__ = []
    strC_ = []   
    for i in range(len(user.S__)):
        strS__.append(user.S__[i].qasm())     
    for i in range(len(user.C_)):
        strC_.append(user.C_[i].qasm()) 
        
    json_strS__ = json.dumps(strS__)
    json_strC_ = json.dumps(strC_)
    
    return str(user.Kkey), str(user.bi_dS), str(user.bi_dC), json_strS__, json_strC_, str(user.pos_dS), str(user.pos_dC), str(user.ba_dS), str(user.ba_dC),str(user.seed), str(user.bS), str(user.bH)
    """
    print(user.Kkey)
    print(user.bi_dS)
    print(user.bi_dC)
    print(json_strS__)
    print(json_strC_)
    print(user.pos_dS)
    print(user.pos_dC)
    print(user.ba_dS)
    print(user.ba_dC)
    print(user.seed)
    print(user.bS)
    print(user.bH) 
    """

def strToLs(stringls):
    ls = stringls[1:len(stringls)-1]
    ls = ls.replace(" ","")    
    ls = [int(x) for x in ls]
    return ls

def Intercepted(json_strS__, json_strC_, backend):
    Eve = Eavesdropper(backend)
     
    strS__ = json.loads(json_strS__)
    strC_ = json.loads(json_strC_)
    
    S__ = []
    C_ = []
    
    for i in range(len(strS__)):
        S__.append(QuantumCircuit().from_qasm_str(strS__[i])) 
    
    for i in range(len(strC_)):
        C_.append(QuantumCircuit().from_qasm_str(strC_[i]))     
          
    #Oh no! Eve has intercepted the messages!
    e1 = Eve.interceptAndResend(S__)
    e2 = Eve.interceptAndResend(C_)
    strS__2 = []
    strC_2 = []   
    for i in range(len(e1)):
        strS__2.append(e1[i].qasm())     
    for i in range(len(e2)):
        strC_2.append(e2[i].qasm())  
    
    json_strS__2 = json.dumps(strS__2)
    json_strC_2 = json.dumps(strC_2)    
    
    print(json_strS__2)
    print(json_strC_2)

def ReceiveData1(json_strS__, json_strC_, strpos_dS, strpos_dC, strba_dS, strba_dC, backend):
#returns decoy bits + lists without decoys
    user = Contributor(None, backend, None, None, None)
    strS__ = json.loads(json_strS__)
    strC_ = json.loads(json_strC_)        
    S__ = []
    C_ = []   
    
    for i in range(len(strS__)):
        S__.append(QuantumCircuit().from_qasm_str(strS__[i])) 
    
    for i in range(len(strC_)):
        C_.append(QuantumCircuit().from_qasm_str(strC_[i])) 
    
    pos_dS = json.loads(strpos_dS)
    pos_dC = json.loads(strpos_dC)
    ba_dS = strToLs(strba_dS)
    ba_dC = strToLs(strba_dC)

    dS = user.receiveDecoy(S__, pos_dS, ba_dS)
    dC = user.receiveDecoy(C_, pos_dC, ba_dC) 
    
    S_ = user.removeDecoy(S__, pos_dS)
    C = user.removeDecoy(C_, pos_dC)
    
    strS_ = []
    strC = []   
    for i in range(len(S_)):
        strS_.append(S_[i].qasm())     
    for i in range(len(C)):
        strC.append(C[i].qasm()) 
    
        
    json_S_ = json.dumps(strS_)
    json_C = json.dumps(strC)
    
    return str(dS), str(dC), json_S_, json_C

def securityCheck(d, bi_d):
    print(json.loads(d) == strToLs(bi_d))


#ReceiveData2 ~ MakeKey
def makeKey(ownKkey, jS_, jC, seed, bS, bH, backend):
    ownKkey = int(ownKkey)
    sS_ = json.loads(jS_)
    sC = json.loads(jC)
    S_ = []
    C = []
    for i in range(len(sS_)):
        S_.append(QuantumCircuit().from_qasm_str(sS_[i])) 
    for i in range(len(sC)):
        C.append(QuantumCircuit().from_qasm_str(sC[i])) 
    seed = int(seed)
    bS = json.loads(bS)
    bH = json.loads(bH)
    user = Contributor(None, backend, None, None, ownKkey)
    S = user.measure2(S_, bS)
    H = user.measure2(C, bH)
    K = int(''.join(map(str, user.unshuffle(S, seed))))
    #securityCheck2(H,K)
    print(K^user.Kkey)

"""
#Very unreliable
def securityCheck2(H,K):
    Htemp = int(''.join(map(str, H)))
    H_ = int(Contributor.decimalToBinary(hash(str(K)))) 
    #print(Htemp)
    #print(H_)
    print(Htemp == H_)
"""  


def mainTest():
    securityProperty = 0.5
    backend = Aer.get_backend('qasm_simulator')
    #GiveData
    #Alice
    strownKkey, strbi_dS, strbi_dC, strS__, strC_, pos_dS, pos_dC, ba_dS, ba_dC, seed, bS, bH = GiveData(securityProperty, backend)
    #Bob
    strownKkey2, strbi_dS2, strbi_dC2, strS__2, strC_2, pos_dS2, pos_dC2, ba_dS2, ba_dC2, seed2, bS2, bH2 = GiveData(securityProperty, backend)
    #ReceiveData1
    #Alice
    AdS, AdC, AS_, AC = ReceiveData1(strS__2, strC_2, pos_dS2, pos_dC2, ba_dS2, ba_dC2, backend)
    #Bob
    BdS, BdC, BS_, BC = ReceiveData1(strS__, strC_, pos_dS, pos_dC, ba_dS, ba_dC, backend)
    #Alice
    #securityCheck(BdS, strbi_dS)
    #securityCheck(BdC, strbi_dC)
    makeKey(strownKkey, AS_, AC, seed2, bS2, bH2, backend)    
    #Bob
    #securityCheck(AdS, strbi_dS2)
    #securityCheck(AdC, strbi_dC2)
    makeKey(strownKkey2, BS_, BC, seed, bS, bH, backend)

def main():
    backend = Aer.get_backend('qasm_simulator')
    while True:
        inArgs = input().split('#')
        if len(inArgs) == 1: #GiveData
            securityProperty = inArgs[0]
            Kkey, bi_dS, bi_dC, S__, C_, pos_dS, pos_dC, ba_dS, ba_dC, seed, bS, bH = GiveData(securityProperty, backend)
            print(Kkey)
            print(bi_dS)
            print(bi_dC)
            print(S__)
            print(C_)
            print(pos_dS)
            print(pos_dC)
            print(ba_dS)
            print(ba_dC)
            print(seed)
            print(bS)
            print(bH)
        
        if len(inArgs) == 4: #SecurityCheck
            dS = inArgs[0]
            dC = inArgs[1]
            bi_dS = inArgs[2]
            bi_dC = inArgs[3]
            securityCheck(dS, bi_dS)
            securityCheck(dC, bi_dC)
                    
        if len(inArgs) == 6: #ReceiveData1
            S__ = inArgs[0]
            C_ = inArgs[1]
            pos_dS = inArgs[2]
            pos_dC = inArgs[3]
            ba_dS = inArgs[4]
            ba_dC = inArgs[5]
            dS, dC, S_, C = ReceiveData1(S__, C_, pos_dS, pos_dC, ba_dS, ba_dC, backend)
            print(dS)
            print(dC)
            print(S_)
            print(C)
              
        #have to add a "" at the end to make this work
        if len(inArgs) == 7: #makeKey(ALICEKkey, ALICES_, ALICEAC, BOBseed, BOBbS, BOBbH, backend)    
            Kkey = inArgs[0]
            S_ = inArgs[1]
            C = inArgs[2]
            seed = inArgs[3]
            bS = inArgs[4]
            bH = inArgs[5]
            makeKey(Kkey, S_, C, seed, bS, bH, backend)
        
        if len(inArgs) == 2: #Eve intercepts
            S__ = inArgs[0]
            C_ = inArgs[1]
            Interecepted(S__, C_, backend)
            
 
#mainTest()
main()


    
