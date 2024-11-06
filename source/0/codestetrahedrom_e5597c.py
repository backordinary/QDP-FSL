# https://github.com/Danuzco/tetrancomp/blob/052e0ea7ae1594c0784b4b1e227bc52ee1319128/codesTetrahedrom.py
import numpy as np
import qiskit.quantum_info as qi
import itertools
import os
import re
import math
from numpy import cross, eye, dot
from scipy.linalg import expm, norm


def vectorState(bloch_vector):
    x,y,z = bloch_vector
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    I = np.identity(2)
    
    rho = (I + (x*sx + y*sy + z*sz))/2
    rho = rho/np.trace(rho)
    _, eigenv = np.linalg.eigh(rho)
    
    return eigenv[:,1]


def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))


class nCompoundTetra():
    
    """
    Valid for n= 1,2,3 and 4
    """
    
    def __init__(self,baseTetrahedra,n):
        self.vectors = baseTetrahedra
        self.order = n
        self.theta = np.pi/n
        
    def create_nCompoundTetrahedron(self):
        n = self.order
        vecs = self.vectors
        theta = self.theta
        k = (vecs[0] + vecs[1])/2
        T = []
        for i in range(0,n):
            rvects = M(k, i*theta)@np.array(vecs).T
            rvects = np.hstack( ( np.zeros((4,3)), rvects.T) )

            T.append(rvects)
            
        return T
    
    
    def getEdges(self):
        T = self.create_nCompoundTetrahedron()
        X = {}
        Y = {}
        Z = {}

        for j,t in enumerate(T):
            X[j] = []
            Y[j] = []
            Z[j] = []
            for u,v in list(itertools.combinations( t[:,3:], r = 2)):
                r1,r2,r3 = u
                X[j].append(r1)
                Y[j].append(r2)
                Z[j].append(r3)

                r1,r2,r3 = v
                X[j].append(r1)
                Y[j].append(r2)
                Z[j].append(r3)
        return X.values(),Y.values(),Z.values()
    
    def getVectorMeasurements(self):
        T = self.create_nCompoundTetrahedron()
        Measurements = {}
        for j,t in enumerate(T):
            Measurements[j] = []
            for p in t:
                vs = vectorState(p[3:])
                Measurements[j].append(vs)
        return Measurements