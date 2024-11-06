# https://github.com/soosub/bachelor-thesis/blob/7da3447e1fd77a9d94f79b7939dfc952d4f0e11e/Implementation/Zhou_3/INTERP.py
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:43:18 2020

@author: joost
"""

from QAOA import QAOA
import qiskit
from tqdm import tqdm

class INTERP(QAOA):
    def give_angles(self):
        '''returns all the angles calculated in the process of the INTERP method
        Note, if you start with initial angles with p > 1, watch out for the ordering
        The length of the angles factors might be more useful in that case'''
        return self.gammas, self.betas       
    
    def get_angles_INTERP(self, p, initial_gamma = [0.35], initial_beta = [0.8], remove_degeneracy = True):
        '''
        Finds quasi-optimal angles for a given graph G
        Iteratively increase p, using an interpolation-based strategy
        to find initial points for optimization
        '''
        assert len(initial_gamma) == len(initial_beta), "Angle arrays must be of the same length"
        
        def INTERP_next_angles(angles):
            '''Takes in a set of p angles and returns a new set of p+1 angles using the INTERP method'''
            
            assert len(gamma) == len(beta), "Both lists of parameters gamma, beta must be of the same length"
            
            p = len(angles)
            
            new_angles = [angles[0]] # i = 1, the first angles coincide
            for i in range(2, p+1):
                new_angles.append( (i-1)/p*angles[i-2] + (p-i+1)/p*angles[i-1]) # i in {2, ... , p}, interpolation 
            new_angles.append(angles[-1]) # i = p+1, the last angles coincide
            return new_angles
    
        # find first set of angles
        gamma,beta = self.find_local_angles(initial_gamma,initial_beta)
        if remove_degeneracy:
            gamma,beta = QAOA.remove_degeneracies_list(gamma,beta)
        self.gammas_list = [gamma]
        self.betas_list = [beta]
        
        for i in tqdm(range(p-len(initial_gamma))):
            gamma_ansatz,beta_ansatz = INTERP_next_angles(gamma), INTERP_next_angles(beta) # new initial points
            gamma,beta = self.find_local_angles(gamma_ansatz,beta_ansatz) # new (local) optima  
            if remove_degeneracy:
                gamma,beta = QAOA.remove_degeneracies_list(gamma,beta)
            self.gammas_list.append(gamma)
            self.betas_list.append(beta)
        
        return gamma, beta