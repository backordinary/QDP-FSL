# https://github.com/zemaitistrys/bubble_swap/blob/e037ff873a0a1c6e474a6ca52ced4c8cfe1dc5d6/bubble_swap/random_swap_benchmark.py
import qiskit
import scipy as sp
import pandas as pd
import numpy as np
from numpy import pi
import math as mt
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from qiskit.circuit import Clbit, Qubit
from qiskit.providers.aer.noise import NoiseModel
from main import GridCircuit

# code for benchmarking bubble swap, which builds upon the grid circuit code
# this includes routing algorithms to entangle two distance qubit with bubble swaps,
# either with one bubble or two bubbles

class RandomSwapBenchmark:
    
    # class created for benchmarking bubble swap against regular swap within an x * y 
    # grid of qubits, which can be interacted with through several layers of abstraction
    
    def __init__(self, x, y, provider, print_statements = False):
        self.x = x
        self.y = y
        self.provider = provider
        self.print_statements = print_statements # there's some print statements
        # as a sanity check that one can turn on
        
    def randomized_benchmark(self, seed, no_coord_samples, no_state_samples, postselect = True, mitigate = True):
        
        # Benchmarks regular, single bubble- and double bubble- swap-based routing 
        # to entangle two states randomly picked from an x * y grid of qubits.
        # More specifically, all qubits (except the bubbles) in the grid are initialized
        # in random single-qubit states, then the coordinates of the bubble(s) and of two
        # nonzero states (I and J) are randomly picked. Then the I and J states are "pushed together"
        # through swaps (regular or bubble) until they are neighbouring - this
        # imitates a segment of a quantum circuit where I and J need a two-qubit gate
        # between them. After pushing together the states, they are measured and compared
        # to their initial values - specifically we measure in Z and X bases, calculate
        # the KL divergence in both cases and add it up as the metric of similarity.
        
        # This is done for no_coord_samples different sets of coords, and for each 
        # set of coords it's repeated no_state_samples times with different states
        # on each qubit. Everything is saved in a pandas dataframe and returned.
        
        # generating list of coordinates to sample 
        all_coords = np.array([i for i in range(self.x * self.y)])
        res = []
        for i in range(no_coord_samples):
            for j in range(no_state_samples):
                np.random.seed(i)
                q_i, q_j, q_b = np.random.choice(all_coords, 3, replace = False)
                x_i = q_i % self.x
                y_i = q_i // self.x
                x_j = q_j % self.x
                y_j = q_j // self.x
                x_b = q_b % self.x
                y_b = q_b // self.x
                # deprecated; should add x_b2 and y_b2 to account for the addition
                # of a double bubble swap circuit
                res.append(self.routing_test(x_i, y_i, x_j, y_j, x_b, y_b, postselect, mitigate, seed = j))
        res_df = pd.DataFrame(res)
        return res_df
        
    def routing_test(self, x_i, y_i, x_j, y_j, x_b1, y_b1, x_b2, y_b2, postselect = True, mitigate = True, seed = 123):
        
        # see comments of randomized_benchmark function for background
        
        # single bubble swap
        bsc = self.BubbleSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, print_statements = self.print_statements, seed = seed)
        bsc.push_states_together()
        
        # regular bubble swap
        rsc = self.RegularSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, bsc.swap_path, print_statements = self.print_statements, seed = seed)
        rsc.push_states_together()
        
        # noiseless version to compare to
        rsc_noiseless = self.RegularSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, bsc.swap_path, noiseless = True, seed = seed)
        rsc_noiseless.push_states_together()
        
        # double bubble swap
        dbsc = self.DoubleBubbleSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, x_b2, y_b2, seed = seed)
        dbsc.push_i()
        
        # now repeat all but add a hadamard at the end
        bsc_x = self.BubbleSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, seed = seed)
        bsc_x.push_states_together()
        rsc_x = self.RegularSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, bsc.swap_path, seed = seed)
        rsc_x.push_states_together()
        rsc_x_noiseless = self.RegularSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, bsc.swap_path, noiseless = True, seed = seed)
        rsc_x_noiseless.push_states_together()
        dbsc_x = self.DoubleBubbleSwapCircuit(self.provider, self.x, self.y, x_i, y_i, x_j, y_j, x_b1, y_b1, x_b2, y_b2, seed = seed)
        dbsc_x.push_i()
        
        for sc in [bsc_x, rsc_x, rsc_x_noiseless, dbsc_x]:
            sc.gcirc.gate('h', [sc.x_i, sc.y_i])
            sc.gcirc.gate('h', [sc.x_j, sc.y_j])
        
        
        # then some kind of tomography; for a start just measure in both bases
        # and find kl divergence
        res = []
        for sc in [bsc, rsc, rsc_noiseless, dbsc, bsc_x, rsc_x, rsc_x_noiseless, dbsc_x]:
            sc.gcirc.measure([sc.x_i,sc.y_i], 0)
            sc.gcirc.measure([sc.x_j,sc.y_j], 1)
            res.append(sc.gcirc.run(10000, postselect, mitigate))
            
        kl_bubble = self.kl(model = res[2], target = res[0]) + self.kl(model = res[6], target = res[4])
        
        kl_regular = self.kl(model = res[2], target = res[1]) + self.kl(model = res[6], target = res[5])
        
        kl_double_bubble = self.kl(model = res[2], target = res[3]) + self.kl(model = res[6], target = res[7])
        
        return {
            'x_i': x_i,
            'y_i': y_i,
            'x_j': x_j,
            'y_j': y_j,
            'x_b1': x_b1,
            'y_b1': y_b1,
            'x_b2': x_b2,
            'y_b2': y_b2,
            'KL bubble': kl_bubble,
            'KL regular': kl_regular,
            'KL double bubble': kl_double_bubble,
            'state seed': seed,
            'postselected': int(postselect),
            'mitigated': int(mitigate)}
    
    
    def dict_to_array(self, probs_dict, epsilon = 1e-16):
        
        # converts dict of counts into array of counts sorted by bitstring value
        keys_list = list(probs_dict.keys())
        keyno = len(keys_list)
        bitno = len(keys_list[0])
        nocounts = 2**bitno - keyno
        beta = epsilon * nocounts / keyno
        probs_array = np.zeros(2**bitno)
        for i in range(2**bitno):
            i_bin = format(i, '0'+str(bitno)+'b')
            if i_bin in probs_dict:
                probs_array[i] = probs_dict[i_bin] - beta
            else:
                probs_array[i] = epsilon
        return probs_array   
    
    def kl(self, model, target):
        # finds kl between model and target
        if type(model) is dict:
            model = self.dict_to_array(model)
        if type(target) is dict:
            target = self.dict_to_array(target)
        return np.dot(target, np.log(target/model)) 
    
    class BubbleSwapCircuit:
        
        # creates class for routing I and J by using bubble B through push_states_together
        # function. the routing algorithm is the following:
        # 1. find neighbours of I and neighbours of J that I/J could switch with to decrease
        #    the distance between I and J. pick the neigbour closest to B as the "target"
        #    for the bubble.
        # 2. look at the neighbours of B and find the one (or pick one if several) that
        #    decreases distance to the target. go there
        # 3. repeat step 2 until you are at the target
        # 4. switch I/J with the bubble currently at the target.
        # 5. go back to step 1 and repeat until I and J are neighbours.
        
        def __init__(self, provider, x, y, x_i, y_i, x_j, y_j, x_b, y_b, angles = None, seed = 123, noiseless = False, print_statements = False):
            
            self.provider = provider
            self.x = x
            self.y = y
            
            # point I
            self.x_i = x_i
            self.y_i = y_i
            
            # point J
            self.x_j = x_j
            self.y_j = y_j
            
            # bubble B
            self.x_b = x_b
            self.y_b = y_b
            
            # angles: either (no_qubits, 3) sized array or None
            # this is the angles of the u rotation applied to each of the (non-bubble) qubits
            self.angles = angles
            
            # seed for generating angles
            self.seed = seed
            
            # whether we want noiseless simulator (put true to get your baseline to compare to)
            self.noiseless = noiseless
            
            self.gcirc = self.CreateCirc()
            
            # path of I/J swaps that the 
            # list of coordinate pairs (data & bubble qubit)
            self.swap_path = [] 
            self.print_statements = print_statements
            # showing the path the data qubits took
                            
        def current_distance(self):
            return self.manhattan_distance(self.x_i, self.y_i, self.x_j, self.y_j)
        def CreateCirc(self):
            # creates the circuit
            if self.noiseless:
                gcirc = GridCircuit(provider = self.provider, x = self.x, y = self.y, max_no_swaps = (self.x+self.y)**2, noise_model = None)
            else:
                gcirc = GridCircuit(provider = self.provider, x = self.x, y = self.y, max_no_swaps = (self.x+self.y)**2)
                
            # populates the qubits with random states
            if self.angles is None:
                np.random.seed(self.seed)
                self.angles = []
                for i in range(gcirc.no_qubits):
                    self.angles.append(np.random.rand(3)*pi)
            for i in range(self.x):
                for j in range(self.y):
                    qubit = gcirc.convert_coords(i,j)
                    if [i,j] == [self.x_i, self.y_i]:
                        self.angles_i = self.angles[qubit]
                    if [i,j] == [self.x_j, self.y_j]:
                        self.angles_j = self.angles[qubit]
                    if [i,j] != [self.x_b, self.y_b]:
                        gcirc.gate('u', [i,j], parameters = self.angles[qubit])
            return gcirc
        def manhattan_distance(self, x1, y1, x2, y2):
            # decisions to swap are made based on minimizing distances travelled;
            # to prevent going to a point outside the grid we return 1000 for distance if any
            # point in the distance calculation is outside the grid
            for x in [x1, x2]:
                if x < 0 or x > self.x:
                    return 1000
            for y in [y1, y2]:
                if y < 0 or y > self.y:
                    return 1000
            return abs(x1-x2)+abs(y1-y2)
        def travel_distance(self, p1, p2):
            
            # returns distance bubble would have to travel to get from p1 to p2 avoiding I and J
            for p in [p1,p2]:
                
                # if either point is outside boundary, it is "infinitely far away"
                if p[0] < 0 or p[1] < 0 or p[0] > self.x-1 or p[1] > self.y-1:
                    return 1000
                
                # if either point matches I or J, it can't go around it, so is also "infinitely far away"
                for q in [[self.x_i, self.y_i], [self.x_j, self.y_j]]:
                    if p == q:
                        return 1000
            
            # if either I or J is in a straight line between target and bubble,
            # return manhattan distance + 2, otherwise return manhattan distance
            if p1[0] == p2[0] and ((self.x_i == p1[0] and (self.y_i - p1[1])*(self.y_i - p2[1]) < 0) or (self.x_j == p1[0] and (self.y_j - p1[1])*(self.y_j - p2[1]) < 0)):
                return self.manhattan_distance(*p1, *p2) + 2
            if p1[1] == p2[1] and ((self.y_i == p1[1] and (self.x_i - p1[0])*(self.x_i - p2[0]) < 0) or (self.y_j == p1[1] and (self.x_j - p1[0])*(self.x_j - p2[0]) < 0)):
                return self.manhattan_distance(*p1, *p2) + 2
            return self.manhattan_distance(*p1, *p2)
        
        def move_bubble(self, bubble_target, data_target):
            
            # moves bubble to the bubble_target point where it will be able to switch with data_target
            distance = self.travel_distance([self.x_b, self.y_b], bubble_target)
            if self.print_statements:
                print('distance = ' + str(distance))
            # if we're there already, bubble swap with data qubit and update coordinate pointers
            if distance == 0:
                self.gcirc.bswap(data_target, bubble_target)
                self.swap_path.append([data_target, bubble_target])
                if data_target == [self.x_i, self.y_i]:
                    self.x_i, self.y_i = bubble_target
                    self.x_b, self.y_b = data_target
                    if self.print_statements:
                        print('swapped I at ' + str(data_target) + ' with bubble at ' + str(bubble_target))
                    return
                if data_target == [self.x_j, self.y_j]:
                    self.x_j, self.y_j = bubble_target
                    self.x_b, self.y_b = data_target
                    if self.print_statements:
                        print('swapped J at ' + str(data_target) + ' with bubble at ' + str(bubble_target))
                    return
            # if we're not there, find where to go to decrease the distance to the target
            else:
                # list where bubble could go
                bubble_next_step_candidates = [[self.x_b+1, self.y_b],[self.x_b-1, self.y_b],[self.x_b, self.y_b+1],[self.x_b, self.y_b-1]]
                
                # find the distance to target in each case
                distances_to_target = []
                for candidate in bubble_next_step_candidates:
                    distances_to_target.append(self.travel_distance(candidate, bubble_target))
                
                # take the coordinates corresponding to the smallest distance to target
                bubble_next_step = bubble_next_step_candidates[distances_to_target.index(min(distances_to_target))]
                
                # swap in that direction
                self.gcirc.bswap(bubble_next_step, [self.x_b, self.y_b])
                if self.print_statements:
                    print('moved bubble from ' + str([self.x_b, self.y_b]) + ' to ' + str(bubble_next_step))
                self.x_b, self.y_b = bubble_next_step
                if self.print_statements:
                    print('bubble at ' + str([self.x_b,self.y_b]))
                    
                # repeat
                return self.move_bubble(bubble_target, data_target)
                    
        def push_states_together(self):
            if self.current_distance() == 1:
                return
            else:
                
                # find neighbours of I/J that I/J could move to to decrease distance between them
                if self.x_i == self.x_j:
                    candidate_bubble_targets = [[self.x_i, min(self.y_i, self.y_j)+1],[self.x_i, max(self.y_i, self.y_j)-1]]
                    corresp_data_targets = [[self.x_i, min(self.y_i, self.y_j)],[self.x_i, max(self.y_i, self.y_j)]]
                if self.y_i == self.y_j:
                    candidate_bubble_targets = [[min(self.x_i, self.x_j)+1,self.y_i],[max(self.x_i, self.x_j)-1,self.y_i]]
                    corresp_data_targets = [[min(self.x_i, self.x_j),self.y_i],[max(self.x_i, self.x_j),self.y_i]]
                if self.x_i > self.x_j and self.y_i > self.y_j:
                    candidate_bubble_targets = [[self.x_i-1,self.y_i],[self.x_i,self.y_i-1],[self.x_j+1,self.y_j],[self.x_j,self.y_j+1]]
                    corresp_data_targets = [[self.x_i,self.y_i],[self.x_i,self.y_i],[self.x_j,self.y_j],[self.x_j,self.y_j]]
                if self.x_i > self.x_j and self.y_i < self.y_j:
                    candidate_bubble_targets = [[self.x_i-1,self.y_i],[self.x_i,self.y_i+1],[self.x_j+1,self.y_j],[self.x_j,self.y_j-1]]
                    corresp_data_targets = [[self.x_i,self.y_i],[self.x_i,self.y_i],[self.x_j,self.y_j],[self.x_j,self.y_j]]
                if self.x_i < self.x_j and self.y_i > self.y_j:
                    candidate_bubble_targets = [[self.x_i+1,self.y_i],[self.x_i,self.y_i-1],[self.x_j-1,self.y_j],[self.x_j,self.y_j+1]]
                    corresp_data_targets = [[self.x_i,self.y_i],[self.x_i,self.y_i],[self.x_j,self.y_j],[self.x_j,self.y_j]]
                if self.x_i < self.x_j and self.y_i < self.y_j:
                    candidate_bubble_targets = [[self.x_i+1,self.y_i],[self.x_i,self.y_i+1],[self.x_j-1,self.y_j],[self.x_j,self.y_j-1]]
                    corresp_data_targets = [[self.x_i,self.y_i],[self.x_i,self.y_i],[self.x_j,self.y_j],[self.x_j,self.y_j]]
                
                # find which of those candidates is currently closest to the bubble
                distances_to_bubble = []
                for candidate in candidate_bubble_targets:
                    distances_to_bubble.append(self.travel_distance([self.x_b, self.y_b], candidate))
                
                distance = min(distances_to_bubble)
                bubble_target = candidate_bubble_targets[distances_to_bubble.index(distance)]
                data_target = corresp_data_targets[distances_to_bubble.index(distance)]
                
                if self.print_statements:
                    print('bubble target at ' + str(bubble_target) + ', data target at ' + str(data_target))
                    
                # move bubble to bubble target and swap
                self.move_bubble(bubble_target, data_target)
                
                # repeat
                return self.push_states_together()
    
    class DoubleBubbleSwapCircuit:
        
        # class for running routing algorithm which is equivalent to single bubble swap circuit
        # except for there being two bubbles, and each bubble pushes its corresponding state
        # in alternating order
        
        def __init__(self, provider, x, y, x_i, y_i, x_j, y_j, x_b1, y_b1, x_b2, y_b2, angles = None, seed = 123, noiseless = False, print_statements = False):
            self.provider = provider
            self.x = x
            self.y = y
            self.x_i = x_i
            self.y_i = y_i
            self.x_j = x_j
            self.y_j = y_j
            self.x_b1 = x_b1
            self.y_b1 = y_b1
            self.x_b2 = x_b2
            self.y_b2 = y_b2
            self.angles = angles
            self.seed = seed
            self.noiseless = noiseless
            self.gcirc = self.CreateCirc()
            self.swap_path = [] # list of coordinate pairs (data & bubble qubit)
            self.print_statements = print_statements
            # showing the path the data qubits took
            
        def current_distance(self):
            return self.manhattan_distance(self.x_i, self.y_i, self.x_j, self.y_j)
        def CreateCirc(self):
            if self.noiseless:
                gcirc = GridCircuit(provider = self.provider, x = self.x, y = self.y, max_no_swaps = (self.x+self.y)**2, noise_model = None)
            else:
                gcirc = GridCircuit(provider = self.provider, x = self.x, y = self.y, max_no_swaps = (self.x+self.y)**2)
            if self.angles is None:
                np.random.seed(self.seed)
                self.angles = []
                for i in range(gcirc.no_qubits):
                    self.angles.append(np.random.rand(3)*pi)
            for i in range(self.x):
                for j in range(self.y):
                    qubit = gcirc.convert_coords(i,j)
                    if [i,j] == [self.x_i, self.y_i]:
                        self.angles_i = self.angles[qubit]
                    if [i,j] == [self.x_j, self.y_j]:
                        self.angles_j = self.angles[qubit]
                    if [i,j] != [self.x_b1, self.y_b1] or [i,j] != [self.x_b2, self.y_b2]:
                        gcirc.gate('u', [i,j], parameters = self.angles[qubit])
            return gcirc
        def manhattan_distance(self, x1, y1, x2, y2):
            return abs(x1-x2)+abs(y1-y2)
        def travel_distance(self, p1, p2):
            
            # returns distance bubble would have to travel to get from p1 to p2 avoiding I and J
            for p in [p1,p2]:
                
                # if either point is outside boundary, it is "infinitely far away"
                if p[0] < 0 or p[1] < 0 or p[0] > self.x-1 or p[1] > self.y-1:
                    return 1000
                
                # if either point matches I or J, it can't go around it, so is also "infinitely far away"
                for q in [[self.x_i, self.y_i], [self.x_j, self.y_j]]:
                    if p == q:
                        return 1000
            
            # if either I or J is in a straight line between target and bubble,
            # return manhattan distance + 2, otherwise return manhattan distance
            if p1[0] == p2[0] and ((self.x_i == p1[0] and (self.y_i - p1[1])*(self.y_i - p2[1]) < 0) or (self.x_j == p1[0] and (self.y_j - p1[1])*(self.y_j - p2[1]) < 0)):
                return self.manhattan_distance(*p1, *p2) + 2
            if p1[1] == p2[1] and ((self.y_i == p1[1] and (self.x_i - p1[0])*(self.x_i - p2[0]) < 0) or (self.y_j == p1[1] and (self.x_j - p1[0])*(self.x_j - p2[0]) < 0)):
                return self.manhattan_distance(*p1, *p2) + 2
            return self.manhattan_distance(*p1, *p2)
        
        def move_bubble(self, bubble_target, data_target):
            
            # moves bubble to the bubble_target point where it will be able to switch with data_target
            if data_target == [self.x_i, self.y_i]:
                distance = self.travel_distance([self.x_b1, self.y_b1], bubble_target)
            if data_target == [self.x_j, self.y_j]:
                distance = self.travel_distance([self.x_b2, self.y_b2], bubble_target)
            if self.print_statements:
                print('distance = ' + str(distance))
            # if we're there already, bubble swap with data qubit and update coordinate pointers
            if distance == 0:
                self.gcirc.bswap(data_target, bubble_target)
                self.swap_path.append([data_target, bubble_target])
                if data_target == [self.x_i, self.y_i]:
                    self.x_i, self.y_i = bubble_target
                    self.x_b1, self.y_b1 = data_target
                    if self.print_statements:
                        print('swapped I at ' + str(data_target) + ' with bubble 1 at ' + str(bubble_target))
                    return
                if data_target == [self.x_j, self.y_j]:
                    self.x_j, self.y_j = bubble_target
                    self.x_b2, self.y_b2 = data_target
                    if self.print_statements:
                        print('swapped J at ' + str(data_target) + ' with bubble 2 at ' + str(bubble_target))
                    return
            # if we're not there, find where to go to decrease the distance to the target
            else:
                # list where bubble could go
                if data_target == [self.x_i, self.y_i]:
                    bubble_next_step_candidates = [[self.x_b1+1, self.y_b1],[self.x_b1-1, self.y_b1],[self.x_b1, self.y_b1+1],[self.x_b1, self.y_b1-1]]
                if data_target == [self.x_j, self.y_j]:
                    bubble_next_step_candidates = [[self.x_b2+1, self.y_b2],[self.x_b2-1, self.y_b2],[self.x_b2, self.y_b2+1],[self.x_b2, self.y_b2-1]]
                    
                # find the distance to target in each case
                distances_to_target = []
                for candidate in bubble_next_step_candidates:
                    distances_to_target.append(self.travel_distance(candidate, bubble_target))
                
                # take the coordinates corresponding to the smallest distance to target
                bubble_next_step = bubble_next_step_candidates[distances_to_target.index(min(distances_to_target))]
                
                # swap in that direction
                if data_target == [self.x_i, self.y_i]:
                    self.gcirc.bswap(bubble_next_step, [self.x_b1, self.y_b1])
                    if self.print_statements:
                        print('moved bubble 1 from ' + str([self.x_b1, self.y_b1]) + ' to ' + str(bubble_next_step))
                    self.x_b1, self.y_b1 = bubble_next_step
                if data_target == [self.x_j, self.y_j]:
                    self.gcirc.bswap(bubble_next_step, [self.x_b2, self.y_b2])
                    if self.print_statements:
                        print('moved bubble from ' + str([self.x_b2, self.y_b2]) + ' to ' + str(bubble_next_step))
                    self.x_b2, self.y_b2 = bubble_next_step
                return self.move_bubble(bubble_target, data_target)
                    
        def push_i(self):
            
            if self.current_distance() == 1:
                return
            else:
                
                print('bubble 1 at ' + str([self.x_b1, self.y_b1]))
                # possible directions of change for [x_i, y_i]
                step_sizes = [[1,0],[-1,0],[0,1],[0,-1]]
                
                # coordinates of changed I
                neighbour_coords = [[self.x_i + step[0], self.y_i + step[1]] for step in step_sizes]
                
                # distances from changed I to J
                distances_ij = [self.manhattan_distance(*coords, self.x_j, self.y_j) for coords in neighbour_coords]
                
                # distances from bubble 1 to its potential positions right before swap
                distances_tob = [self.travel_distance([self.x_b1, self.y_b1], coords) for coords in neighbour_coords]
                
                # indices of changed I for which I gets closer to J
                indices_min_ij_distance = [i for i,x in enumerate(distances_ij) if x == min(distances_ij)]
                
                # distances from bubble to those ('valid') changed I coordinates
                distances_tob_for_valid_candidates = [distances_tob[i] for i in indices_min_ij_distance]
                
                # index of neighbour_coords corresponding to the changed I coords closest to the bubble
                bubble_target_index = indices_min_ij_distance[distances_tob_for_valid_candidates.index(min(distances_tob_for_valid_candidates))]
                
                # fetching bubble target
                bubble_target = neighbour_coords[bubble_target_index]
                
                data_target = [self.x_i, self.y_i]
                
                if self.print_statements:
                    print('bubble target at ' + str(bubble_target) + ', data target at ' + str(data_target))
                self.move_bubble(bubble_target, data_target)
                
                # alternate between using bubble1 and bubble2
                return self.push_j()
            
        def push_j(self):
            
            if self.current_distance() == 1:
                return
            else:
                
                print('bubble 2 at ' + str([self.x_b2, self.y_b2]))
                # possible directions of change for [x_i, y_i]
                step_sizes = [[1,0],[-1,0],[0,1],[0,-1]]
                
                # coordinates of changed I
                neighbour_coords = [[self.x_j + step[0], self.y_j + step[1]] for step in step_sizes]
                
                # distances from changed I to J
                distances_ij = [self.manhattan_distance(*coords, self.x_i, self.y_i) for coords in neighbour_coords]
                
                # distances from bubble 1 to its potential positions right before swap
                distances_tob = [self.travel_distance([self.x_b2, self.y_b2], coords) for coords in neighbour_coords]
                
                # indices of changed I for which I gets closer to J
                indices_min_ij_distance = [i for i,x in enumerate(distances_ij) if x == min(distances_ij)]
                
                # distances from bubble to those ('valid') changed I coordinates
                distances_tob_for_valid_candidates = [distances_tob[i] for i in indices_min_ij_distance]
                
                # index of neighbour_coords corresponding to the changed I coords closest to the bubble
                bubble_target_index = indices_min_ij_distance[distances_tob_for_valid_candidates.index(min(distances_tob_for_valid_candidates))]
                
                # fetching bubble target
                bubble_target = neighbour_coords[bubble_target_index]
                
                data_target = [self.x_j, self.y_j]
                
                if self.print_statements:
                    print('bubble target at ' + str(bubble_target) + ', data target at ' + str(data_target))
                self.move_bubble(bubble_target, data_target)
                
                # alternate between using bubble1 and bubble2
                return self.push_i()
            
    class RegularSwapCircuit:
        
        # takes swap path from single (double) bubble swap and swaps I and J
        # along the same path, but with regular (3 CNOT no postselection) swap
        def __init__(self, provider, x, y, x_i, y_i, x_j, y_j, x_b, y_b, swap_path, angles = None, seed = 123, noiseless = False, print_statements = False):
            self.provider = provider
            self.x = x
            self.y = y
            self.x_i = x_i
            self.y_i = y_i
            self.x_j = x_j
            self.y_j = y_j
            self.x_b = x_b
            self.y_b = y_b
            self.angles = angles
            self.seed = seed
            self.noiseless = noiseless
            self.gcirc = self.CreateCirc()
            self.swap_path = swap_path
            self.print_statements = print_statements
        def CreateCirc(self):
            if self.noiseless:
                gcirc = GridCircuit(provider = self.provider, x = self.x,y = self.y, max_no_swaps = (self.x+self.y)**2, noise_model = None)
            else:
                gcirc = GridCircuit(provider = self.provider, x = self.x,y = self.y, max_no_swaps = (self.x+self.y)**2)
            if self.angles is None:
                np.random.seed(self.seed)
                self.angles = []
                for i in range(gcirc.no_qubits):
                   self. angles.append(np.random.rand(3)*pi)
            for i in range(self.x):
                for j in range(self.y):
                    qubit = gcirc.convert_coords(i,j)
                    if [i,j] == [self.x_i, self.y_i]:
                        self.angles_i = self.angles[qubit]
                    if [i,j] == [self.x_j, self.y_j]:
                        self.angles_j = self.angles[qubit]
                    if [i,j] != [self.x_b, self.y_b]:
                        gcirc.gate('u', [i,j], parameters = self.angles[qubit])
            return gcirc
        def push_states_together(self):
            
            # just take swap path and swap along it
            
            if self.print_statements:
                print('I in ' + str([self.x_i, self.y_i]))
                print('J in ' + str([self.x_j, self.y_j]))
            for swap in self.swap_path:
                self.gcirc.gate('swap', swap[0], swap[1])
                if swap[0] == [self.x_i, self.y_i]:
                    self.x_i, self.y_i = swap[1]
                    if self.print_statements:
                        print('swapped I at ' + str(swap[0]) + ' into ' + str(swap[1]))
                        print('I in ' + str([self.x_i, self.y_i]))
                        print('J in ' + str([self.x_j, self.y_j]))
                if swap[0] == [self.x_j, self.y_j]:
                    self.x_j, self.y_j = swap[1]
                    if self.print_statements:
                        print('swapped J at ' + str(swap[0]) + ' into ' + str(swap[1]))
                        print('I in ' + str([self.x_i, self.y_i]))
                        print('J in ' + str([self.x_j, self.y_j]))
            return
                    
                
            
        