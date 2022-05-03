from operator import matmul
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import optimize
import math
from numpy import random
import cmath
import time
from sympy import S, symbols, printing

FLOATING_POINT_PRECISION = 1e-10

# Helper function to compute the timesteps to matrix exponentials in a higher order
# product formula. This can be done with only the length of an array, the time, and
# the order of the simulator needed. Returns a list of tuples of the form (index, time),
# where index refers to which hamiltonian term at that step and time refers to the scaled time.
# Assumes your list is of the form [H_1, H_2, ... H_numTerms] 
# For example if you want to do e^{i H_3 t} e^{i H_2 t} e^{i H_1 t} | psi >, then calling this with
# computeTrotterTimesteps(3, t, 1) will return [(0, t), (1, t), (2, t)] where we assume your
# Hamiltonian terms are stored like [H_1, H_2, H_3] and we return the index
# Note the reverse ordering due to matrix multiplication :'( 
def computeTrotterTimesteps(numTerms, simTime, trotterOrder = 1):
    if type(trotterOrder) != type(1):
        print('[computeTrotterTimesteps] trotterOrder input is not an int')
        return 1

    if trotterOrder == 1:
        return [(ix, simTime) for ix in range(numTerms)]

    elif trotterOrder == 2:
        ret = []
        firstOrder = computeTrotterTimesteps(numTerms, simTime / 2.0, 1)
        ret += firstOrder.copy()
        firstOrder.reverse()
        ret += firstOrder
        return ret

    elif trotterOrder % 2 == 0:
        timeConst = 1.0/(4 - 4**(1.0 / trotterOrder - 1))
        outter = computeTrotterTimesteps(numTerms, timeConst * simTime, trotterOrder - 2)
        inner = computeTrotterTimesteps(numTerms, (1. - 4. * timeConst) * simTime, trotterOrder - 2)
        ret = [] + 2 * outter + inner + 2 * outter
        return ret

    else:
        print("[computeTrotterTimesteps] trotterOrder seems to be bad")
        return 1

# A basic trotter simulator organizer.
# Inputs
# - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
#                     in the list is numpy matrix (preferably sparse, no guarantee on data struct
#                     actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
#                     ASSUME SQUARE MATRIX INPUTS
# - time: Floating point (no size guarantees) representing time for TOTAL simulation, NOT per
#         iteration. "t" parameter in the literature.
# - iterations: "r" parameter. This object will handle repeating the channel r times and dividing 
#               overall simulation into t/r chunks.
# - order: The trotter order, represented as "2k" in literature. 
class TrotterSim:
    def __init__(self, hamiltonian_list = [], order = 1):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.order = order

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list)

    # Helper function to compute spectral norm list. Used solely constructor
    def prep_hamiltonian_lists(self, ham_list):
        for h in ham_list:
            temp_norm = np.linalg.norm(h, ord=2)
            if temp_norm < FLOATING_POINT_PRECISION:
                print("[prep_hamiltonian_lists] Spectral norm of a hamiltonian found to be 0")
                self.spectral_norms = []
                self.hamiltonian_list = []
                return 1
            self.spectral_norms.append(temp_norm)
            self.hamiltonian_list.append(h / temp_norm)
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        global FLOATING_POINT_PRECISION
        if type(psi_init) != type(self.initial_state):
            print("[set_initial_state]: input type not numpy ndarray")
            return 1

        if psi_init.size != self.initial_state.size:
            print("[set_initial_state]: input size not matching")
            return 1

        # check that the frobenius aka l2 norm is 1
        if np.linalg.norm(psi_init, ord = 2) - 1.0 > FLOATING_POINT_PRECISION:
            print("[set_initial_state]: input is not properly normalized")
            return 1

        # check that each dimension has magnitude between 0 and 1
        for ix in range(len(psi_init)):
            if np.abs(psi_init[ix]) > 1.0:
                print("[set_initial_state]: too big of a dimension in vector")
                return 1

        # Should be good to go now
        self.initial_state = psi_init
        return 0

    # Helper functions to generate the sequence of gates for product formulas given an input time
    # up to the simulator function to handle iterations and such. Can probably move all of these 
    # into one single function.
    def first_order_op(self, op_time):
        evol_op = np.identity(self.hilbert_dim)
        for ix in range(len(self.hamiltonian_list)):
            h_term = self.hamiltonian_list[ix] * self.spectral_norms[ix]
            exp_h = linalg.expm(1.0j * op_time  * h_term)
            evol_op = np.matmul(evol_op, exp_h)
        return evol_op
    
    def first_order_op_reverse(self, op_time):
        evol_op = np.identity(self.hilbert_dim)
        for ix in range(1, len(self.hamiltonian_list)+1):
            h_term = self.hamiltonian_list[-ix] * self.spectral_norms[-ix]
            exp_h = linalg.expm(1.0j * op_time  * h_term)
            evol_op = np.matmul(evol_op, exp_h)
        return evol_op
    
    def second_order_op(self, op_time):
        forward = self.first_order_op(op_time / 2.0)
        backward = self.first_order_op_reverse(op_time / 2.0)
        return np.matmul(backward, forward)

    def higher_order_op(self, order, op_time):
        if type(order) != type(2):
            print("[higher_order_op] provided input order (" + str(order) + ") is not an integer")
            return 1
        elif order == 1:
            return self.first_order_op(op_time)
        elif order == 2:
            return self.second_order_op(op_time)
        elif order % 2 == 0:
            time_const = 1.0/(4 - 4**(1.0/order - 1))
            outer = np.linalg.matrix_power(self.higher_order_op(order - 2, time_const * op_time), 2)
            inner = self.higher_order_op(order - 2, (1. - 4. * time_const) * op_time)
            ret = np.matmul(outer, inner)
            ret = np.matmul(ret, outer)
            return ret
        else:
            print("[higher_order_op] Encountered incorrect order (" + str(order) + ") for trotter formula")
            return 1
    
    def simulate(self, time, iterations):
        if type(iterations) != type(3) or iterations < 1:
            print("[simulate] Incorrect type for iterations, must be integer greater than 1.")
            return 1
        evol_op = self.higher_order_op(self.order, (1.0 * time) / (1.0 * iterations))
        evol_op = np.linalg.matrix_power(evol_op, iterations)
        self.final_state = np.dot(evol_op, self.initial_state)
        return np.copy(self.final_state)

    def infidelity(self, time, iterations):
        H = []
        for i in range(len(self.spectral_norms)):
            H.append(self.hamiltonian_list[i] * self.spectral_norms[i])
        sim_state = self.simulate(time, iterations)
        good_state = np.dot(linalg.expm(1j * sum(H) * time), self.initial_state)
        #infidelity = 1 - (np.abs(np.dot(good_state.conj().T, sim_state)))**2
        infidelity = (np.linalg.norm(sim_state - good_state))
        return infidelity
    

#Trotter Simulator... same as above but with matrix vector multiplicaiton. Above alg seems to be doing something weird.
#Needs cleaning up and a recursive method for higher order product formulas 
class TrotterSim2:
    def __init__(self, hamiltonian_list = [], order = 1):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.order = order

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list)
    
    # Helper function to compute spectral norm list. Used solely constructor
    def prep_hamiltonian_lists(self, ham_list):
        for h in ham_list:
            temp_norm = np.linalg.norm(h, ord=2)
            if temp_norm < FLOATING_POINT_PRECISION:
                print("[prep_hamiltonian_lists] Spectral norm of a hamiltonian found to be 0")
                self.spectral_norms = []
                self.hamiltonian_list = []
                return 1
            self.spectral_norms.append(temp_norm)
            self.hamiltonian_list.append(h / temp_norm)
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        global FLOATING_POINT_PRECISION
        if type(psi_init) != type(self.initial_state):
            print("[set_initial_state]: input type not numpy ndarray")
            return 1

        if psi_init.size != self.initial_state.size:
            print("[set_initial_state]: input size not matching")
            return 1

        # check that the frobenius aka l2 norm is 1
        if np.linalg.norm(psi_init, ord = 2) - 1.0 > FLOATING_POINT_PRECISION:
            print("[set_initial_state]: input is not properly normalized")
            return 1

        # check that each dimension has magnitude between 0 and 1
        for ix in range(len(psi_init)):
            if np.abs(psi_init[ix]) > 1.0:
                print("[set_initial_state]: too big of a dimension in vector")
                return 1

        # Should be good to go now
        self.initial_state = psi_init
        return 0

                             
    def simulate(self, time, iterations):
        op_time = time/iterations
        steps = computeTrotterTimesteps(len(self.hamiltonian_list), op_time, self.order)
        psi = self.initial_state
        for (ix, timestep) in steps:
            psi = linalg.expm(1j * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep) @ psi
        return psi

    def infidelity(self, time, iterations):
        H = []
        for i in range(len(self.spectral_norms)):
            H.append(self.hamiltonian_list[i] * self.spectral_norms[i]) #This might be the error
        sim_state = self.simulate(time, iterations)
        good_state = np.dot(linalg.expm(1j * sum(H) * time), self.initial_state)
        #infidelity = 1 - (np.abs(np.dot(good_state.conj().T, sim_state)))**2
        infidelity = (np.linalg.norm(sim_state - good_state))                     
        return infidelity                         
    
    
# QDRIFT Simulator
# Inputs
# - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
#                     in the list is numpy matrix (preferably sparse, no guarantee on data struct
#                     actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
#                     ASSUME SQUARE MATRIX INPUTS
# - time: Floating point (no size guarantees) representing time for TOTAL simulation, NOT per
#         iteration. "t" parameter in the literature.
# - samples: "big_N" parameter. This object controls the number of times we sample from the QDrift channel, and each
#            exponetial is applied with time replaced by time*sum(spectral_norms)/big_N.
# - rng_seed: Seed for the random number generator so results are reproducible.
    
class QDriftSimulator:
    def __init__(self, hamiltonian_list = [], rng_seed = 1):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.rng_seed = rng_seed

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1.
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list)
        np.random.seed(self.rng_seed)

    def prep_hamiltonian_lists(self, ham_list):
        for h in ham_list:
            temp_norm = np.linalg.norm(h, ord=2)
            if temp_norm < FLOATING_POINT_PRECISION:
                print("[prep_hamiltonian_lists] Spectral norm of a hamiltonian found to be 0")
                self.spectral_norms = []
                self.hamiltonian_list = []
                return 1
            self.spectral_norms.append(temp_norm)
            self.hamiltonian_list.append(h / temp_norm)
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        global FLOATING_POINT_PRECISION
        if type(psi_init) != type(self.initial_state):
            print("[set_initial_state]: input type not numpy ndarray")
            return 1

        if psi_init.size != self.initial_state.size:
            print("[set_initial_state]: input size not matching")
            return 1

        # check that the frobenius aka l2 norm is 1
        if np.linalg.norm(psi_init, ord='fro') - 1.0 > FLOATING_POINT_PRECISION:
            print("[set_initial_state]: input is not properly normalized")
            return 1

        # check that each dimension has magnitude between 0 and 1
        for ix in range(len(psi_init)):
            if np.abs(psi_init[ix]) > 1.0:
                print("[set_initial_state]: too big of a dimension in vector")
                return 1

        # Should be good to go now
        self.initial_state = psi_init
        return 0

    # RETURNS A 0 BASED INDEX TO BE USED IN CODE!!
    def draw_hamiltonian_sample(self):
        sample = np.random.random()
        tot = 0.
        lamb = np.sum(self.spectral_norms)
        for ix in range(len(self.spectral_norms)):
            if sample > tot and sample < tot + self.spectral_norms[ix] / lamb:
                return ix
            tot += self.spectral_norms[ix] / lamb
        return len(self.spectral_norms) - 1

    def simulate(self, time, samples):
        evol_op = np.identity(self.hilbert_dim)
        tau = time * np.sum(self.spectral_norms) / (samples * 1.0)
        final = np.copy(self.initial_state)
        for n in range(samples):
            ix = self.draw_hamiltonian_sample()
            exp_h = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
            final = exp_h @ final
        self.final_state = final
        return np.copy(self.final_state)

    def sample_channel_inf(self, time, samples, mcsamples):
        sample_fidelity = []
        H = []
        for i in range(len(self.spectral_norms)):
            H.append(self.hamiltonian_list[i] * self.spectral_norms[i])
        for s in range(mcsamples):
            sim_state = self.simulate(time, samples)
            good_state = np.dot(linalg.expm(1j * sum(H) * time), self.initial_state)
            sample_fidelity.append((np.abs(np.dot(good_state.conj().T, sim_state)))**2)
        infidelity = 1- sum(sample_fidelity) / mcsamples 
        return infidelity
    

# Composite Simulator
# Inputs
# - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
#                     in the list is numpy matrix (preferably sparse, no guarantee on data struct
#                     actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
#                     ASSUME SQUARE MATRIX INPUTS
# - time: Floating point (no size guarantees) representing time for TOTAL simulation, NOT per
#         iteration. "t" parameter in the literature.
# - samples: "big_N" parameter. This object controls the number of times we sample from the QDrift channel, and each
#            exponetial is applied with time replaced by time*sum(spectral_norms)/big_N.
# - rng_seed: Seed for the random number generator so results are reproducible.
# - order: The Trotter-Suzuki product formula order for both the inner and outer loop of the algorithm
# - partition: a string indicating how to divide the simulation into a composition of Trotter and QDrift steps. Has 2 options 
#              for either probabilistic partitioning or an optimized cost partition.
# - epsilon: The total error accumulated over the simulation           

###############################################################################################################################################################
#Composite simulation but using a framework with lists of tuples instead of lists of matrices for improved runtime
#This code adopts the convention that for lists of tuples, indices are stored in [0] and values in [1]
class CompositeSim:
    def __init__(self, hamiltonian_list = [], order = 1, partition = "random", rng_seed = 1, nb_optimizer = False, weight_threshold = 0.5, nb = 1, epsilon = 0.001):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.a_norms = [] #contains the partitioned norms, as well as the index of the matrix they come from
        self.b_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.rng_seed = rng_seed
        self.order = order #REDEFINE THIS IN TROTTER!!!!
        self.partition = partition
        self.nb_optimizer = nb_optimizer
        self.epsilon = epsilon #simulation error
        self.weight_threshold = weight_threshold

        self.nb = nb #number of Qdrift channel samples. Useful to define as an attribute if we are choosing whether or not to optimize over it.
        self.time = 1 #DISCUSS THIS

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1.
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list) #do we want this done before or after the partitioning?
        np.random.seed(self.rng_seed)
        self.partitioning() #note error was raised because partition() is a built in python method
        
        print("There are " + str(len(self.a_norms)) + " terms in Trotter") #make the partition known
        print("There are " + str(len(self.b_norms)) + " terms in QDrift")
        if nb_optimizer == True:
            print("Nb is equal to " + str(self.nb))

    def prep_hamiltonian_lists(self, ham_list):
        for h in ham_list:
            temp_norm = np.linalg.norm(h, ord=2)
            if temp_norm < FLOATING_POINT_PRECISION:
                print("[prep_hamiltonian_lists] Spectral norm of a hamiltonian found to be 0")
                self.spectral_norms = []
                self.hamiltonian_list = []
                return 1
            self.spectral_norms.append(temp_norm)
            self.hamiltonian_list.append(h / temp_norm)
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        global FLOATING_POINT_PRECISION
        if type(psi_init) != type(self.initial_state):
            print("[set_initial_state]: input type not numpy ndarray")
            return 1

        if psi_init.size != self.initial_state.size:
            print("[set_initial_state]: input size not matching")
            return 1

        # check that the frobenius aka l2 norm is 1
        if np.linalg.norm(psi_init, ord = 2) - 1.0 > FLOATING_POINT_PRECISION:
            print("[set_initial_state]: input is not properly normalized")
            return 1

        # check that each dimension has magnitude between 0 and 1
        for ix in range(len(psi_init)):
            if np.abs(psi_init[ix]) > 1.0:
                print("[set_initial_state]: too big of a dimension in vector")
                return 1

        # Should be good to go now
        self.initial_state = psi_init
        return 0

    #First order cost functions to optimize over
    def nb_first_order_cost(self, weight): #first order cost, currently computes equation 31 from paper. Weight is a list of all weights with Nb in the last entry
        cost = 0.0
        qd_sum = 0.0
        for i in range(len(self.spectral_norms)):
            qd_sum += (1-weight[i]) * self.spectral_norms[i]
            for j in range(len(self.spectral_norms)):
                commutator_norm = np.linalg.norm(np.matmul(self.hamiltonian_list[i], self.hamiltonian_list[j]) - np.matmul(self.hamiltonian_list[j], self.hamiltonian_list[i]), ord = 2)
                cost += (2/(5**(1/2))) * ((weight[i] * weight[j] * self.spectral_norms[i] * self.spectral_norms[j] * commutator_norm) + 
                    (weight[i] * (1-weight[j]) * self.spectral_norms[i] * self.spectral_norms[j] * commutator_norm))
        cost += (qd_sum**2) * 4/weight[-1] #dividing by Nb at the end (this form is just being used so I can easily optimize Nb as well)
        return cost

    def first_order_cost(self, weight): #first order cost, currently computes equation 31 from paper. Function does not have nb as an omptimizable parameter
        cost = 0.0
        qd_sum = 0.0
        for i in range(len(self.spectral_norms)):
            qd_sum += (1-weight[i]) * self.spectral_norms[i]
            for j in range(len(self.spectral_norms)):
                commutator_norm = np.linalg.norm(np.matmul(self.hamiltonian_list[i], self.hamiltonian_list[j]) - np.matmul(self.hamiltonian_list[j], self.hamiltonian_list[i]), ord = 2)
                cost += (2/(5**(1/2))) * ((weight[i] * weight[j] * self.spectral_norms[i] * self.spectral_norms[j] * commutator_norm) + 
                    (weight[i] * (1-weight[j]) * self.spectral_norms[i] * self.spectral_norms[j] * commutator_norm))
        cost += (qd_sum**2) * 4/self.nb #dividing by Nb at the end (this form is just being used so I can easily optimize Nb as well)
        return cost


    #partitioning method to execute the partitioning method of the users choice, random likely not used in practice
    def partitioning(self):
        if self.partition == "prob":
            gamma = 2*5**(self.order -1)
            lamb = sum(self.spectral_norms)
            chi = (lamb/len(self.spectral_norms)) * ((self.nb * (self.epsilon/(lamb * self.time))**(1-1/(2*self.order)) * 
            ((2*self.order + gamma)/(2*self.order +1))**(1/(2*self.order)) * gamma**(1/(2*self.order)) / 2**(1-1/self.order))**(1/2) -1) #discrepancy with order and k
            
            for i in range(len(self.spectral_norms)):
                num = np.random.random()
                prob=(1- min(self.spectral_norms[i]*chi, 1))
                if prob >= num:
                    self.a_norms.append(([i, self.spectral_norms[i]]))
                else:
                    self.b_norms.append(([i, self.spectral_norms[i]]))
            return 0
        
        #Nelder-Mead optimization protocol based on analytic cost function for first order
        elif self.partition == "optimize": 
            if self.nb_optimizer == True: #if Nb is a parameter we wish to numerically optimize
                guess = [0.5 for x in range(len(self.spectral_norms))] #guess for the weights 
                guess.append(2) #initial guess for Nb
                upper_bound = [1 for x in range(len(self.spectral_norms))]
                upper_bound.append(20) #no upper bound for Nb but set to some number we can compute instead of np.inf
                lower_bound = [0 for x in range(len(self.spectral_norms) + 1)]  #lower bound for Nb is 0
                optimized_weights = optimize.minimize(self.nb_first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(upper_bound, lower_bound))
                print(optimized_weights.x)
                for i in range(len(self.spectral_norms)):
                    if optimized_weights.x[i] >= self.weight_threshold:
                        self.a_norms.append([i, self.spectral_norms[i]])
                    elif optimized_weights.x[i] < self.weight_threshold:
                        self.b_norms.append([i, self.spectral_norms[i]])
                self.a_norms = np.array(self.a_norms, dtype='complex')
                self.b_norms = np.array(self.b_norms, dtype='complex')
                self.nb = int(optimized_weights.x[-1] + 1) #nb must be of type int so take the ceiling 
                return 0

            if self.nb_optimizer == False: #same as above leaving Nb as user-defined
                guess = [0.5 for x in range(len(self.spectral_norms))] #guess for the weights 
                upper_bound = [1 for x in range(len(self.spectral_norms))]
                lower_bound = [0 for x in range(len(self.spectral_norms))]  
                optimized_weights = optimize.minimize(self.first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(upper_bound, lower_bound))
                print(optimized_weights.x)
                for i in range(len(self.spectral_norms)):
                    if optimized_weights.x[i] >= self.weight_threshold:
                        self.a_norms.append([i, self.spectral_norms[i]])
                    elif optimized_weights.x[i] < self.weight_threshold:
                        self.b_norms.append([i, self.spectral_norms[i]])
                self.a_norms = np.array(self.a_norms, dtype='complex')
                self.b_norms = np.array(self.b_norms, dtype='complex')
                return 0
        
        elif self.partition == "random":
            for i in range(len(self.spectral_norms)):
                sample = np.random.random()
                if sample >= 0.5:
                    self.a_norms.append([i, self.spectral_norms[i]])
                elif sample < 0.5:
                    self.b_norms.append([i, self.spectral_norms[i]])
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0
        
        elif self.partition == "chop": #cutting off at some value defined by the user
            for i in range(len(self.spectral_norms)):
                if self.spectral_norms[i] >= self.weight_threshold:
                    self.a_norms.append(([i, self.spectral_norms[i]]))
                else:
                    self.b_norms.append(([i, self.spectral_norms[i]]))
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        elif self.partition == "trotter":
            self.a_norms = self.spectral_norms
            self.b_norms = []
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        elif self.partition == "qdrift":
            self.b_norms = self.spectral_norms
            self.a_norms = []
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        else:
            print("Invalid input for attribute 'partition' ")
            return 1

    #Trotter functions -- modified from the Trotter sim to also take the list of tuples containing the indices of operators and their respective norms to symmetrize as an input
    def first_order(self, op_time, norms_list):
        ops_index = []
        for i in range(len(norms_list)):
            ops_index.append([norms_list[i][0], 1j * norms_list[i][1] * op_time])
        return np.array(ops_index, dtype = 'complex')
                            
    def second_order(self, op_time, norms_list):
        ops_index = []
        for i in range(len(norms_list)):
            ops_index.append([norms_list[i][0], 1j * norms_list[i][1] * op_time/2])
        for i in range(1, len(norms_list)+1):
            ops_index.append([norms_list[-i][0], 1j * norms_list[-i][1] * op_time/2])
        return np.array(ops_index, dtype = 'complex')

    def higher_order(self, op_time, order, norms_list):
        if type(order) != type(2):
            print("[higher_order_op] provided input order (" + str(order) + ") is not an integer")
            return 1
        elif order == 1:
            return self.first_order(op_time, norms_list)
        elif order == 2:
            return self.second_order(op_time, norms_list)
        elif order == 4:
            time_const = 1.0/(4 - 4**(1.0/(order - 1)))
            fourth_order_op = []
            #for i in range(1, order/2):
            outer = []
            inner = []
            outer.append(self.second_order(op_time))
            for i in range(len(self.spectral_norms)):
                inner.append(linalg.expm(1j*self.spectral_norms[i]* op_time/2 * self.hamiltonian_list[i]))
            for i in range(1, len(self.spectral_norms)+1):
                inner.append(linalg.expm(1j*self.spectral_norms[-i]* op_time/2 * self.hamiltonian_list[-i]))
            fourth_order_op.append(outer)
            fourth_order_op.append(outer)
            fourth_order_op.append(inner)             #removed [:]
            fourth_order_op.append(outer)
            fourth_order_op.append(outer)
            return fourth_order_op
                             
        else:
            print("[higher_order_op] Encountered incorrect order (" + str(order) + ") for trotter formula")
            return 1
            
    #QDrift functions, [0] and [1] show up based on where indices[0] and norms[1] are stored in the tuple
    def draw_hamiltonian_sample(self):
        sample = np.random.random()
        tot = 0.
        lamb = sum(self.b_norms)[1]
        for ix in range(len(self.b_norms)):
            if sample > tot and sample < tot + self.b_norms[ix][1] / lamb:
                return ix
            tot += self.b_norms[ix][1] / lamb
        return len(self.b_norms) - 1 #why is this here again?

    def qdrift_list(self, samples, time):
        operator_index = []
        tau = time * (sum(self.b_norms)[1]) / (samples * 1.0)
        for n in range(samples):
            ix = self.draw_hamiltonian_sample()
            operator_index.append([self.b_norms[ix][0], 1j * tau])
        return np.array(operator_index, dtype = 'complex')

    #Simulate and error scaling 
    def simulate(self, time, samples, iterations, do_outer_loop): 
        if self.nb_optimizer == False: 
            self.nb = samples
        else: pass  #specifying the number of samples having optimized Nb does nothing

        if do_outer_loop == True:
            inner_loop = np.concatenate(((self.higher_order(time, self.order, self.a_norms)), (self.qdrift_list(self.nb, time))), 0) #creates inner loop
            outer_loop = (self.higher_order(-1j/iterations, self.order, inner_loop)) #creates the outerloop, -1j so as not to multiply j again
            final = np.copy(self.initial_state)

            for i in range(len(outer_loop)*iterations):
                final = linalg.expm(outer_loop[i%len(outer_loop)][1] * self.hamiltonian_list[int((outer_loop[i%len(outer_loop)][0]).real)]) @ final
        
            self.final_state = final
            return np.copy(self.final_state)

        elif do_outer_loop ==  False:
            inner_loop = np.concatenate(((self.higher_order(time/iterations, self.order, self.a_norms)), (self.qdrift_list(self.nb, time/iterations))), 0) #creates inner loop (include number of iterations here)
            final = np.copy(self.initial_state)

            for i in range(len(inner_loop)*iterations):
                final = linalg.expm(inner_loop[i%len(inner_loop)][1] * self.hamiltonian_list[int((inner_loop[i%len(inner_loop)][0]).real)]) @ final

            self.final_state = final
            return np.copy(self.final_state)
            
    #Monte-Carlo sample the infidelity, should work for composite channel    
    def sample_channel_inf(self, time, samples, iterations, mcsamples, do_outer_loop): 
        sample_fidelity = []
        H = []
        for i in range(len(self.spectral_norms)):
            H.append(self.hamiltonian_list[i] * self.spectral_norms[i])
        for s in range(mcsamples):
            sim_state = self.simulate(time, samples, iterations, do_outer_loop)
            good_state = np.dot(linalg.expm(1j * sum(H) * time), self.initial_state)
            sample_fidelity.append((np.abs(np.dot(good_state.conj().T, sim_state)))**2)
        infidelity = 1- sum(sample_fidelity) / mcsamples 
        return infidelity





# Create a simple evolution operator, compare the difference with known result. Beware floating pt
# errors
# H = sigma_X
def test_first_order_op():
    sigma_x = np.array([[0,1],[1,0]])
    sim = TrotterSim([sigma_x], order=1)

def test_second_order_op():
    sigma_x = np.array([[0, 1], [1, 0]])
    sim = TrotterSim([sigma_x], order=2)

def test_higher_order_op():
    sigma_x = np.array([[0,1], [1,0]])
    sim = TrotterSim([sigma_x], order=6)

def test_trotter():
    hilb_dim = 16
    X = np.array([[0, 1],[1, 0]], dtype='complex')
    Y = np.array([[0, -1j], [1j, 0]], dtype='complex')
    Z = np.array([[1, 0], [0, -1]], dtype='complex')
    I = np.array([[1, 0], [0, 1]], dtype='complex')
    
    # h1 = np.random.random() * np.kron(X, X)
    # h2 = np.random.random() * np.kron(X, Y)
    # h3 = np.random.random() * np.kron(X, Z)
    # h4 = np.random.random() * np.kron(Y, Z)
    # h5 = np.random.random() * np.kron(Y, Y)
    # h1 = np.random.random() * np.kron(X, X)

    h1 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h1 += h1.conjugate().T
    h2 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h2 += h1.conjugate().T
    h3 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h3 += h1.conjugate().T
    h4 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h4 += h1.conjugate().T
    h5 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h5 += h1.conjugate().T
    h6 = np.random.randn(hilb_dim, hilb_dim) + 1j * np.random.randn(hilb_dim, hilb_dim)
    h6 += h1.conjugate().T

    h = [h1, h2, h3, h4, h5, h6]
    input_state = np.array([1] + [0] * (hilb_dim - 1), dtype='complex').flatten()

    sim1 = TrotterSim(h, order = 1)
    sim1.set_initial_state(input_state)
    sim2 = TrotterSim(h, order = 2)
    sim2.set_initial_state(input_state)
    sim4 = TrotterSim(h, order = 4)
    sim4.set_initial_state(input_state)

    iterations = 2**16
    t_list = np.logspace(-4, -1, 100)
    inf1 = []
    inf2 = []
    inf4 = []
    for t in t_list:
        inf1.append(sim1.infidelity(t, iterations))
        inf2.append(sim2.infidelity(t, iterations))
        inf4.append(sim4.infidelity(t, iterations))
    log_inf1 = np.log10(inf1).flatten()
    log_inf2 = np.log10(inf2).flatten()
    log_inf4 = np.log10(inf4).flatten()
    log_t = np.log10(t_list)

    # Note we use the same t scale for all orders
    fig, axs = plt.subplots(3, sharex = True)
    fig.suptitle("Log-log t vs infidelity for Trotter formula orders 1, 2, and 4")

    # Order 1
    axs[0].set_title("Order 1")
    axs[0].plot(log_t, log_inf1, 'bo-')
    axs[0].set(ylabel='log(infidelity)')

    fit_points = 50 # declare the starting point to fit in the data
    p = np.polyfit(log_t[0 : fit_points], log_inf1[0 : fit_points], 1)
    f = np.poly1d(p)

    t_new = np.linspace(log_t[fit_points], log_t[-1], 50)
    y_new = f(t_new)

    data = symbols("t")
    poly = sum(S("{:6.2f}".format(v)) * data**i for i, v in enumerate(p[::-1]))
    eq_latex = printing.latex(poly)

    axs[0].plot(t_new, y_new, 'r--', label = "${}$".format(eq_latex))
    axs[0].legend(fontsize = "large")

    # Order 2
    axs[1].set_title("Order 2")
    axs[1].plot(log_t, log_inf2, 'bo-')
    axs[1].set(ylabel='log(infidelity)')

    fit_points = 50 # declare the starting point to fit in the data
    p = np.polyfit(log_t[0 : fit_points], log_inf2[0 : fit_points], 1)
    f = np.poly1d(p)

    t_new = np.linspace(log_t[fit_points], log_t[-1], 50)
    y_new = f(t_new)

    data = symbols("t")
    poly = sum(S("{:6.2f}".format(v)) * data**i for i, v in enumerate(p[::-1]))
    eq_latex = printing.latex(poly)

    axs[1].plot(t_new, y_new, 'r--', label = "${}$".format(eq_latex))
    axs[1].legend(fontsize = "large")

    # Order 4
    axs[2].set_title("Order 4")
    axs[2].plot(log_t, log_inf4, 'bo-')
    axs[2].set(ylabel='log(infidelity)')

    fit_points = 50 # declare the starting point to fit in the data
    p = np.polyfit(log_t[0 : fit_points], log_inf4[0 : fit_points], 1)
    f = np.poly1d(p)

    t_new = np.linspace(log_t[fit_points], log_t[-1], 50)
    y_new = f(t_new)

    data = symbols("t")
    poly = sum(S("{:6.2f}".format(v)) * data**i for i, v in enumerate(p[::-1]))
    eq_latex = printing.latex(poly)

    axs[2].plot(t_new, y_new, 'r--', label = "${}$".format(eq_latex))
    axs[2].legend(fontsize = "large")
    plt.show()


def test_qdrift():
    time = 0.3
    bigN = 500
    X = np.array([[0, 1],[1, 0]], dtype='complex')
    Y = np.array([[0, -1j], [1j, 0]], dtype='complex')
    Z = np.array([[1, 0], [0, -1]], dtype='complex')
    I = np.array([[1, 0], [0, 1]], dtype='complex')
    
    h1 = np.kron(X, X)
    h2 = np.kron(X, Y)
    h3 = np.kron(X, Z)
    h4 = np.kron(Y, Z)
    h5 = np.kron(Y, Y)
    h = [h1, h2, h3, h4, h5]

    input_state = np.array([1, 0, 0, 0]).reshape((4,1))
    qdsim = QDriftSimulator(h)
    qdsim.set_initial_state(input_state)

    exact_op = linalg.expm(1j * sum(h) * time)
    expected = np.dot(exact_op, input_state)
    
    fidelities = []
    num_samps = 50
    for ix in range(50):
        qd_out = qdsim.simulate(time, bigN)
        tmp = np.abs(np.dot(expected.conj().T, qd_out))**2
        fidelities.append(tmp)
    print("[test_qd] empirical infidelity: ", 1 - sum(fidelities) / (1. * num_samps))

test = False
if test:
    test_first_order_op()
    test_second_order_op()
    test_higher_order_op()
    test_trotter()
    # test_qdrift()
