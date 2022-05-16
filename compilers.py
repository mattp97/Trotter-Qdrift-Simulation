from ast import And
from asyncore import loop
from operator import matmul
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import optimize
from scipy import interpolate
import math
from numpy import random
import cmath
import time
from sympy import S, symbols, printing
from skopt import gp_minimize

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
        self.order = order
        self.gate_count = 0

        # Use the first computational basis state as the initial state until the user specifies.
        if len(hamiltonian_list) == 0:
            self.initial_state = np.zeros((1,1))
        else:
            self.initial_state = np.zeros((hamiltonian_list[0].shape[0]))
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

    # Assumes terms are already normalized
    def set_hamiltonian(self, mat_list = [], norm_list = []):
        if len(mat_list) != len(mat_list):
            print("[Trott - set_hamiltonian] Incorrect length arrays")
            return 1
        self.hamiltonian_list = mat_list
        self.spectral_norms = norm_list

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        self.initial_state = psi_init

                             
    def simulate(self, time, iterations):
        self.gate_count = 0
        if len(self.hamiltonian_list) == 0:
            return np.copy(self.initial_state)

        op_time = time/iterations
        steps = computeTrotterTimesteps(len(self.hamiltonian_list), op_time, self.order)
        psi = self.initial_state
        for (ix, timestep) in steps:
            psi = linalg.expm(1j * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep) @ psi
            self.gate_count += 1
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
        self.rng_seed = rng_seed
        self.gate_count = 0

        # Use the first computational basis state as the initial state until the user specifies.
        if len(hamiltonian_list) == 0:
            self.initial_state = np.zeros((1,1))
        else:
            self.initial_state = np.zeros((hamiltonian_list[0].shape[0]))
        self.initial_state[0] = 1
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
        self.initial_state = psi_init

    # Assumes terms are already normalized
    def set_hamiltonian(self, mat_list = [], norm_list = []):
        if len(mat_list) != len(mat_list):
            print("[QD - set_hamiltonian] Incorrect length arrays")
            return 1
        self.hamiltonian_list = mat_list
        self.spectral_norms = norm_list

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
        self.gate_count = 0
        evol_op = np.identity(len(self.initial_state))

        if len(self.hamiltonian_list) == 0:
            return np.copy(self.initial_state)

        tau = time * np.sum(self.spectral_norms) / (samples * 1.0)
        final = np.copy(self.initial_state)
        for n in range(samples):
            ix = self.draw_hamiltonian_sample()
            exp_h = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
            final = exp_h @ final
        self.final_state = final
        self.gate_count = samples
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
    def __init__(self, hamiltonian_list = [], inner_order = 1, outter_order = 1, initial_time = 0.1, partition = "random", rng_seed = 1, nb_optimizer = False, weight_threshold = 0.5, nb = 1, epsilon = 0.001):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.a_norms = [] #contains the partitioned norms, as well as the index of the matrix they come from
        self.b_norms = []
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.rng_seed = rng_seed
        self.outter_order = outter_order 
        self.inner_order = inner_order
        self.partition = partition
        self.nb_optimizer = nb_optimizer
        self.epsilon = epsilon #simulation error
        self.weight_threshold = weight_threshold

        self.qdrift_sim = QDriftSimulator()
        self.trotter_sim = TrotterSim(order = inner_order)

        self.nb = nb #number of Qdrift channel samples. Useful to define as an attribute if we are choosing whether or not to optimize over it.
        self.time = initial_time 
        self.gate_count = 0 #Used to keep track of the operators in analyse_sim

        # Use the first computational basis state as the initial state until the user specifies.
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1.
        self.final_state = np.copy(self.initial_state)

        self.prep_hamiltonian_lists(hamiltonian_list) #do we want this done before or after the partitioning?
        np.random.seed(self.rng_seed)
        self.partitioning() #note error was raised because partition() is a built in python method
        self.reset_nested_sims()

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

    # Prep simulators with terms, isolated as function for reusability in partitioning
    def reset_nested_sims(self):
        qdrift_terms, qdrift_norms = [] , []
        trott_terms, trott_norms = [] , []
        for ix in range(len(self.a_norms)):
            index = int(self.a_norms[ix][0].real)
            norm = self.a_norms[ix][1]
            trott_terms.append(self.hamiltonian_list[index])
            trott_norms.append(norm)
        
        for ix in range(len(self.b_norms)):
            index = int(self.b_norms[ix][0].real)
            norm = self.b_norms[ix][1]
            qdrift_terms.append(self.hamiltonian_list[index])
            qdrift_norms.append(norm)
        self.qdrift_sim.set_hamiltonian(qdrift_terms, qdrift_norms)
        self.trotter_sim.set_hamiltonian(trott_terms, trott_norms)

    #First order cost functions to optimize over
    def nb_first_order_cost(self, weight): #first order cost, currently computes equation 31 from paper. Weight is a list of all weights with Nb in the last entry
        cost = 0.0                         #Error with this function, it may not be possible to optimize Nb with this structure given the expression of the function
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

    #Function that allows for the optimization of the nb parameter in the probabilistic partitioning scheme (at each timestep)
    def prob_nb_optima(self, test_nb):
        k = self.inner_order/2
        upsilon = 2*(5**(k -1))
        lamb = sum(self.spectral_norms)
        test_chi = (lamb/len(self.spectral_norms)) * ((test_nb * (self.epsilon/(lamb * self.time))**(1-(1/(2*k))) * 
        ((2*k + upsilon)/(2*k +1))**(1/(2*k)) * (upsilon**(1/(2*k)) / 2**(1-(1/k))))**(1/2) - 1) 
            
        test_probs = []
        for i in range(len(self.spectral_norms)):
            test_probs.append(float(np.abs((1/self.spectral_norms[i])*test_chi))) #ISSUE
        return max(test_probs)


    #partitioning method to execute the partitioning method of the users choice, random likely not used in practice
    def partitioning(self):
        if ((self.partition == "trotter" or self.partition == "qdrift" or self.partition == "random" or self.partition == "optimize")): #This condition may change for the optimize scheme later
            print("This partitioning method does not require repartitioning") #Checks that our scheme is sane, prevents unnecessary time wasting
            return 1

        if self.partition == "prob":
            if self.trotter_sim.order > 1: k = self.trotter_sim.order/2
            else: return "partition not defined for this order"
            
            upsilon = 2*(5**(k -1))
            lamb = sum(self.spectral_norms)

            if self.nb_optimizer == True:
                optimal_nb = optimize.minimize(self.prob_nb_optima, self.nb, method='Nelder-Mead', bounds = optimize.Bounds([0], [np.inf], keep_feasible = False)) #Nb attribute serves as an inital geuss in this partition
                nb_high = int(optimal_nb.x +1)
                nb_low = int(optimal_nb.x)
                prob_high = self.prob_nb_optima(nb_high) #check higher, (nb must be int)
                prob_low = self.prob_nb_optima(nb_low) #check lower 
                if prob_high > prob_low:
                    self.nb = nb_low
                else:
                    self.nb = nb_high
            else:
                self.nb = int(((lamb * self.time/(self.epsilon))**(1-(1/(2*k))) * ((2*k +1)/(2*k + upsilon))**(1/(2*k)) * (2**(1-(1/k))/ upsilon**(1/(2*k)))) +1)
            
            print(self.nb)
            
            chi = (lamb/len(self.spectral_norms)) * ((self.nb * (self.epsilon/(lamb * self.time))**(1-(1/(2*k))) * 
            ((2*k + upsilon)/(2*k +1))**(1/(2*k)) * (upsilon**(1/(2*k)) / 2**(1-(1/k))))**(1/2) - 1) 
            
            for i in range(len(self.spectral_norms)):
                num = np.random.random()
                prob=(1- min((1/self.spectral_norms[i])*chi, 1))
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
                optimized_weights = optimize.minimize(self.nb_first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(lower_bound, upper_bound))
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
                optimized_weights = optimize.minimize(self.first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(lower_bound, upper_bound)) 
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

        #Nelder-Mead optimization protocol based on analytic cost function for first order
        elif self.partition == "optimize chop": 
            guess = sum(self.spectral_norms)/len(self.spectral_norms) #guess the average for the weights
            upper_bound = [1 for x in range(len(self.spectral_norms))]
            upper_bound.append(20) #no upper bound for Nb but set to some number we can compute instead of np.inf
            lower_bound = [0 for x in range(len(self.spectral_norms) + 1)]  #lower bound for Nb is 0
            optimized_threshold = optimize.minimize(self.nb_first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(upper_bound, lower_bound))
            print(optimized_threshold.x)
            for i in range(len(self.spectral_norms)):
                if self.spectral_norms[i] >= optimized_threshold:
                    self.a_norms.append(([i, self.spectral_norms[i]]))
                else:
                    self.b_norms.append(([i, self.spectral_norms[i]]))
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        elif self.partition == "trotter":
            for i in range(len(self.spectral_norms)):
                self.a_norms.append([i, self.spectral_norms[i]])
            self.b_norms = []
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        elif self.partition == "qdrift":
            for i in range(len(self.spectral_norms)):
                self.b_norms.append([i, self.spectral_norms[i]])
            self.a_norms = []
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        else:
            print("Invalid input for attribute 'partition' ")
            return 1

    def repartition(self):
        self.a_norms = []
        self.b_norms = []
        self.partitioning()
        self.reset_nested_sims()

    #Simulate and error scaling 
    def simulate(self, time, samples, iterations): 
        if self.nb_optimizer == False: 
            self.nb = samples  #specifying the number of samples having optimized Nb does nothing
        
        self.gate_count = 0
        channel_visits = computeTrotterTimesteps(2, time / (1. * iterations), self.outter_order)
        current_state = np.copy(self.initial_state)
        for i in range(iterations):
            for (ix, sim_time) in channel_visits:
                if ix == 0:
                    self.trotter_sim.set_initial_state(current_state)
                    current_state = self.trotter_sim.simulate(sim_time, 1)
                if ix == 1:
                    self.qdrift_sim.set_initial_state(current_state)
                    current_state = self.qdrift_sim.simulate(sim_time, samples)
                self.gate_count += self.trotter_sim.gate_count
                self.gate_count += self.qdrift_sim.gate_count
        
        return current_state
            
    #Monte-Carlo sample the infidelity, should work for composite channel
    def sample_channel_inf(self, time, samples, iterations, mcsamples): 
        sample_fidelity = []
        H = []
        for i in range(len(self.spectral_norms)):
            H.append(self.hamiltonian_list[i] * self.spectral_norms[i])
        for s in range(mcsamples):
            sim_state = self.simulate(time, samples, iterations)
            good_state = np.dot(linalg.expm(1j * sum(H) * time), self.initial_state)
            sample_fidelity.append((np.abs(np.dot(good_state.conj().T, sim_state)))**2)
        infidelity = 1- sum(sample_fidelity) / mcsamples 
        return infidelity #this is of type array for some reason?

    #To analyse the number of gates required to meet a certain error threshold, function is guarenteed to work with iterations =1, but can be sped up by a better geuss.
    def sim_channel_performance(self, time, samples, iterations, mcsamples):
        good_inf = []
        bad_inf = []
        lower_bound = iterations
        #Exponential search to set upper bound, then binary search in that interval
        infidelity = self.sample_channel_inf(time, samples, iterations, mcsamples)

        if infidelity < self.epsilon:
            print("[sim_channel_performance] Iterations too large, already below error threshold")
            return 1
        # Iterate up until some max cutoff
        upper_bound = iterations
        for n in range(20):
            infidelity = self.sample_channel_inf(time, samples, iterations, mcsamples)
            upper_bound *=2 
            if infidelity < self.epsilon:
                break

        #Binary search
        while lower_bound < upper_bound:
            mid = 1+ (upper_bound - lower_bound)//2
            infidelity = self.sample_channel_inf(time, samples, mid, mcsamples)
            if (self.sample_channel_inf(time, samples, mid + 5, mcsamples) < self.epsilon) and (self.sample_channel_inf(time, samples, mid - 5, mcsamples) > self.epsilon):
                break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
            elif infidelity < self.epsilon:
                upper_bound = mid - 1
            else:
                lower_bound = mid + 1

        for i in range (mid+1, mid +5):
            infidelity = self.sample_channel_inf(time, samples, i, mcsamples)
            good_inf.append([self.gate_count, float(infidelity)])
        for j in range (mid-5, mid +1):
            infidelity = self.sample_channel_inf(time, samples, j, mcsamples)
            bad_inf.append([self.gate_count, float(infidelity)])

        good_inf = np.array(good_inf)
        bad_inf = np.array(bad_inf)
        gate_data = np.concatenate((bad_inf, good_inf), 0)
        print(gate_data)
        
        #poi = np.interp([self.epsilon], list(gate_data[:,1]), list(gate_data[:,0])) #interpolates where the error threshold is saturated (point of intersection)
        #return float(poi)
        #INTERPOLATOR IS JUST RETURNING THE ENDPOINT!!!
        fit = np.poly1d(np.polyfit(gate_data[:,1], gate_data[:,0], 1)) #linear best fit 
        return fit(self.epsilon)