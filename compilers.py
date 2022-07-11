from ast import And
from asyncore import loop
from mimetypes import init
from operator import matmul
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import linalg
from scipy import optimize
from scipy import interpolate
import math
from numpy import random
import cmath
import time as time_this
from sqlalchemy import false
from sympy import S, symbols, printing
from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import cProfile, pstats, io

#A simple profiler. To use this, place @profile above the function of interest
def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue()) 
        return retval
    return inner

#a decorator that can allow one to conditionally test functions (like the one above) or apply other decorators
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator

FLOATING_POINT_PRECISION = 1e-10
MC_SAMPLES_DEFAULT=100

#A function to generate a random initial state that is normalized
def initial_state_randomizer(hilbert_dim):
    initial_state = []
    x = np.random.random(hilbert_dim)
    y = np.random.random(hilbert_dim)
    initial_state = x + (1j * y) 
    initial_state_norm = initial_state / np.linalg.norm(initial_state)
    return initial_state_norm

#A function to calculate the trace distance between two numpy arrays (density operators)
def trace_distance(rho, sigma):
    diff = rho - sigma
    w, v = np.linalg.eigh(diff)
    dist = 1/2 * sum(np.abs(w))
    return dist

def exact_time_evolution_density(hamiltonian_list, time, initial_rho):
    if len(hamiltonian_list) == 0:
        print("[exact_time_evolution] pls give me hamiltonian")
        return 1
    exp_op = linalg.expm(1j * sum(hamiltonian_list) * time)
    return exp_op @ initial_rho @ exp_op.conj().T

# Inputs are self explanatory except simulator which can be any of 
# TrotterSim, QDriftSim, CompositeSim
# Outputs: a single shot estimate of the infidelity according to the exact output provided. 
def single_infidelity_sample(simulator, time, exact_output, iterations = 1, nbsamples = 1):
    simulator.reset_init_state()
    sim_output = []

    if type(simulator) == QDriftSim:
        sim_output = simulator.simulate(time, nbsamples)
    
    if type(simulator) == TrotterSim:
        sim_output = simulator.simulate(time, iterations)
    
    if type(simulator) == CompositeSim:
        sim_output = simulator.simulate(time, nbsamples, iterations)

    infidelity = 1 - (np.abs(np.dot(exact_output.conj().T, sim_output)).flat[0])**2
    return (infidelity, simulator.gate_count)

def mutli_infidelity_sample(simulator, time, exact_output, iterations=1, nbsamples=1, mc_samples=MC_SAMPLES_DEFAULT):
    ret = []

    # No need to sample TrotterSim, just return single element list
    if type(simulator) == TrotterSim:
        ret.append(single_infidelity_sample(simulator, time, exact_output, iterations=iterations, nbsamples=nbsamples))
        ret *= mc_samples
    else:
        for samp in range(mc_samples):
            ret.append(single_infidelity_sample(simulator, time, exact_output, iterations=iterations, nbsamples=nbsamples))

    return ret

def is_threshold_met(infidelities, threshold):
    mean = np.mean(infidelities)
    std_dev = np.std(infidelities)

    if threshold > mean + 2 * std_dev:
        return True
    else:
        return False

def exact_time_evolution(hamiltonian_list, time, initial_state):
    if len(hamiltonian_list) == 0:
        print("[exact_time_evolution] pls give me hamiltonian")
        return 1

    return linalg.expm(1j * sum(hamiltonian_list) * time) @ initial_state


def find_optimal_iterations(simulator, hamiltonian_list, time=1., infidelity_threshold=0.05, mc_samples=MC_SAMPLES_DEFAULT):
    if len(hamiltonian_list) == 0:
        print("[find_optimal_iterations] no hamiltonian terms provided?")
        return 1
    
    initial_state = np.zeros((hamiltonian_list[0].shape[0]))
    initial_state[0] = 1.
    # TODO SET SIMS TO USE initiail_state
    exact_final_state = exact_time_evolution(hamiltonian_list, time, initial_state=initial_state)

    # now we can simplify infidelity to nice lambda, NOTE THIS IS TUPLE 
    if type(simulator) == TrotterSim or type(simulator) == CompositeSim:
        get_inf = lambda x: mutli_infidelity_sample(simulator, time, exact_final_state, iterations = x, mc_samples=mc_samples)
    elif type(simulator) == QDriftSim:
        get_inf = lambda x: mutli_infidelity_sample(simulator, time, exact_final_state, nbsamples= x, mc_samples=mc_samples)

    # compute reasonable upper and lower bounds
    iter_lower = 1
    iter_upper = 2 ** 20
    # print("[find_optimal_iterations] finding bounds")
    for n in range(20):
        # print("[find_optimal_iterations] n: ", n)
        inf_tup, costs = zip(*get_inf(2 ** n))
        inf_list = list(inf_tup)
        # print("[find_optimal_iterations] mean infidelity:", np.mean(inf_list))

        if is_threshold_met(inf_list, infidelity_threshold) == False:
            iter_lower = 2 ** n
        else:
            iter_upper = 2 ** n
            break

    # bisection search until we find it.
    mid = 1
    count = 0
    current_inf = (1., 1)
    print("[find_optimal_iterations] beginning search with lower, upper:", iter_lower, iter_upper)
    while iter_upper - iter_lower  > 1 and count < 30:
        count += 1
        mid = (iter_upper + iter_lower) / 2.0
        iters = math.ceil(mid)
        # print("[find_optimal_iterations] count:", count, ", upper:",iter_upper, ", lower: ", iter_lower, ", mid:", mid)
        
        inf_tup, costs = zip(*get_inf(iters))
        infidelities = list(inf_tup)
        # print("[find_optimal_iterations] current_inf:", np.mean(infidelities))
        if is_threshold_met(infidelities, infidelity_threshold):
            iter_lower = iters
        else:
            iter_upper = iters
    ret = get_inf(iter_upper)
    inf_tup, costs = zip(*ret)
    print("[find_optimal_iterations] iters:", iter_upper, ", inf_mean: ", np.mean(list(inf_tup)), " +- (", np.std(list(inf_tup)), ")")
    print("[find_optimal_iterations] Average cost:", np.mean(list(costs)))
    return get_inf(iter_upper)
# Helper function to compute the timesteps to matrix exponentials in a higher order
# product formula. This can be done with only the length of an array, the time, and
# the order of the simulator needed. Returns a list of tuples of the form (index, time),
# where index refers to which hamiltonian term at that step and time refers to the scaled time.
# Assumes your list is of the form [H_1, H_2, ... H_numTerms] 
# For example if you want to do e^{i H_3 t} e^{i H_2 t} e^{i H_1 t} | psi >, then calling this with
# compute_trotter_timesteps(3, t, 1) will return [(0, t), (1, t), (2, t)] where we assume your
# Hamiltonian terms are stored like [H_1, H_2, H_3] and we return the index
# Note the reverse ordering due to matrix multiplication :'( 
def compute_trotter_timesteps(numTerms, simTime, trotterOrder = 1):
    if type(trotterOrder) != type(1):
        print('[compute_trotter_timesteps] trotterOrder input is not an int')
        return 1

    if trotterOrder == 1:
        return [(ix, simTime) for ix in range(numTerms)]

    elif trotterOrder == 2:
        ret = []
        firstOrder = compute_trotter_timesteps(numTerms, simTime / 2.0, 1)
        ret += firstOrder.copy()
        firstOrder.reverse()
        ret += firstOrder
        return ret

    elif trotterOrder % 2 == 0:
        timeConst = 1.0/(4 - 4**(1.0 / (trotterOrder - 1)))
        outer = compute_trotter_timesteps(numTerms, timeConst * simTime, trotterOrder - 2)
        inner = compute_trotter_timesteps(numTerms, (1. - 4. * timeConst) * simTime, trotterOrder - 2)
        ret = [] + 2 * outer + inner + 2 * outer
        return ret

    else:
        print("[compute_trotter_timesteps] trotterOrder seems to be bad")
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
        self.exp_op_cache = dict()
        self.conj_cache = dict()

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
    
    def get_hamiltonian_list(self):
        ret = []
        for ix in range(len(self.hamiltonian_list)):
            ret.append(np.copy(self.hamiltonian_list[ix]) * self.spectral_norms[ix])
        return ret

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        self.initial_state = psi_init

    def reset_init_state(self):
        if len(self.hamiltonian_list) > 0:
            self.initial_state = np.zeros((self.hamiltonian_list[0].shape[0]))
            self.initial_state[0] = 1.
        #else:
            #print("[TrotterSim.reset_init_state] I don't have any dimensions")

    def simulate(self, time, iterations):
        self.gate_count = 0
        if len(self.hamiltonian_list) == 0:
            return np.copy(self.initial_state)

        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["iterations"] != iterations):
                self.exp_op_cache.clear()
        
        if self.exp_op_cache == {}:
            self.exp_op_cache["time"] = time
            self.exp_op_cache["iterations"] = iterations
            op_time = time/iterations
            steps = compute_trotter_timesteps(len(self.hamiltonian_list), op_time, self.order)
            key = 0
            for (ix, timestep) in steps:
                self.exp_op_cache[key] = linalg.expm(1j * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep)
                key +=1
        num_ops = len(self.exp_op_cache) - 2 #2 of the keys are for time and iters
        psi = np.copy(self.initial_state)
        for k in range(num_ops * iterations): 
            psi = self.exp_op_cache[k % num_ops] @ psi
            self.final_state = psi
            self.gate_count += 1
        return psi

    def simulate_density(self, time, iterations):
        self.gate_count = 0
        if len(self.hamiltonian_list) == 0:
            return np.copy(self.initial_state)

        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["iterations"] != iterations):
                self.exp_op_cache.clear()
        
        if self.exp_op_cache == {}:
            self.exp_op_cache["time"] = time
            self.exp_op_cache["iterations"] = iterations
            op_time = time/iterations
            steps = compute_trotter_timesteps(len(self.hamiltonian_list), op_time, self.order)
            key = 0
            for (ix, timestep) in steps:
                self.exp_op_cache[key] = linalg.expm(1j * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep)
                self.conj_cache[key] = self.exp_op_cache.get(key).conj().T
                key +=1
        num_ops = len(self.exp_op_cache) - 2 #2 of the keys are for time and iters
        psi = np.copy(self.initial_state)
        for k in range(num_ops * iterations): 
            psi = self.exp_op_cache[k % num_ops] @ psi @ self.conj_cache[k % num_ops]
            self.final_state = psi
            self.gate_count += 1
        return psi

    def infidelity(self, time, iterations):
        H = []
        for i in range(len(self.spectral_norms)):
            H.append(self.hamiltonian_list[i] * self.spectral_norms[i]) 
        sim_state = self.simulate(time, iterations)
        good_state = np.dot(linalg.expm(1j * sum(H) * time), self.initial_state)
        infidelity = 1 - (np.abs(np.dot(good_state.conj().T, sim_state)))**2
        #infidelity = (np.linalg.norm(sim_state - good_state)) #-- Euclidean distance                    
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
    
class QDriftSim:
    def __init__(self, hamiltonian_list = [], rng_seed = 1):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.rng_seed = rng_seed
        self.gate_count = 0
        self.exp_op_cache = dict()
        self.conj_cache = dict()

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
    
    def get_hamiltonian_list(self):
        ret = []
        for ix in range(len(self.hamiltonian_list)):
            ret.append(np.copy(self.hamiltonian_list[ix]) * self.spectral_norms[ix])
        return ret

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    def set_initial_state(self, psi_init):
        self.initial_state = psi_init

    def reset_init_state(self):
        if len(self.hamiltonian_list) > 0:
            self.initial_state = np.zeros((self.hamiltonian_list[0].shape[0]))
            self.initial_state[0] = 1.
        #else:
            #print("[qdrift_sim reset_init_state] I have no dims")

    # Assumes terms are already normalized
    def set_hamiltonian(self, mat_list = [], norm_list = []):
        if len(mat_list) != len(norm_list):
            print("[QD - set_hamiltonian] Incorrect length arrays")
            return 1
        self.hamiltonian_list = mat_list
        self.spectral_norms = norm_list

    # RETURNS A 0 BASED INDEX TO BE USED IN CODE!!
    def draw_hamiltonian_samples(self, num_samples):
        samples = []
        for i in range(num_samples):
            sample = np.random.random()
            tot = 0.
            lamb = np.sum(self.spectral_norms)
            for ix in range(len(self.spectral_norms)):
                if sample > tot and sample < tot + self.spectral_norms[ix] / lamb:
                    index = ix
                    break
                tot += self.spectral_norms[ix] / lamb
            samples.append(index)
        return samples
    
    def simulate(self, time, samples):
        self.gate_count = 0
        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["samples"] != samples):
                self.exp_op_cache.clear()
                # WARNING: potential could allow for "time creep", by adjusting time 
                # in multiple instances of FLOATING POINT PRECISION it could slowly
                # drift from the time that the exponential operators used

        if (len(self.hamiltonian_list) == 0): # or (len(self.hamiltonian_list) == 1) caused issues in comp sim
            return np.copy(self.initial_state) #make the choice not to sample a lone qdrift term

        tau = time * np.sum(self.spectral_norms) / (samples * 1.0)
        self.exp_op_cache["time"] = time
        self.exp_op_cache["samples"] = samples
        final = np.copy(self.initial_state)
        obtained_samples = self.draw_hamiltonian_samples(samples)
        for ix in obtained_samples:
            if ix in self.exp_op_cache:
                final = self.exp_op_cache.get(ix) @ final
            else:
                exp_h = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
                final = exp_h @ final
                self.exp_op_cache[ix] = exp_h
        self.final_state = final
        self.gate_count = samples
        return np.copy(self.final_state)

    def simulate_density(self, time, samples):
        self.gate_count = 0
        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["samples"] != samples):
                self.exp_op_cache.clear()

        if (len(self.hamiltonian_list) == 0): # or (len(self.hamiltonian_list) == 1) caused issues in comp sim
            return np.copy(self.initial_state) #make the choice not to sample a lone qdrift term

        tau = time * np.sum(self.spectral_norms) / (samples * 1.0)
        self.exp_op_cache["time"] = time
        self.exp_op_cache["samples"] = samples
        final = np.copy(self.initial_state)
        obtained_samples = self.draw_hamiltonian_samples(samples)
        for ix in obtained_samples:
            if ix in self.exp_op_cache:
                final = self.exp_op_cache.get(ix) @ final @ self.exp_op_cache.get(ix).conj().T
            else:
                exp_h = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
                final = exp_h @ final @ exp_h.conj().T
                self.exp_op_cache[ix] = exp_h
        self.final_state = final
        self.gate_count = samples
        return np.copy(self.final_state)

    def construct_density(self, time, samples):
        self.gate_count = 0
        lamb = np.sum(self.spectral_norms)
        tau = time * lamb / (samples * 1.0)
        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["samples"] != samples) or (len(self.conj_cache) != len(self.spectral_norms)): #incase partition changes
                self.exp_op_cache.clear()
                self.conj_cache.clear() #based on code will follow the conditions above

        if (len(self.hamiltonian_list) == 0): # or (len(self.hamiltonian_list) == 1) caused issues in comp sim
            return np.copy(self.initial_state) #make the choice not to sample a lone qdrift term
        
        if self.exp_op_cache == {}:
            self.exp_op_cache["time"] = time
            self.exp_op_cache["samples"] = samples
            for k in range(len(self.spectral_norms)):
                self.exp_op_cache[k] = linalg.expm(1.j * tau * self.hamiltonian_list[k])
                self.conj_cache[k] = self.exp_op_cache.get(k).conj().T

        rho = np.copy(self.initial_state)
        for i in range(samples):
            channel_output = np.zeros((self.hamiltonian_list[0].shape[0], self.hamiltonian_list[0].shape[0]), dtype = 'complex')
            for j in range(len(self.spectral_norms)):
                channel_output += (self.spectral_norms[j]/lamb) * self.exp_op_cache.get(j) @ rho @ self.conj_cache.get(j) #an error is creeping in here (I think for the case len(b) = 1)
            rho = channel_output

        self.final_state = rho
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
    def __init__(self, hamiltonian_list = [], inner_order = 1, outer_order = 1, initial_time = 0.1, partition = "random", 
    rng_seed = 1, nb_optimizer = False, weight_threshold = 0.5, nb = 1, epsilon = 0.001, state_rand = False):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.a_norms = [] #contains the partitioned norms, as well as the index of the matrix they come from
        self.b_norms = [] 
        self.hilbert_dim = hamiltonian_list[0].shape[0] 
        self.rng_seed = rng_seed
        self.outer_order = outer_order 
        self.inner_order = inner_order
        self.partition = partition
        self.nb_optimizer = nb_optimizer
        self.epsilon = epsilon #simulation error
        self.weight_threshold = weight_threshold
        self.state_rand = state_rand

        self.qdrift_sim = QDriftSim()
        self.trotter_sim = TrotterSim(order = inner_order)

        self.nb = nb #number of Qdrift channel samples. Useful to define as an attribute if we are choosing whether or not to optimize over it. Only used in analytic cost optimization
        self.time = initial_time 
        self.gate_count = 0 #Used to keep track of the operators in analyse_sim
        self.gate_data = [] #used to access the gate counts in the notebook
        self.optimized_gatecost = 0 #Used to store the optimized gate cost in the partition methods that optimize the overall sim cost

        #Choose to randomize the initial state or just use computational |0>
        #Should probably add functionality to take an initial state as input at some point
        if self.state_rand == True:
            self.initial_state = initial_state_randomizer(self.hilbert_dim)
        else:
            # Use the first computational basis state as the initial state until the user specifies.
            self.initial_state = np.zeros((self.hilbert_dim, 1))
            self.initial_state[0] = 1.
        
        self.final_state = np.copy(self.initial_state)
        self.unparsed_hamiltonian = np.copy(hamiltonian_list) #the unmodified input matrix

        self.prep_hamiltonian_lists(hamiltonian_list) #do we want this done before or after the partitioning?
        np.random.seed(self.rng_seed)
        self.partitioning(self.weight_threshold) #note error was raised because partition() is a built in python method
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

    def get_hamiltonian_list(self):
        ret = []
        for ix in range(len(self.hamiltonian_list)):
            ret.append(np.copy(self.hamiltonian_list[ix]) * self.spectral_norms[ix])
        return ret

    def reset_init_state(self):
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1.
        self.trotter_sim.reset_init_state()
        self.qdrift_sim.reset_init_state()

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

    ################################################################
    #OPTIMIZATION FUNCTIONS
    ################################################################
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

    ####################
    #PARTITIONING METHOD to execute the partitioning method of the users choice, random likely not used in practice
    ####################
    def partitioning(self, weight_threshold):
        #if ((self.partition == "trotter" or self.partition == "qdrift" or self.partition == "random" or self.partition == "optimize")): #This condition may change for the optimize scheme later
            #print("This partitioning method does not require repartitioning") #Checks that our scheme is sane, prevents unnecessary time wasting
            #return 1 maybe work this in again later?

        if self.partition == "prob":
            if self.trotter_sim.order > 1: k = self.trotter_sim.order/2
            else: 
                raise Exception("partition not defined for this order")
            
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
            
            print("Nb is " + str(self.nb))
            
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
                    if optimized_weights.x[i] >= weight_threshold:
                        self.a_norms.append([i, self.spectral_norms[i]])
                    elif optimized_weights.x[i] < weight_threshold:
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
                    if optimized_weights.x[i] >= weight_threshold:
                        self.a_norms.append([i, self.spectral_norms[i]])
                    elif optimized_weights.x[i] < weight_threshold:
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
                if self.spectral_norms[i] >= weight_threshold:
                    self.a_norms.append(([i, self.spectral_norms[i]]))
                else:
                    self.b_norms.append(([i, self.spectral_norms[i]]))
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        elif self.partition == "optimal chop": 
            #This partition method is to be used differently than the others in the notebook. It optimizes the gate cost, 
            #so there is no need to run sim_channel_performance again, instead just call repartitioning for a given time
            #and the cost is stored in the attribute self.optimized_gatecost. This function relies on self.time which is handled
            #by repartition()
            w_guess = sum(self.spectral_norms)/len(self.spectral_norms) #guess the average for the weights
            nb_guess = (1/2) * len(self.spectral_norms)
            condition = self.nb_optimizer
            if condition == True:
                dim1 = Integer(name='samples', low=0, high= len(self.spectral_norms) * 100)
                dim2 = Real(name='weight_threshold', low=0, high = max(self.spectral_norms))
                dimensions = [dim1, dim2]
            else:
                dim2 = Real(name='weight_threshold', low=0, high=max(self.spectral_norms))
                dimensions = [dim2]
            self.partition = "chop" #A trick to optimize the chop partition method when the function below calls self.partitioning

            #A function similar to that of sim_channel_performance, however, this one is defined only to be optimized not executed
            @use_named_args(dimensions=dimensions)
            def nb_optimal_performance(samples, weight_threshold):  
                good_inf = []
                bad_inf = []
                mcsamples = 250 #might want to fix how this is handled
                self.nb = samples
                self.repartition(self.time, weight_threshold = weight_threshold) #time is being dealt with in a weird way
            
                exact_state = exact_time_evolution(self.unparsed_hamiltonian, self.time, self.initial_state)
                get_inf = lambda x: self.sample_channel_inf(self.time, samples = samples, iterations = x, mcsamples = mcsamples, exact_state = exact_state)
                median_samples = 1
                lower_bound = 1
                upper_bound = 1
                
                #Exponential search to set upper bound, then binary search in that interval
                infidelity = get_inf(lower_bound)
                if infidelity < self.epsilon:
                    print("SAMPLE COUNT: " + str(samples))
                    print("[sim_channel_performance] Iterations too large, already below error threshold")
                    return self.gate_count
                # Iterate up until some max cutoff
                break_flag = False
                upper_bound = upper_bound*2 
                for n in range(10):
                    infidelity = get_inf(upper_bound) 
                    if infidelity < self.epsilon:
                        break_flag = True
                        break
                    else:
                        upper_bound *= (upper_bound)
                        print(infidelity, self.gate_count, upper_bound)
                if break_flag == False :
                    raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
                print("the upper bound is " + str(upper_bound))

                #catch an edge case
                if upper_bound == 2:
                    return self.gate_count

                #Binary search
                break_flag_2 = False
                while lower_bound < upper_bound:
                    mid = lower_bound + (upper_bound - lower_bound)//2
                    if (mid == 2) or (mid ==1): 
                        break_flag_2 = True
                        break #catching another edge case
                    inf_plus = []
                    inf_minus = []
                    for n in range(median_samples):
                        inf_plus.append(get_inf(mid+2))
                        inf_minus.append(get_inf(mid-2))
                    med_inf_plus = statistics.median(inf_plus)
                    med_inf_minus = statistics.median(inf_minus)
                    print((med_inf_minus - self.epsilon, self.epsilon - med_inf_plus, upper_bound, lower_bound))
                    if (med_inf_plus < self.epsilon) and (med_inf_minus > self.epsilon): #Causing Problems
                        break_flag_2 = True
                        break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
                    elif med_inf_plus < self.epsilon:
                        upper_bound = mid - 1
                    else:
                        lower_bound = mid + 1
                if break_flag_2 == False:
                    print("[sim_channel_performance] function did not find a good point")

                #Compute some surrounding points and interpolate
                for i in range (mid+1, mid +3):
                    infidelity = get_inf(i)
                    good_inf.append([self.gate_count, float(infidelity)])
                for j in range (max(mid-2, 1), mid +1): #catching an edge case
                    infidelity = get_inf(j)
                    bad_inf.append([self.gate_count, float(infidelity)])
                #Store the points to interpolate
                good_inf = np.array(good_inf)
                bad_inf = np.array(bad_inf)
                self.gate_data = np.concatenate((bad_inf, good_inf), 0)
                #print(self.gate_data)
                fit = np.poly1d(np.polyfit(self.gate_data[:,1], self.gate_data[:,0], 1)) #linear best fit 
                return fit(self.epsilon)

            @use_named_args(dimensions=dimensions)
            def optimal_performance(weight_threshold):  
                good_inf = []
                bad_inf = []
                mcsamples = 250 #might want to fix how this is handled
                samples = self.nb  #this function is exactly the same as above with this change
                self.repartition(self.time, weight_threshold = weight_threshold) #time is being dealt with in a weird way
            
                exact_state = exact_time_evolution(self.unparsed_hamiltonian, self.time, self.initial_state)
                get_inf = lambda x: self.sample_channel_inf(self.time, samples = samples, iterations = x, mcsamples = mcsamples, exact_state = exact_state)
                median_samples = 1
                lower_bound = 1
                upper_bound = 1
                
                #Exponential search to set upper bound, then binary search in that interval
                infidelity = get_inf(lower_bound)
                if infidelity < self.epsilon:
                    print("SAMPLE COUNT: " + str(samples))
                    print("[sim_channel_performance] Iterations too large, already below error threshold")
                    return self.gate_count
                # Iterate up until some max cutoff
                break_flag = False
                upper_bound = upper_bound*2 
                for n in range(10):
                    infidelity = get_inf(upper_bound) 
                    if infidelity < self.epsilon:
                        break_flag = True
                        break
                    else:
                        upper_bound *= (upper_bound)
                        print(infidelity, self.gate_count, upper_bound)
                if break_flag == False :
                    raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
                print("the upper bound is " + str(upper_bound))

                #catch an edge case
                if upper_bound == 2:
                    return self.gate_count

                #Binary search
                break_flag_2 = False
                while lower_bound < upper_bound:
                    mid = lower_bound + (upper_bound - lower_bound)//2
                    if (mid == 2) or (mid ==1): 
                        break_flag_2 = True
                        break #catching another edge case
                    inf_plus = []
                    inf_minus = []
                    for n in range(median_samples):
                        inf_plus.append(get_inf(mid+2))
                        inf_minus.append(get_inf(mid-2))
                    med_inf_plus = statistics.median(inf_plus)
                    med_inf_minus = statistics.median(inf_minus)
                    print((med_inf_minus - self.epsilon, self.epsilon - med_inf_plus, upper_bound, lower_bound))
                    if (med_inf_plus < self.epsilon) and (med_inf_minus > self.epsilon): #Causing Problems
                        break_flag_2 = True
                        break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
                    elif med_inf_plus < self.epsilon:
                        upper_bound = mid - 1
                    else:
                        lower_bound = mid + 1
                if break_flag_2 == False:
                    print("[sim_channel_performance] function did not find a good point")

                #Compute some surrounding points and interpolate
                for i in range (mid+1, mid +3):
                    infidelity = get_inf(i)
                    good_inf.append([self.gate_count, float(infidelity)])
                for j in range (max(mid-2, 1), mid +1): #catching an edge case
                    infidelity = get_inf(j)
                    bad_inf.append([self.gate_count, float(infidelity)])
                #Store the points to interpolate
                good_inf = np.array(good_inf)
                bad_inf = np.array(bad_inf)
                self.gate_data = np.concatenate((bad_inf, good_inf), 0)
                #print(self.gate_data)
                fit = np.poly1d(np.polyfit(self.gate_data[:,1], self.gate_data[:,0], 1)) #linear best fit 
                return fit(self.epsilon)

            if condition == True: #the case where we optimize nb
                result = gbrt_minimize(func=nb_optimal_performance,dimensions=dimensions, n_calls=30, n_initial_points = 5, 
                random_state=4, verbose = True, acq_func = "LCB")
                self.nb = result.x[0]
            else: 
                result = gbrt_minimize(func=optimal_performance,dimensions=dimensions, n_calls=20, n_initial_points = 5, 
                random_state=4, verbose = True, acq_func = "LCB")
                
            print(result.fun)
            print(result.x)
            self.optimized_gatecost = result.fun
            self.partition = "optimal chop" #reset for iterating
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

##########################################################################################
#Simulation Section -- contains functions to call in the workbook to simulate hamiltonians
##########################################################################################
    #A function that repartitions the hamiltonian. Ideally to be called when iterating simulate over timesteps
    def repartition(self, time, weight_threshold = None):
        self.a_norms = []
        self.b_norms = []
        self.time = time
        self.partitioning(weight_threshold)
        self.reset_nested_sims()
        return 0
    
    #a function to decide on a good number of monte carlo samples for the following simulation functions (avoid noise errors)
    def sample_decider(self, time, samples, iterations, mc_sample_guess):
        exact_state = exact_time_evolution(self.unparsed_hamiltonian, time, self.initial_state)
        sample_guess = mc_sample_guess
        inf_samples = [1, 0]
        for k in range(1, 25):
            inf_samples[k%2] = self.sample_channel_inf(time, samples, iterations, sample_guess, exact_state)
            print(inf_samples)
            if np.abs((inf_samples[0] - inf_samples[1])) < (0.1 * self.epsilon): #choice of precision
                break
            else:
                sample_guess *= 2
        return int(sample_guess/2)

    #Simulate and error scaling
    def simulate(self, time, samples, iterations): 
        if (self.nb_optimizer == False) and (self.partition != 'prob'): 
            self.nb = samples  #specifying the number of samples having optimized Nb does nothing
        if len(self.b_norms) == 1: self.nb = 1 #edge case, dont sameple the same gate over and over again
        self.gate_count = 0
        outer_loop_timesteps = compute_trotter_timesteps(2, time / (1. * iterations), self.outer_order)
        self.reset_init_state()
        current_state = np.copy(self.initial_state)
        for i in range(iterations):
            for (ix, sim_time) in outer_loop_timesteps:
                if ix == 0:
                    self.trotter_sim.set_initial_state(current_state)
                    current_state = self.trotter_sim.simulate(sim_time, 1)
                    self.gate_count += self.trotter_sim.gate_count
                if ix == 1:
                    self.qdrift_sim.set_initial_state(current_state)
                    current_state = self.qdrift_sim.simulate(sim_time, self.nb)
                    self.gate_count += self.qdrift_sim.gate_count
        return current_state
            
    # A function that uses Monte-Carlo sampling to approximate the infidelity. If evaluating a Trotter channel, mcsamples should be set to 1.
    def sample_channel_inf(self, time, samples, iterations, mcsamples, exact_state): 
        sample_fidelity = 0
        for s in range(mcsamples):
            sim_state = self.simulate(time, samples, iterations)
            sample_fidelity += (np.abs(np.dot(exact_state.conj().T, sim_state).flat[0]))**2
        infidelity = 1- (sample_fidelity / mcsamples) 
        return infidelity #this is of type array for some reason?

    # A function to analyse the cost of a simulation with a certain error tolerance. The funciton is expensive -- it searches for the point 
    # (the number of iterations or samples) at which the error tolerance is met, and does so with an exponential then binary search,
    # followed by a linear interpolation.
    def sim_channel_performance(self, time, samples, iterations, mcsamples): #This function is not robust due to mc noise 
        good_inf = []
        bad_inf = []
        exact_state = exact_time_evolution(self.unparsed_hamiltonian, time, self.initial_state)
        if self.partition == "qdrift":
            get_inf = lambda x: self.sample_channel_inf(time, samples = x, iterations = 1, mcsamples = mcsamples, exact_state = exact_state)
            median_samples = 1
            lower_bound = samples
            upper_bound = samples
        elif self.partition == "trotter":
            get_inf = lambda x: self.sample_channel_inf(time, samples = 1, iterations = x, mcsamples = 1, exact_state = exact_state)
            median_samples = 1
            lower_bound = iterations
            upper_bound = iterations
        else:
            get_inf = lambda x: self.sample_channel_inf(time, samples = samples, iterations = x, mcsamples = mcsamples, exact_state = exact_state)
            median_samples = 1
            lower_bound = iterations
            upper_bound = iterations
        
        #Exponential search to set upper bound, then binary search in that interval
        infidelity = get_inf(lower_bound)
        if infidelity < self.epsilon:
            print("[sim_channel_performance] Iterations too large, already below error threshold")
            return self.gate_count
        # Iterate up until some max cutoff
        break_flag = False
        upper_bound = upper_bound*2 #incase user input is 1
        for n in range(10):
            infidelity = get_inf(upper_bound) 
            if infidelity < self.epsilon:
                break_flag = True
                break
            else:
                upper_bound *= (upper_bound)
                print(infidelity, self.gate_count)
        if break_flag == False :
            raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
        print("the upper bound is " + str(upper_bound))

        #catch an edge case
        if upper_bound == 2:
            return self.gate_count

        #Binary search
        break_flag_2 = False
        while lower_bound < upper_bound:
            mid = lower_bound+ (upper_bound - lower_bound)//2
            if (mid == 2) or (mid ==1): 
                break_flag_2 = True
                break #catching another edge case
            inf_plus = []
            inf_minus = []
            for n in range(median_samples):
                inf_plus.append(get_inf(mid+2))
                inf_minus.append(get_inf(mid-2))
            med_inf_plus = statistics.median(inf_plus)
            med_inf_minus = statistics.median(inf_minus)
            print((med_inf_minus - self.epsilon, self.epsilon - med_inf_plus, upper_bound, lower_bound))
            if (med_inf_plus < self.epsilon) and (med_inf_minus > self.epsilon): #Causing Problems
                break_flag_2 = True
                break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
            elif med_inf_plus < self.epsilon:
                upper_bound = mid - 1
            else:
                lower_bound = mid + 1
        if break_flag_2 == False:
            print("[sim_channel_performance] function did not find a good point")

        #Compute some surrounding points and interpolate
        for i in range (mid+1, mid +3):
            infidelity = get_inf(i)
            good_inf.append([self.gate_count, float(infidelity)])
        for j in range (max(mid-2, 1), mid +1): #catching an edge case
            infidelity = get_inf(j)
            bad_inf.append([self.gate_count, float(infidelity)])
        #Store the points to interpolate
        good_inf = np.array(good_inf)
        bad_inf = np.array(bad_inf)
        self.gate_data = np.concatenate((bad_inf, good_inf), 0)
        #print(self.gate_data)
        fit = np.poly1d(np.polyfit(self.gate_data[:,1], self.gate_data[:,0], 1)) #linear best fit 
        return fit(self.epsilon)

######################################################################################################
#To perform the same as above, but with the density matrix formalisim to avoid the use of infidelity
######################################################################################################
class DensityMatrixSim:
    def __init__(self, hamiltonian_list = [], inner_order = 1, outer_order = 1, initial_time = 0.1, partition = "random", 
    rng_seed = 1, nb_optimizer = False, weight_threshold = 0.5, nb = 1, epsilon = 0.001, state_rand = False, pure = True):

        self.hamiltonian_list = []
        self.spectral_norms = []
        self.a_norms = [] #contains the partitioned norms, as well as the index of the matrix they come from
        self.b_norms = [] 
        self.hilbert_dim = hamiltonian_list[0].shape[0] 
        self.rng_seed = rng_seed
        self.outer_order = outer_order 
        self.inner_order = inner_order
        self.partition = partition
        self.nb_optimizer = nb_optimizer
        self.epsilon = epsilon #simulation error
        self.weight_threshold = weight_threshold
        self.state_rand = state_rand

        self.initial_rho = np.zeros((self.hilbert_dim, 2)) #density matrix
        self.pure = pure #boolean for pure or mixed states

        self.qdrift_sim = QDriftSim()
        self.trotter_sim = TrotterSim(order = inner_order)

        self.nb = nb #number of Qdrift channel samples. Useful to define as an attribute if we are choosing whether or not to optimize over it. Only used in analytic cost optimization
        self.time = initial_time 
        self.gate_count = 0 #Used to keep track of the operators in analyse_sim
        self.gate_data = [] #used to access the gate counts in the notebook
        self.optimized_gatecost = 0 #Used to store the optimized gate cost in the partition methods that optimize the overall sim cost

        #Choose to randomize the initial state or just use computational |0>
        #Should probably add functionality to take an initial state as input at some point
        if self.state_rand == True:
            self.initial_state = initial_state_randomizer(self.hilbert_dim)
        else:
            # Use the first computational basis state as the initial state until the user specifies.
            self.initial_state = np.zeros((self.hilbert_dim, 1))
            self.initial_state[0] = 1.
        
        if self.pure == True:
            self.prep_pure_rho()
        else:
            self.prep_pure_rho()
            #raise Exception("mixed states not yet supported") -- comment out for now
            print('mixed states not supported')

        self.final_rho = np.copy(self.initial_rho)
        self.unparsed_hamiltonian = np.copy(hamiltonian_list) #the unmodified input matrix

        self.prep_hamiltonian_lists(hamiltonian_list) #do we want this done before or after the partitioning?
        np.random.seed(self.rng_seed)
        self.partitioning(self.weight_threshold) #note error was raised because partition() is a built in python method
        self.reset_nested_sims()

        print("There are " + str(len(self.a_norms)) + " terms in Trotter") #make the partition known
        print("There are " + str(len(self.b_norms)) + " terms in QDrift")
        if nb_optimizer == True:
            print("Nb is equal to " + str(self.nb))

    #Class functions
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

    def prep_pure_rho(self):
        #print(self.initial_state, self.initial_state.conj())
        self.initial_rho = np.outer(self.initial_state, self.initial_state.conj())

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

    def repartition(self, time, weight_threshold = None):
        self.a_norms = []
        self.b_norms = []
        self.time = time
        self.partitioning(weight_threshold)
        self.reset_nested_sims()
        return 0

    def reset_init_state(self):
        self.initial_state = np.zeros((self.hilbert_dim, self.hilbert_dim))
        self.initial_state[0] = 1.
        self.trotter_sim.reset_init_state()
        self.qdrift_sim.reset_init_state()

#Optimizations functions
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

#Partitioning       
    def partitioning(self, weight_threshold):
        #if ((self.partition == "trotter" or self.partition == "qdrift" or self.partition == "random" or self.partition == "optimize")): #This condition may change for the optimize scheme later
            #print("This partitioning method does not require repartitioning") #Checks that our scheme is sane, prevents unnecessary time wasting
            #return 1 maybe work this in again later?

        if self.partition == "prob":
            if self.trotter_sim.order > 1: k = self.trotter_sim.order/2
            else: 
                raise Exception("partition not defined for this order") 
                
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
            
            print("Nb is " + str(self.nb))
            
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
                if self.spectral_norms[i] >= weight_threshold:
                    self.a_norms.append(([i, self.spectral_norms[i]]))
                else:
                    self.b_norms.append(([i, self.spectral_norms[i]]))
            self.a_norms = np.array(self.a_norms, dtype='complex')
            self.b_norms = np.array(self.b_norms, dtype='complex')
            return 0

        elif self.partition == "optimal chop": 
            #This partition method is to be used differently than the others in the notebook. It optimizes the gate cost, 
            #so there is no need to run sim_channel_performance again, instead just call repartitioning for a given time
            #and the cost is stored in the attribute self.optimized_gatecost. This function relies on self.time which is handled
            #by repartition()
            w_guess = statistics.median(self.spectral_norms) #guess the median for the weights
            nb_guess = int(len(self.spectral_norms))
            condition = self.nb_optimizer
            if condition == True:
                dim1 = Integer(name='samples', low=1, high= len(self.spectral_norms) * 20)
                dim2 = Real(name='weight_threshold', low=0, high = max(self.spectral_norms))
                dimensions = [dim1, dim2]
            else:
                dim2 = Real(name='weight_threshold', low=0, high=max(self.spectral_norms))
                dimensions = [dim2]
            self.partition = "chop" #A trick to optimize the chop partition method when the function below calls self.partitioning

            #A function similar to that of sim_channel_performance, however, this one is defined only to be optimized not executed
            @use_named_args(dimensions=dimensions)
            def nb_optimal_performance(samples, weight_threshold):  
                self.nb = samples
                time = self.time
                self.repartition(self.time, weight_threshold = weight_threshold) #time is being dealt with in a weird way

                get_trace_dist = lambda x : self.sim_trace_distance(time, samples, iterations=x)
                lower_bound = 1
                upper_bound = 1

                trace_dist = get_trace_dist(lower_bound)
                if trace_dist < self.epsilon:
                    print("[sim_channel_performance] Iterations too large, already below error threshold")
                    return self.gate_count
                # Iterate up until some max cutoff
                break_flag = False
                upper_bound = upper_bound*2 #incase user input is 1
                for n in range(20):
                    trace_dist = get_trace_dist(upper_bound) 
                    if trace_dist < self.epsilon:
                        break_flag = True
                        break
                    else:
                        upper_bound *= 2
                        #print(trace_dist, self.gate_count)
                if break_flag == False :
                    raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
                #print("the upper bound is " + str(upper_bound))

                if upper_bound == 2:
                    return self.gate_count

                #Binary search
                break_flag_2 = False
                while lower_bound < upper_bound:
                    mid = lower_bound + (upper_bound - lower_bound)//2
                    if (mid == 2) or (mid ==1): 
                        return self.gate_count #catching another edge case
                    if (get_trace_dist(mid +1) < self.epsilon) and (get_trace_dist(mid-1) > self.epsilon): #Causing Problems
                        break_flag_2 = True
                        break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
                    elif get_trace_dist(mid) < self.epsilon:
                        upper_bound = mid - 1
                    else:
                        lower_bound = mid + 1
                if break_flag_2 == False:
                    print("[sim_channel_performance] function did not find a good point")

                get_trace_dist(mid)
                return self.gate_count

            @use_named_args(dimensions=dimensions)
            def optimal_performance(weight_threshold):  
                samples = self.nb
                time = self.time
                self.repartition(self.time, weight_threshold = weight_threshold) #time is being dealt with in a weird way

                get_trace_dist = lambda x : self.sim_trace_distance(time, samples, iterations=x)
                lower_bound = 1
                upper_bound = 1

                trace_dist = get_trace_dist(lower_bound)
                if trace_dist < self.epsilon:
                    print("[sim_channel_performance] Iterations too large, already below error threshold")
                    return self.gate_count
                # Iterate up until some max cutoff
                break_flag = False
                upper_bound = upper_bound*2 #incase user input is 1
                for n in range(20):
                    trace_dist = get_trace_dist(upper_bound) 
                    if trace_dist < self.epsilon:
                        break_flag = True
                        break
                    else:
                        upper_bound *= 2
                        #print(trace_dist, self.gate_count)
                if break_flag == False :
                    raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
                #print("the upper bound is " + str(upper_bound))

                if upper_bound == 2:
                    return self.gate_count

                #Binary search
                break_flag_2 = False
                while lower_bound < upper_bound:
                    mid = lower_bound + (upper_bound - lower_bound)//2
                    if (mid == 2) or (mid ==1): 
                        return self.gate_count #catching another edge case
                    if (get_trace_dist(mid +1) < self.epsilon) and (get_trace_dist(mid-1) > self.epsilon): #Causing Problems
                        break_flag_2 = True
                        break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
                    elif get_trace_dist(mid) < self.epsilon:
                        upper_bound = mid - 1
                    else:
                        lower_bound = mid + 1
                if break_flag_2 == False:
                    print("[sim_channel_performance] function did not find a good point")

                get_trace_dist(mid)
                return self.gate_count

            if condition == True: #the case where we optimize nb
                result = gbrt_minimize(func=nb_optimal_performance,dimensions=dimensions, n_calls=20, n_initial_points = 3, 
                random_state=4, verbose = False, acq_func = "LCB", x0 = [nb_guess, w_guess])
                self.nb = result.x[0]
            else: 
                result = gbrt_minimize(func=optimal_performance,dimensions=dimensions, n_calls=15, n_initial_points = 3, 
                random_state=4, verbose = False, acq_func = "LCB", x0 = [w_guess])
                
            #print(result.fun)
            #print(result.x)
            self.optimized_gatecost = result.fun
            self.partition = "optimal chop" #reset for iterating
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

#Simulate and error analysis
    def simulate(self, time, samples, iterations): 
        if (self.nb_optimizer == False) and (self.partition != 'prob'): 
            self.nb = samples  #specifying the number of samples having optimized Nb does nothing
        if len(self.b_norms) == 1: self.nb = 1 #edge case, dont sameple the same gate over and over again
        self.gate_count = 0
        outer_loop_timesteps = compute_trotter_timesteps(2, time / (1. * iterations), self.outer_order)
        #self.reset_init_state() #causes problems in the case where we output the outer product (pure state case)
        if self.pure == True:
            current_state = np.copy(self.initial_state)
            for i in range(iterations):
                for (ix, sim_time) in outer_loop_timesteps:
                    if ix == 0:
                        self.trotter_sim.set_initial_state(current_state)
                        current_state = self.trotter_sim.simulate(sim_time, 1)
                        self.gate_count += self.trotter_sim.gate_count
                    if ix == 1:
                        self.qdrift_sim.set_initial_state(current_state)
                        current_state = self.qdrift_sim.simulate(sim_time, self.nb)
                        self.gate_count += self.qdrift_sim.gate_count
            return np.outer(current_state, current_state.conj())
        else:
            current_state = np.copy(self.initial_rho)
            for i in range(iterations):
                for (ix, sim_time) in outer_loop_timesteps:
                    if ix == 0:
                        self.trotter_sim.set_initial_state(current_state)
                        current_state = self.trotter_sim.simulate_density(sim_time, 1)
                        self.gate_count += self.trotter_sim.gate_count
                    if ix == 1:
                        self.qdrift_sim.set_initial_state(current_state)
                        current_state = self.qdrift_sim.construct_density(sim_time, self.nb)
                        self.gate_count += self.qdrift_sim.gate_count
            return current_state

    def sim_trace_distance(self, time, samples, iterations):
        sim_density_op = self.simulate(time, samples, iterations)
        exact_density_op = exact_time_evolution_density(self.unparsed_hamiltonian, time, self.initial_rho)
        trace_dist = trace_distance(sim_density_op, exact_density_op)
        return trace_dist

    def sim_channel_performance(self, time):  
        if self.partition == "qdrift":
            get_trace_dist = lambda x : self.sim_trace_distance(time, samples = x, iterations=1)
        elif self.partition == 'trotter':
            get_trace_dist = lambda x : self.sim_trace_distance(time, samples = 1, iterations=x)
        else: 
            get_trace_dist = lambda x : self.sim_trace_distance(time, samples = self.nb, iterations=x)
        
        lower_bound = 1
        upper_bound = 2
        trace_dist = get_trace_dist(lower_bound)
        if trace_dist < self.epsilon:
            print("[sim_channel_performance] Iterations too large, already below error threshold")
            return self.gate_count
        # Iterate up until some max cutoff
        break_flag = False
        for n in range(27):
            trace_dist = get_trace_dist(upper_bound) 
            if trace_dist < self.epsilon:
                break_flag = True
                break
            else:
                upper_bound *= 2
                #print(trace_dist, self.gate_count)
        if break_flag == False:
            raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
        #print("the upper bound is " + str(upper_bound))

        if (upper_bound == 2):
            return self.gate_count
        #Binary search
        break_flag_2 = False
        while lower_bound < upper_bound:
            mid = lower_bound + (upper_bound - lower_bound)//2
            if (mid == 2) or (mid ==1): 
                return self.gate_count #catching another edge case
            if (get_trace_dist(mid +1) < self.epsilon) and (get_trace_dist(mid-1) > self.epsilon): #Causing Problems
                break_flag_2 = True
                break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
            elif get_trace_dist(mid) < self.epsilon:
                upper_bound = mid - 1
            else:
                lower_bound = mid + 1
        if break_flag_2 == False:
            print("[sim_channel_performance] function did not find a good point")

        get_trace_dist(mid)
        return self.gate_count

        #Compute some surrounding points and interpolate
        #good_dist = []
        #bad_dist = []
        #for i in range (mid+1, mid +3):
        #    trace_dist = get_trace_dist(i)
        #    good_dist.append([self.gate_count, float(trace_dist)])
        #for j in range (max(mid-2, 1), mid +1): #catching an edge case
        #    trace_dist = get_trace_dist(j)
        #    bad_dist.append([self.gate_count, float(trace_dist)])
        #Store the points to interpolate
        #good_dist = np.array(good_dist)
        #bad_dist = np.array(bad_dist)
        #self.gate_data = np.concatenate((bad_dist, good_dist), 0)
        #print(self.gate_data)
        #fit = np.poly1d(np.polyfit(self.gate_data[:,1], self.gate_data[:,0], 1)) #linear best fit 
        #return fit(self.epsilon)