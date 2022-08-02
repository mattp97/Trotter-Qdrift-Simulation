from ast import And
from asyncore import loop
from mimetypes import init
from operator import matmul
# from telnetlib import AYT
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import linalg
from scipy import optimize
from scipy import interpolate
import math
from numpy import arange, linspace, outer, random
import cmath
import time as time_this
import scipy
from sympy import S, symbols, printing
from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import cProfile, pstats, io
from compilers import CompositeSim, TrotterSim, QDriftSim, LRsim, profile, conditional_decorator

MC_SAMPLES_DEFAULT = 10
COST_LOOP_DEPTH = 30
ITERATION_BOUNDS_LOOP_DEPTH = 100 # specifies power of 2 for maximum number of iterations to search through
CROSSOVER_CUTOFF_PERCENTAGE = 0.01
POSSIBLE_PARTITIONS = ["first_order_trotter", "second_order_trotter", "qdrift"]

# A simple function that computes the graph distance between two sites
def dist(site1, site2):
    distance_vec = site1 - site2
    distance = np.abs(distance_vec[0]) + np.abs(distance_vec[1])
    return distance

# A simple function that initializes a graph in the form of an np.array of coordinates 
def initialize_graph(x_sites, y_sites):
    coord_list = []
    for i in range(x_sites):
        for j in range(y_sites):
            coord_list.append([i,j])
    return np.array(coord_list)

#A funciton that initializes a Pauli operator in the correct space, acting on a specific qubit
def initialize_operator(operator_2d, acting_space, space_dimension):
    I = np.array([[1, 0],
        [0, 1]])
    if acting_space>space_dimension:
        return 'error'
    for i in range(acting_space):
        operator_2d = np.kron(operator_2d, I)
    for j in range(space_dimension - acting_space-1):
        operator_2d = np.kron(I, operator_2d)
    return operator_2d
    
#Initialize Hamiltonian 
def graph_hamiltonian(x_dim, y_dim, rng_seed):
    X = np.array([[0, 1],
     [1, 0]])
    Z = np.array([[1, 0],
        [0, -1]])
    Y = np.array([[0, -1j],
        [1j, 0]])
    I = np.array([[1, 0],
        [0, 1]])
    np.random.seed(rng_seed)
    hamiltonian_list = []
    graph = initialize_graph(x_dim, y_dim)
    for i in range(x_dim*y_dim):
        for j in range(y_dim*x_dim):
            if i != j:
                alpha = np.random.normal()
                hamiltonian_list.append(alpha * 
                    np.matmul(initialize_operator(Z, i, x_dim*y_dim), initialize_operator(Z, j, x_dim*y_dim)) *
                        10.0**(-dist(graph[i], graph[j])))
            
        alpha = np.random.normal()
        hamiltonian_list.append(4* alpha * initialize_operator(X, i, x_dim*y_dim))
                
    return np.array(hamiltonian_list)

#A function to generate a random initial state that is normalized
# MH - This is not a good way to sample a uniform random state from high dimensional hilbert spaces, you get
#    "measure concentration" around specific directions. For small dims should work fine, for large dims tread
#    cautiously.
# def initial_state_randomizer(hilbert_dim):
#     initial_state = []
#     x = np.random.random(hilbert_dim)
#     y = np.random.random(hilbert_dim)
#     initial_state = x + (1j * y) 
#     initial_state_norm = initial_state / np.linalg.norm(initial_state)
#     return initial_state_norm

#A function to calculate the trace distance between two numpy arrays (density operators)
def trace_distance(rho, sigma):
    # MATT H: I think the below is probably inefficient, no need to compute entire eigendecomposition when a square root + trace will work.
    # diff = rho - sigma
    # w, v = np.linalg.eigh(diff)
    # dist = 1/2 * sum(np.abs(w))
    # return dist
    diff = rho - sigma
    tot = scipy.linalg.sqrtm(diff @ np.copy(diff).conj().T)
    return 0.5 * np.abs(np.trace(tot)) # Note: absolute value after trace is because we have 'complex' variables, so taking the norm should be fine??

def exact_time_evolution_density(hamiltonian_list, time, initial_rho):
    if len(hamiltonian_list) == 0:
        print("[exact_time_evolution] pls give me hamiltonian")
        return 1
    exp_op = linalg.expm(1j * sum(hamiltonian_list) * time)
    return exp_op @ initial_rho @ exp_op.conj().T


# Inputs are self explanatory except simulator which can be any of 
# TrotterSim, QDriftSim, CompositeSim
# Outputs: a single shot estimate of the infidelity according to the exact output provided. 
# @profile
def single_infidelity_sample(simulator, time, exact_final_state, iterations = 1, nbsamples = 1):
    sim_output = []
    # exact_output = simulator.simulate_exact_output(time)

    if type(simulator) == QDriftSim:
        sim_output = simulator.simulate(time, nbsamples)
    
    if type(simulator) == TrotterSim:
        sim_output = simulator.simulate(time, iterations)
    
    if type(simulator) == CompositeSim:
        sim_output = simulator.simulate(time, iterations)

    if type(simulator) == LRsim:
        sim_output = simulator.simulate(time, iterations)

    if simulator.use_density_matrices == False:
        infidelity = 1 - (np.abs(np.dot(exact_final_state.conj().T, sim_output)).flat[0])**2
    else:
        # check that shapes match
        if exact_final_state.shape != sim_output.shape:
            print("[single_infidelity_sample] tried computing density matrix infidelity with incorrect shapes.")
            print("[single_infidelity_sample] exact output shape =", exact_final_state.shape, ", sim output shape =", sim_output.shape)
            return 1.
        exact_sqrt = scipy.linalg.sqrtm(exact_final_state)
        tot_sqrt = scipy.linalg.sqrtm(np.linalg.multi_dot([exact_sqrt, sim_output, np.copy(exact_sqrt)]))
        fidelity = np.abs(np.trace(tot_sqrt))
        print("[single_infidelity_sample] fidelity:", fidelity)
        infidelity = 1. - (np.abs(np.trace(tot_sqrt)) ** 2)
    return (infidelity, simulator.gate_count)

# @profile
def multi_infidelity_sample(simulator, time, exact_final_state, iterations=1, nbsamples=1, mc_samples=MC_SAMPLES_DEFAULT):
    ret = []
    # No need to sample TrotterSim, just return single element list
    if type(simulator) == TrotterSim:
        ret.append(single_infidelity_sample(simulator, time, exact_final_state, iterations=iterations, nbsamples=nbsamples))
        ret *= mc_samples
    else:
        for samp in range(mc_samples):
            ret.append(single_infidelity_sample(simulator, time, exact_final_state, iterations=iterations, nbsamples=nbsamples))

    return ret

def single_trace_distance_sample(simulator, time, exact_final_state, iterations=1, nbsamples=1):
    if type(simulator) == QDriftSim:
        sim_output = simulator.simulate(time, nbsamples)
    else:
        sim_output = simulator.simulate(time, iterations)
    if simulator.use_density_matrices == False:
        return trace_distance(np.outer(sim_output, np.copy(sim_output).conj().T), np.outer(exact_final_state, np.copy(exact_final_state).conj().T))
    else:
        return trace_distance(sim_output, exact_final_state)
    

def multi_trace_distance_sample(simulator, time, exact_final_state, iterations=1, nbsamples=1, mc_samples=MC_SAMPLES_DEFAULT):
    ret = []
    if type(simulator) == TrotterSim or len(simulator.qdrift_norms) == 0:
        ret.append(single_trace_distance_sample(simulator, time, exact_final_state, iterations=iterations, nbsamples=nbsamples))
        ret *= mc_samples
    else:
        sim_out =  np.zeros(simulator.initial_state.shape, dtype='complex128')
        for _ in range(mc_samples):
            sim_out += simulator.simulate(time, iterations)
        sim_out /= mc_samples
        if simulator.use_density_matrices == False:
            ret =  trace_distance(np.outer(sim_out, np.copy(sim_out).conj().T), np.outer(exact_final_state, np.copy(exact_final_state).conj().T))
        else:
            ret = trace_distance(sim_out, exact_final_state)
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

# Performs an exponential backoff to find an iteration bound (upper or lower)
# Inputs
# infidelity_fn: a callable that returns a list of infidelities to be fed into is_threshold_met
# infidelity_bound: the infidelity_threshold needed for is_threshold_met
# iter_bound: the current guess for the iteration bound
# is_lower_bound: boolean to tell you if we are trying to search for an upper or lower bound
# returns - The first value it finds that satisfies the upper/lower bound invariant
def get_iteration_bound(infidelity_fn, infidelity_bound, iter_bound, is_lower_bound):
    curr_guess = iter_bound
    for i in range(COST_LOOP_DEPTH):
        if is_threshold_met(infidelity_fn(curr_guess), infidelity_bound):
            if is_lower_bound:
                curr_guess = math.ceil(curr_guess / 2)
            else:
                return curr_guess
        else:
            if is_lower_bound:
                return curr_guess
            else:
                curr_guess = int(curr_guess * 2)
    print("[get_iteration_bound] Iteration upper bound exceeded. Proceed with caution.")
    return curr_guess

# Performs exponential backoff to find iteration bounds
# WARNING: With randomized composite channels there is a possibility this does not converge correctly. If you get a "good" sample at
# a heuristic that should be bad then you will search for a lower bound until you exit.  
# Inputs
# - inf_fn: a callable that takes in an iteration and returns a list of infidelities to be fed into is_threshold_met
# - infidelity_threshold: the threshold needed to be fed into is_threshold_met
# - heuristic: the guess as to where a good place to start is
# Returns:
# (iter_lower, iter_upper)
def get_iteration_bounds(inf_fn, infidelity_threshold, heuristic, verbose=False):
    if heuristic < 10:
        if is_threshold_met(inf_fn(10), infidelity_threshold):
            return (0, 10)
        else:
            iter_lower = 5
            iter_upper = -1
    else:
        # This is to help with randomized partitions (aka qdrift heavy)
        heuristic = math.floor(0.8 * heuristic) 
        if is_threshold_met(inf_fn(heuristic), infidelity_threshold):
            iter_lower = -1
            iter_upper = heuristic
        else:
            iter_lower = heuristic
            iter_upper = -1
    curr_guess = max(iter_lower, iter_upper)
    if verbose:
        print("[get_iteration_bounds] Beginning search with iter_lower=", iter_lower, ", iter_upper=", iter_upper, ", curr_guess=",curr_guess)
    for i in range(ITERATION_BOUNDS_LOOP_DEPTH):
        search_for_upper_bound = (iter_lower > 0) and (iter_upper < 0)
        search_for_lower_bound = (iter_lower < 0) and (iter_upper > 0)
        if verbose:
            print("[get_iteration_bounds] iteration: ", i, ", current guess: ", curr_guess)
        if search_for_lower_bound:
            # step = min(100, math.floor(curr_guess / 2.0))
            step = int(curr_guess)
            new_guess = int(curr_guess - step)
            if is_threshold_met(inf_fn(new_guess), infidelity_threshold):
                # lower bound requires the threshold to NOT be met
                curr_guess = new_guess
            else:
                # threshold was NOT met so we have found our iter_lower
                iter_lower = new_guess
                continue

        elif search_for_upper_bound:
            # step = min(100, math.ceil(curr_guess * 2))
            step = math.ceil(curr_guess)
            new_guess = int(curr_guess + step)
            if is_threshold_met(inf_fn(new_guess), infidelity_threshold):
                # upper bound needs to meet threshold so we are good
                iter_upper = new_guess
                continue
            else:
                curr_guess = new_guess
        else:
            return (iter_lower, iter_upper)

    print("[get_iteration_bounds] Iteration depth reached, unclear what to do.")
    raise Exception("get_iteration_bounds")


# Varies the number of iterations needed for a simulator to acheive its infidelity threshold. WARNING: will modify the input
# simulator starting state to a random basis state! Probably need to change this. 
# Inputs
# - simulator: Either Trotter, QDrift, or Composite simulator. If a QDrift simulator is used, then the number
#              of samples is varied. If a Composite simulator is used then iterations is varied while QDrift
#              samples are held fixed.
# - mc_samples: Monte Carlo samples for wave function states, shouldn't be necessary with density matrices?
# Returns (cost, iterations) - a tuple consisting of the gate cost and number of iterations needed to satisfy is_threshold_met 
def find_optimal_cost(simulator, time, infidelity_threshold, heuristic = -1, mc_samples=MC_SAMPLES_DEFAULT, verbose=False):
    hamiltonian_list = simulator.get_hamiltonian_list()
    exact_final_state = simulator.simulate_exact_output(time)

    if verbose:
        print("*" * 75)
        print("[find_optimal_cost] computing cost for simulator with partitioning:")
        simulator.print_partition()
        print("[find_optimal_cost] time, infidelity_threshold: ", time, infidelity_threshold)
    # now we can simplify infidelity to nice lambda, NOTE get_inf_and_cost returns a TUPLE 
    if type(simulator) == TrotterSim or type(simulator) == CompositeSim or type(simulator) == LRsim:
        get_inf_and_cost = lambda x: multi_infidelity_sample(simulator, time, exact_final_state, iterations = x, mc_samples=mc_samples)
    elif type(simulator) == QDriftSim:
        get_inf_and_cost = lambda x: multi_infidelity_sample(simulator, time, exact_final_state, nbsamples= x, mc_samples=mc_samples)
    
    def get_inf(x):
        inf_tup, _ = zip(*get_inf_and_cost(x))
        return list(inf_tup)
    
    iter_lower, iter_upper = get_iteration_bounds(get_inf, infidelity_threshold, heuristic, verbose=verbose)

    if verbose:
        print("[find_optimal_cost] found iteration bounds - lower, upper:", iter_lower, ", ", iter_upper)
    # bisection search until we find it.
    mid = 1
    count = 0
    costs = 0
    while iter_upper - iter_lower  > 1 and count < COST_LOOP_DEPTH:
        count += 1
        mid = (iter_upper + iter_lower) / 2.0
        iters = math.ceil(mid)
        
        if verbose:
            print("[find_optimal_cost] searching midpoint: ", mid)

        inf_tup, costs = zip(*get_inf_and_cost(iters))
        infidelities = list(inf_tup)
        # upper bounds always satisfy the threshold. 
        if is_threshold_met(infidelities, infidelity_threshold):
            iter_upper = iters
        else:
            iter_lower = iters
    if count == COST_LOOP_DEPTH:
        print("[find_optimal_cost] Reached loop depth, results may be inaccurate")
    ret = get_inf_and_cost(iter_upper)
    inf_tup, costs = zip(*ret)
    if verbose:
        print("[find_optimal_cost] computed infidelity avg:", np.mean(list(inf_tup)))
        print("[find_optimal_cost] iters, cost: ", iter_upper, ", ", np.mean(costs))
    return (np.mean(costs), iter_upper)

def crossover_criteria_met(cost1, cost2):
    diff = np.abs(cost1 - cost2)
    avg = np.mean([cost1, cost2])
    if diff / avg < CROSSOVER_CUTOFF_PERCENTAGE:
        return True
    else:
        return False

# Computes the time where the cost between partitions is less than 1% of their difference.
# Inputs:
# simulator - a Composite sim to be partitioned
# partition1 - first partition to evaluate
# partition2 - second partition to evaluate
# time_left - left endpoint for search
# time_right - right endpoint for search
# Returns: either computed time or the best guess. Probably should fix this to indicate the cost difference between the partitions
def find_crossover_time(simulator, partition1, partition2, time_left, time_right, inf_thresh=0.05, verbose=False):
    partition_sim(simulator, partition_type=partition1)
    cost_left_1, _ = find_optimal_cost(simulator, time_left, inf_thresh, verbose=verbose)
    cost_right_1, _ = find_optimal_cost(simulator, time_right, inf_thresh, verbose=verbose)
    partition_sim(simulator, partition_type=partition2)
    cost_left_2, _ = find_optimal_cost(simulator, time_left, inf_thresh, verbose=verbose)
    cost_right_2, _ = find_optimal_cost(simulator, time_right, inf_thresh, verbose=verbose)

    # Tells us if we start on the lower times with partition1 being cheaper than partition2
    start_with_1 = cost_left_1 < cost_left_2
    # Check that they actually cross
    if start_with_1 and (cost_right_1 < cost_right_2):
        print("[find_crossover_time] Partitions do not actually cross!")
        return -1.
    elif (start_with_1 == False) and (cost_right_2 < cost_right_1):
        print("[find_crossover_time] Partitions do not actually cross!")
        return -1.
    
    # Check if either endpoints cross
    if crossover_criteria_met(cost_left_1, cost_left_2):
        return time_left
    if crossover_criteria_met(cost_right_1, cost_right_2):
        return time_right
    
    # Bisection search
    t_lower = time_left
    t_upper = time_right
    t_mid = np.mean([t_lower, t_upper])
    if verbose:
        print("[find_crossover_time] beginning search with:")
        print("t_lower = ", t_upper)
        print("t_upper = ", t_upper)
        print("t_mid = ", t_mid)
        print("start_with_1 = ", start_with_1)
    for _ in range(COST_LOOP_DEPTH):
        if verbose:
            print("[find_crossover_time] evaluating midpoint: ", t_mid)
        t_mid = np.mean([t_upper, t_lower])
        partition_sim(simulator, partition_type=partition1)
        c1, _ = find_optimal_cost(simulator, t_mid, inf_thresh, verbose=verbose)
        partition_sim(simulator, partition_type=partition2)
        c2, _ = find_optimal_cost(simulator, t_mid, inf_thresh, verbose=verbose)
        if crossover_criteria_met(c1, c2):
            return t_mid
        if start_with_1 and (c1 < c2):
            t_lower = t_mid
        elif start_with_1 and (c2 < c1):
            t_upper = t_mid
        elif (start_with_1 == False) and (c1 < c2):
            t_upper = t_mid
        elif (start_with_1 == False) and (c2 < c1):
            t_lower = t_mid
    print("[find_crossover_time] Could not find acceptable crossover within loop bounds. Returning best guess")
    return t_mid

# Return the probabilities for partitioning and the expected cost output of gbrt_minimize
def find_optimal_partition(simulator, time, infidelity_threshold):
    return partition_sim(simulator, "gbrt_prob", time=time, epsilon=infidelity_threshold)

# Computes the expected cost of a probabilistic partitioning scheme. 
def expected_cost(simulator, partition_probs, time, infidelity_threshold, heuristic = -1, num_samples = MC_SAMPLES_DEFAULT):
    print("#" * 75)
    if type(simulator) != CompositeSim:
        print("[expected_cost] Currently only defined for composite simulators.")
        return 1
    
    hamiltonian_list = simulator.get_hamiltonian_list()
    if len(hamiltonian_list) != len(partition_probs):
        print("[expected_cost] Incorrect length probabilities. # of Hamiltonian terms:", len(hamiltonian_list), ", # of probabailities:", len(partition_probs))
        return 1
    costs, iters = [], []
    for _ in range(num_samples):
        start_time = time_this.time()
        trotter, qdrift = sample_probabilistic_partition(hamiltonian_list, partition_probs)
        simulator.set_partition(trotter, qdrift)
        simulator.print_partition()
        if len(iters) > 0:
            prior_iters = iters[-1]
        else:
            prior_iters = -1
        sampled_cost, sampled_iters = find_optimal_cost(simulator, time, infidelity_threshold, heuristic=prior_iters)
        costs.append(sampled_cost)
        iters.append(sampled_iters)
        print("[expected_cost] completed iteration ", _, ", seconds taken: ", time_this.time() - start_time)
    print("[expected_cost] cost avg and std:", np.mean(costs), " +- (", np.std(costs), ")")
    return np.mean(costs)

# Computes and sets a partition
# Inputs:
# - simulator: A composite simulator, not type checked for flexibility later on
# - partition_type: A string describing what partition method to use.
# - weight_threshold: for "chop" partition, determines the spectral norm cutoff for each term to end up in QDrift
# - optimize: for some partitions?
# - nb_scaling: a parametrization of nb within it's lower bound. Follows the scaling (1 + c)^2 * lower_bound. see paper for lower_bound
# - time: required for probabilistic
# - epsilon: required for probabilistic
def partition_sim(simulator, partition_type = "prob", chop_threshold = 0.5, optimize = False, nb_scaling = 0.0, time=0.01, epsilon=0.05):
    if type(partition_type) != type("string"):
        print("[partition_sim] We only accept strings to describe the partition_type")
        return 1
    
    partition_type = partition_type.lower()

    if partition_type == "prob":
        partition_sim_prob(simulator, time, epsilon, nb_scaling, optimize)
    
    elif partition_type == "optimize":
        partition_sim_optimize(simulator)
    
    elif partition_type == "random":
        partition_sim_random(simulator)
    
    elif partition_type == "chop":
        partition_sim_chop(simulator, chop_threshold)
    
    elif partition_type == "optimal_chop":
        partition_sim_optimal_chop(simulator, time, epsilon)

    elif partition_type == "trotter":
        partition_sim_trotter(simulator)
    
    elif partition_type == "first_order_trotter":
        simulator.set_trotter_order(1)
        partition_sim_trotter(simulator)
    
    elif partition_type == "second_order_trotter":
        simulator.set_trotter_order(2)
        partition_sim_trotter(simulator)

    elif partition_type == "qdrift":
        partition_sim_qdrift(simulator)
    
    elif partition_type == "gbrt_prob":
        partition_sim_gbrt_prob(simulator, time, epsilon)
    
    else:
        print("[partition_sim] Did not recieve valid partition. Valid options are: 'prob', 'optimize', 'random', 'chop', 'optimal_chop', 'trotter', and 'qdrift'.")
        return 1

def partition_sim_prob(simulator, time, epsilon, nb_scaling, optimize):
    if simulator.trotter_sim.order > 1:
         k = simulator.trotter_sim.order/2
    else: 
        print("[partition_sim] partition not defined for this order") 
        return 1
    
    upsilon = 2*(5**(k -1))
    lamb = simulator.get_lambda()
    hamiltonian = simulator.get_hamiltonian_list()

    coefficient_1 = (lamb * time / epsilon)**(1 - (1 / (2 * k)))
    coefficient_2 = ((2 * k + 1) / (2 * k + upsilon))**(1 / (2 * k))
    coefficient_3 = 2**(1-(1/k))/ upsilon**(1/(2 * k))
    nb = int( coefficient_1 * coefficient_2 * coefficient_3 *((1 + nb_scaling)**2) )
    simulator.nb = nb

    # below value for chi is based on nb being computed as (1 + c)^2 * lower_bound, which gives chi this nice form
    chi = lamb * nb_scaling / len(hamiltonian)
    probs = [1 - min(1, chi / np.linalg.norm(hamiltonian[ix])) for ix in range(len(hamiltonian))]
    trotter, qdrift = sample_probabilistic_partition(hamiltonian, probs)
    simulator.set_partition(trotter, qdrift)

    # TODO: how to optimize this quantity? not sure what optimal is without computing gate counts?
    # MATT.H - Not really sure if this is optimizing nb or how it's going about it. Need to fix.
    if optimize and False:
        optimal_nb = optimize.minimize(self.prob_nb_optima, self.nb, method='Nelder-Mead', bounds = optimize.Bounds([0], [np.inf], keep_feasible = False)) #Nb attribute serves as an inital geuss in this partition
        nb_high = int(optimal_nb.x +1)
        nb_low = int(optimal_nb.x)
        prob_high = self.prob_nb_optima(nb_high) #check higher, (nb must be int)
        prob_low = self.prob_nb_optima(nb_low) #check lower 
        if prob_high > prob_low:
            self.nb = nb_low
        else:
            self.nb = nb_high
    return 0

def partition_sim_optimize(simulator):
    if simulator.nb_optimizer == True: #if Nb is a parameter we wish to numerically optimize
        guess = [0.5 for x in range(len(simulator.spectral_norms))] #guess for the weights 
        guess.append(2) #initial guess for Nb
        upper_bound = [1 for x in range(len(simulator.spectral_norms))]
        upper_bound.append(20) #no upper bound for Nb but set to some number we can compute instead of np.inf
        lower_bound = [0 for x in range(len(simulator.spectral_norms) + 1)]  #lower bound for Nb is 0
        optimized_weights = optimize.minimize(simulator.nb_first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(lower_bound, upper_bound))
        print(optimized_weights.x)
        for i in range(len(simulator.spectral_norms)):
            if optimized_weights.x[i] >= weight_threshold:
                simulator.a_norms.append([i, simulator.spectral_norms[i]])
            elif optimized_weights.x[i] < weight_threshold:
                simulator.b_norms.append([i, simulator.spectral_norms[i]])
        simulator.a_norms = np.array(simulator.a_norms, dtype='complex')
        simulator.b_norms = np.array(simulator.b_norms, dtype='complex')
        simulator.nb = int(optimized_weights.x[-1] + 1) #nb must be of type int so take the ceiling 
        return 0

    if simulator.nb_optimizer == False: #same as above leaving Nb as user-defined
        guess = [0.5 for x in range(len(simulator.spectral_norms))] #guess for the weights 
        upper_bound = [1 for x in range(len(simulator.spectral_norms))]
        lower_bound = [0 for x in range(len(simulator.spectral_norms))]  
        optimized_weights = optimize.minimize(simulator.first_order_cost, guess, method='Nelder-Mead', bounds=optimize.Bounds(lower_bound, upper_bound)) 
        print(optimized_weights.x)
        for i in range(len(simulator.spectral_norms)):
            if optimized_weights.x[i] >= weight_threshold:
                simulator.a_norms.append([i, simulator.spectral_norms[i]])
            elif optimized_weights.x[i] < weight_threshold:
                simulator.b_norms.append([i, simulator.spectral_norms[i]])
        simulator.a_norms = np.array(simulator.a_norms, dtype='complex')
        simulator.b_norms = np.array(simulator.b_norms, dtype='complex')
        return 0

def partition_sim_random(simulator):
    hamiltonian = simulator.get_hamiltonian_list()
    trotter, qdrift = sample_probabilistic_partition(hamiltonian, [0.5] * len(hamiltonian))
    simulator.set_partition(trotter, qdrift)
    return 0

def partition_sim_chop(simulator, weight_threshold):
    hamiltonian = simulator.get_hamiltonian_list()
    trotter, qdrift = [], []
    for ix in range(len(hamiltonian)):
        norm = np.linalg.norm(hamiltonian[ix], ord=2)
        if norm >= weight_threshold:
            trotter.append(hamiltonian[ix])
        else:
            qdrift.append(hamiltonian[ix])
    simulator.set_partition(trotter, qdrift)
    return 0

def partition_sim_optimal_chop(simulator, time, epsilon):
    dimensions = [Real(name='weight', low = 0, high = max(simulator.spectral_norms))]
    @use_named_args(dimensions=dimensions)
    def obj_fn(weight):
        partition_sim_chop(simulator, weight)
        return find_optimal_cost(simulator, time, epsilon)[0] # [0] gets the costs throws away iters
    result = gbrt_minimize(obj_fn, dimensions=dimensions, n_calls=30, n_initial_points=5, random_state=4, verbose=True, acq_func="LCB")
    print("result.fun: ", result.fun)
    print("result.x: ", result.x)
    print("result:", result)

# Let boosted regression trees try their best to come up with good probabilities
# Inputs: self-explanatory
# Returns: (probability list, nb, expected cost at optimal)
def partition_sim_gbrt_prob(simulator, time, epsilon):
    hamiltonian = simulator.get_hamiltonian_list()
    dimensions = [Real(0.0, 1.0)] * len(hamiltonian)
    dimensions += [Integer(1, len(hamiltonian))]

    def obj_fn(dims):
        probs = dims[:-1]
        nb = dims[-1]
        simulator.nb = nb
        return expected_cost(simulator, probs, time, epsilon)
    
    result = gbrt_minimize(obj_fn, dimensions=dimensions, x0=[1.0]*len(hamiltonian) + [1], n_calls=20, verbose=True, acq_func="LCB", n_jobs=-1)
    print("results:")
    print("fun:", result.fun)
    print("x:", result.x)
    return (result.x[:-1], result.x[-1], result.fun)

def partition_sim_trotter(simulator):
    ham = simulator.get_hamiltonian_list()
    simulator.set_partition(ham, [])
    return 0

def partition_sim_qdrift(simulator):
    ham = simulator.get_hamiltonian_list()
    simulator.set_partition([], ham)
    return 0

# Inputs
# - hamiltonian_list: list of terms to be partitioned
# - prob_list: list of probabilities to sample a partition. Standard we are using is prob = 1.0 means Trotter, 0.0 means QDrift
# Returns a tuple of lists (trotter, qdrift) to be used in a partitioner.
# TODO- refactor existing code to use this function.
def sample_probabilistic_partition(hamiltonian_list, prob_list):
    if len(hamiltonian_list) != len(prob_list):
        print("[sample_probabilistic_partition] lengths of lists do not match")
        return 1
    trotter, qdrift = [], []
    for ix in range(len(hamiltonian_list)):
        if prob_list[ix] > 1.0 or prob_list[ix] < 0.0:
            print("[sample_probabilistic_partition] probability not within [0.0, 1.0] encountered")
            return 1
        sample = np.random.random()
        if sample <= prob_list[ix]:
            trotter.append(hamiltonian_list[ix])
        else:
            qdrift.append(hamiltonian_list[ix])
    return (trotter, qdrift)


####  FIRST ORDER COST FUNCTIONS FROM SIMULATOR
# These turn out to not modify the simulator directly/are necessary for simulator functioning so moved here. Currently not used so they are prunable.
def nb_first_order_cost(simulator, weight): #first order cost, currently computes equation 31 from paper. Weight is a list of all weights with Nb in the last entry
    cost = 0.0                         #Error with this function, it may not be possible to optimize Nb with this structure given the expression of the function
    qd_sum = 0.0
    for i in range(len(simulator.spectral_norms)):
        qd_sum += (1-weight[i]) * simulator.spectral_norms[i]
        for j in range(len(simulator.spectral_norms)):
            commutator_norm = np.linalg.norm(np.matmul(simulator.hamiltonian_list[i], simulator.hamiltonian_list[j]) - np.matmul(simulator.hamiltonian_list[j], simulator.hamiltonian_list[i]), ord = 2)
            cost += (2/(5**(1/2))) * ((weight[i] * weight[j] * simulator.spectral_norms[i] * simulator.spectral_norms[j] * commutator_norm) + 
                (weight[i] * (1-weight[j]) * simulator.spectral_norms[i] * simulator.spectral_norms[j] * commutator_norm))
    cost += (qd_sum**2) * 4/weight[-1] #dividing by Nb at the end (this form is just being used so I can easily optimize Nb as well)
    return cost

def first_order_cost(simulator, weight): #first order cost, currently computes equation 31 from paper. Function does not have nb as an omptimizable parameter
    cost = 0.0
    qd_sum = 0.0
    for i in range(len(simulator.spectral_norms)):
        qd_sum += (1-weight[i]) * simulator.spectral_norms[i]
        for j in range(len(simulator.spectral_norms)):
            commutator_norm = np.linalg.norm(np.matmul(simulator.hamiltonian_list[i], simulator.hamiltonian_list[j]) - np.matmul(simulator.hamiltonian_list[j], simulator.hamiltonian_list[i]), ord = 2)
            cost += (2/(5**(1/2))) * ((weight[i] * weight[j] * simulator.spectral_norms[i] * simulator.spectral_norms[j] * commutator_norm) + 
                (weight[i] * (1-weight[j]) * simulator.spectral_norms[i] * simulator.spectral_norms[j] * commutator_norm))
    cost += (qd_sum**2) * 4/simulator.nb #dividing by Nb at the end (this form is just being used so I can easily optimize Nb as well)
    return cost

#Function that allows for the optimization of the nb parameter in the probabilistic partitioning scheme (at each timestep)
def prob_nb_optima(simulator, test_nb):
    k = simulator.inner_order/2
    upsilon = 2*(5**(k -1))
    lamb = sum(simulator.spectral_norms)
    test_chi = (lamb/len(simulator.spectral_norms)) * ((test_nb * (simulator.epsilon/(lamb * simulator.time))**(1-(1/(2*k))) * 
    ((2*k + upsilon)/(2*k +1))**(1/(2*k)) * (upsilon**(1/(2*k)) / 2**(1-(1/k))))**(1/2) - 1) 
        
    test_probs = []
    for i in range(len(simulator.spectral_norms)):
        test_probs.append(float(np.abs((1/simulator.spectral_norms[i])*test_chi))) #ISSUE
    return max(test_probs)

#a function to decide on a good number of monte carlo samples for the following simulation functions (avoid noise errors)
def sample_decider(simulator, time, samples, iterations, mc_sample_guess, epsilon):
    exact_state = exact_time_evolution(simulator.unparsed_hamiltonian, time, simulator.initial_state)
    sample_guess = mc_sample_guess
    inf_samples = [1, 0]
    for k in range(1, 25):
        inf_samples[k%2] = simulator.sample_channel_inf(time, samples, iterations, sample_guess, exact_state)
        print(inf_samples)
        if np.abs((inf_samples[0] - inf_samples[1])) < (0.1 * epsilon): #choice of precision
            break
        else:
            sample_guess *= 2
    return int(sample_guess/2)

def hamiltonian_localizer_1d(local_hamiltonian, sub_block_size, sub_blocks = 1):
    #A function to do an m=1 block decomposition of a nearest neighbour hamiltonian. Takes a tupel as input where 
    # indices 0, 1, 2 are the Hamiltonian terms, 1d lattice indices, and legnth of the lattice respectfully.
    #The funciton outputs 3 new "local" lists that can then be partitioned
    a_terms, a_index = [], []
    y_terms, y_index = [], []
    b_terms, b_index = [], []
    terms, indices, length = local_hamiltonian
    midpoint = int(length/2) 
    start = midpoint - (int(sub_block_size/2))
    stop = start + sub_block_size 
    if start < 1:
        raise Exception("sub block is the size of or larger than the Hamiltonian")
    
    ix = 0
    while ix < len(terms): 
        if not set(indices[ix]).isdisjoint(arange(start, stop)) == True: #Y region
            y_terms.append(terms[ix].conj().T)
            y_index.append(indices[ix])

        if not set(indices[ix]).isdisjoint(arange(start, len(terms))) == True: #B region
            b_terms.append(terms[ix])
            b_index.append(indices[ix])

        if not set(indices[ix]).isdisjoint(arange(0, stop)): #A region
            a_terms.append(terms[ix])
            a_index.append(indices[ix])
        ix += 1
    if ((len(a_terms) == 0) or (len(b_terms) == 0) or (len(y_terms) == 0)):
        raise Exception("poor block choice, one of the blocks is empty")
    return (a_terms, y_terms, b_terms)

def local_partition(simulator, partition, weights = None, time = 0.01, epsilon = 0.001): #weights is a list with ordering A, Y, B
        if type(simulator) != LRsim: raise TypeError("only works on LRsims")
        if partition == "chop":
            local_chop(simulator, weights)
        if partition == "optimal_chop":
            optimal_local_chop(simulator, time, epsilon)
        else:
            raise Exception("this is not a valid partition")
        return 0

def local_chop(simulator, weights):
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    for i in range(len(simulator.spectral_norms)):
        a_temp = []
        b_temp = []
        for j in range(len(simulator.spectral_norms[i])):
            if simulator.spectral_norms[i][j] >= weights[i]:
                a_temp.append(simulator.spectral_norms[i][j])
            else:
                b_temp.append(simulator.spectral_norms[i][j])
            
        simulator.internal_sims[i].set_partition(a_temp, b_temp)
    return 0

def optimal_local_chop(simulator, time, epsilon): ### needs exact cost function to be operational
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    w_guess = [] 
    nb_guess = []
    dimensions = []
    dim1 = []
    dim2 = []
    for i in range(len(simulator.spectral_norms)):
        w_guess.append(statistics.median(simulator.spectral_norms[i]))
        nb_guess.append(int(len(simulator.spectral_norms[i])))

        dim1.append(Real(name = "weight_"+str(i), low=0, high = max(simulator.spectral_norms)))
        dim2.append(Integer(name="nb_"+str(i), low=1, high = len(simulator.spectral_norms * 10)))

    guess_points = [w_guess, nb_guess]
    dimensions = [dim1, dim2]

    @use_named_args(dimensions=dimensions)
    def obj_func():
        set_local_nb(simulator, nb_list)
        local_chop(simulator, time, epsilon)
        return exact_cost(simulator, time, epsilon)
    
    gbrt_minimize(func=obj_func,dimensions=dimensions, n_calls=20, n_initial_points = 5, 
                random_state=4, verbose = False, acq_func = "LCB", x0 = guess_points)
    return 0

def exact_cost(simulator, time, epsilon): #relies on the use of density matrices
    if type(simulator) == (CompositeSim): 
        if simulator.partition == "qdrift":
            get_trace_dist = lambda x : sim_trace_distance(time, iterations=1, nb = x)
        elif simulator.partition == 'trotter':
            get_trace_dist = lambda x : sim_trace_distance(time, iterations=x, nb = 1)
        else: 
            get_trace_dist = lambda x : sim_trace_distance(time, iterations=x, nb = simulator.nb)
    if type(simulator) == LRsim:
        get_trace_dist = lambda x: sim_trace_distance(time, iterations=x, nb = simulator.nb)
    else: raise TypeError("only works on LR and Composite Sims")
        
    lower_bound = 1
    upper_bound = 2
    trace_dist = get_trace_dist(lower_bound)
    if trace_dist < epsilon:
        print("[sim_channel_performance] Iterations too large, already below error threshold")
        return simulator.gate_count
    # Iterate up until some max cutoff
    break_flag = False
    for n in range(27):
        trace_dist = get_trace_dist(upper_bound) 
        if trace_dist < epsilon:
            break_flag = True
            break
        else:
            upper_bound *= 2
            #print(trace_dist, self.gate_count)
    if break_flag == False:
        raise Exception("[sim_channel_performance] maximum number of iterations hit, something is probably off")
    #print("the upper bound is " + str(upper_bound))

    if (upper_bound == 2):
        return simulator.gate_count
    #Binary search
    break_flag_2 = False
    while lower_bound < upper_bound:
        mid = lower_bound + (upper_bound - lower_bound)//2
        if (mid == 2) or (mid ==1): 
            return simulator.gate_count #catching another edge case
        if (get_trace_dist(mid +1) < epsilon) and (get_trace_dist(mid-1) > epsilon): #Causing Problems
            break_flag_2 = True
            break #calling the critical point the point where the second point on either side goes from a bad point to a good point (we are in the neighbourhood of the ideal gate count)
        elif get_trace_dist(mid) < epsilon:
            upper_bound = mid - 1
        else:
            lower_bound = mid + 1
    if break_flag_2 == False:
        print("[sim_channel_performance] function did not find a good point")

    get_trace_dist(mid)
    return simulator.gate_count

def sim_trace_distance(simulator, time, iterations, nb): #put nb in here and unparsed
    if type(simulator) == TrotterSim:
        return trace_distance(simulator.simulate(time, iterations), exact_time_evolution(simulator.hamiltonian_list, 
                            time, simulator.initial_state))

    elif type(simulator) == LRsim:
        if type(nb) != list: raise TypeError("local sims require an nb value per site")
        set_local_nb(simulator, nb)
        return trace_distance(simulator.simulate(time, iterations), exact_time_evolution(simulator.hamiltonian_list, 
                            time, simulator.initial_state))

    else: #TODO incorporate samples into this framework -- just call composite sim and set the partition and nb :)
        return trace_distance(simulator.simulate(time, iterations), exact_time_evolution(simulator.hamiltonian_list, 
                            time, simulator.initial_state))
 
def set_local_nb(simulator, nb_list):
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    for i in simulator.internal_sims:
        simulator.internal_sims[i].nb = nb_list[i]


def test():
    hamiltonian = graph_hamiltonian(2, 1, 1)
    t = 0.01
    eps = 0.1
    compsim = CompositeSim(hamiltonian_list=hamiltonian, inner_order=2)
    trottsim = TrotterSim(hamiltonian_list=hamiltonian)
    qsim = QDriftSim(hamiltonian_list=hamiltonian)
    # partition_sim(compsim, "prob", nb_scaling=.01)
    # compsim.print_partition()
    # partition_sim(compsim, "prob", nb_scaling=.05)
    # compsim.print_partition()
    # partition_sim(compsim, "prob", nb_scaling=.1)
    # compsim.print_partition()
    # partition_sim(compsim, "prob", nb_scaling=.5)
    # compsim.print_partition()
    # partition_sim(compsim, "prob", nb_scaling=0.05)
    # compsim.reset_init_state()
    # print("Check to see if the hamiltonian has remained the same after many partitionings.")
    # hamiltonian_copy = compsim.get_hamiltonian_list()
    # print("Difference norm:", np.linalg.norm(sum(hamiltonian) - sum(hamiltonian_copy)))
    # compsim.print_partition()
    # print("testing trotter")
    # print(find_optimal_cost(trottsim, 1, .1))
    # print(find_optimal_cost(compsim, 1, .1))
    # expected_cost(compsim, np.random.random(len(hamiltonian)), 0.01, 0.1, num_samples=100)
    partition_sim_gbrt_prob(compsim, 1, 0.05)
    # print("Now testing optimal chop")
    # partition_sim(compsim, "optimal_chop")

if False:
    test()
