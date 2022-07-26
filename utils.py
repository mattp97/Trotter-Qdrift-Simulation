from ast import And
from asyncore import loop
from mimetypes import init
from operator import matmul
from telnetlib import AYT
from cirq import sample
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
from sympy import S, symbols, printing
from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import cProfile, pstats, io
from compilers import CompositeSim, TrotterSim, QDriftSim, DensityMatrixSim, profile, conditional_decorator

MC_SAMPLES_DEFAULT = 10
COST_LOOP_DEPTH = 30
ITERATION_UPPER_BOUND_MAX = 20 # specifies power of 2 for maximum number of iterations to search through

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
# @profile
def single_infidelity_sample(simulator, time, iterations = 1, nbsamples = 1):
    sim_output = []
    exact_output = simulator.simulate_exact_output(time)

    if type(simulator) == QDriftSim:
        sim_output = simulator.simulate(time, nbsamples)
    
    if type(simulator) == TrotterSim:
        sim_output = simulator.simulate(time, iterations)
    
    if type(simulator) == CompositeSim:
        sim_output = simulator.simulate(time, iterations)

    infidelity = 1 - (np.abs(np.dot(exact_output.conj().T, sim_output)).flat[0])**2
    return (infidelity, simulator.gate_count)

@profile
def multi_infidelity_sample(simulator, time, iterations=1, nbsamples=1, mc_samples=MC_SAMPLES_DEFAULT):
    ret = []

    # No need to sample TrotterSim, just return single element list
    if type(simulator) == TrotterSim:
        ret.append(single_infidelity_sample(simulator, time, iterations=iterations, nbsamples=nbsamples))
        ret *= mc_samples
    else:
        for samp in range(mc_samples):
            ret.append(single_infidelity_sample(simulator, time, iterations=iterations, nbsamples=nbsamples))

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
    for i in range(ITERATION_UPPER_BOUND_MAX):
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
    
    # choose random basis state to start out with
    init = 0 * simulator.initial_state
    init[np.random.randint(0, len(init))] = 1.
    simulator.set_initial_state(init)
    exact_final_state = exact_time_evolution(hamiltonian_list, time, initial_state=init)

    # now we can simplify infidelity to nice lambda, NOTE get_inf_and_cost returns a TUPLE 
    if type(simulator) == TrotterSim or type(simulator) == CompositeSim:
        get_inf_and_cost = lambda x: multi_infidelity_sample(simulator, time, iterations = x, mc_samples=mc_samples)
    elif type(simulator) == QDriftSim:
        get_inf_and_cost = lambda x: multi_infidelity_sample(simulator, time, nbsamples= x, mc_samples=mc_samples)
    
    def get_inf(x):
        inf_tup, _ = zip(*get_inf_and_cost(x))
        return list(inf_tup)
    
    # Branch if we have a heuristic for where the optimal iterations might be. If we do then we need to see if it currently
    # gives an uppper or lower bound, then use a 20% deviation from the heuristic for the corresponding opposite bound.
    # If we do not have a heuristic use an exponential backoff to provide bounds.
    # TODO: this is ~kind of~ sloppy, could probably convert the helper function into just "compute_iteration_bounds" and have
    # it handle all of this logic.
    iter_lower = 1
    iter_upper = 2 ** 20
    if heuristic > 0:
        inf_tup, costs = zip(*get_inf_and_cost(heuristic))
        if is_threshold_met(list(inf_tup), infidelity_threshold):
            if heuristic == 1:
                # This means our heuristic satisfies the threshold and is 1, which is the lowest possible iterations. Return early.
                return (np.mean(costs), heuristic)
            iter_upper = heuristic
            iter_lower = get_iteration_bound(get_inf, infidelity_threshold, heuristic, True)
        else:
            iter_upper = get_iteration_bound(get_inf, infidelity_threshold, heuristic, False)
            iter_lower = heuristic
    else:
        iter_upper = get_iteration_bound(get_inf, infidelity_threshold, 1, False)
        # NOTE: this lower bound should work because get_iteration_bound doubles until it finds an upper so half of this should be a lower.
        iter_lower = math.floor(iter_upper / 2) 

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
    return (np.mean(costs), iter_upper)

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
def partition_sim_gbrt_prob(simulator, time, epsilon):
    hamiltonian = simulator.get_hamiltonian_list()
    dimensions = [Real(0.0, 1.0)] * len(hamiltonian)
    dimensions += [Integer(1, len(hamiltonian))]

    def obj_fn(dims):
        probs = dims[:-1]
        nb = dims[-1]
        simulator.nb = nb
        return expected_cost(simulator, probs, time, epsilon)
    
    result = gbrt_minimize(obj_fn, dimensions=dimensions, x0=[1.0]*len(hamiltonian) + [1], n_calls=20, verbose=True, acq_func="LCB")
    print("results:")
    print("fun:", result.fun)
    print("x:", result.x)

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
            y_terms.append(-1 * terms[ix])
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

def lieb_robinson_sim(localized_hamiltonian, in_trotter_order, partition):
    for i in localized_hamiltonian:
        block_sim = CompositeSim(i, inner_order=in_trotter_order)
        partition_sim(simulator=block_sim, partition_type = partition) #idea to generate each as a sim an keep track of the current state and total gate count
    return None

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