# from telnetlib import AYT
import numpy as np
import statistics
from scipy import linalg
from scipy import optimize
import math
from numpy import arange
import time as time_this
import scipy
import json
import os
from skopt import gbrt_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from compilers import CompositeSim, TrotterSim, QDriftSim, LRsim

MC_SAMPLES_DEFAULT = 100
COST_LOOP_DEPTH = 30
ITERATION_BOUNDS_LOOP_DEPTH = 100 # specifies power of 2 for maximum number of iterations to search through
CROSSOVER_CUTOFF_PERCENTAGE = 0.5
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

#A function to calculate the trace distance between two numpy arrays (density operators)
def trace_distance(rho, sigma):
    if (rho.shape[0] != rho.shape[1]) or (rho.shape != sigma.shape):
        print("[trace_distance] Improper shapes were given:", rho.shape, sigma.shape)
        raise Exception("Incompatible shapes for Trace Distance.")
    diff = rho - sigma
    tot = scipy.linalg.sqrtm(diff @ np.copy(diff).conj().T)
    return 0.5 * np.abs(np.trace(tot)) # Note: absolute value after trace is because we have 'complex' variables, so taking the norm should be fine??

def infidelity(rho, sigma):
    # Density matrices
    if (rho.shape[0] == rho.shape[1]) and (rho.shape == sigma.shape):
        exact_sqrt = scipy.linalg.sqrtm(rho)
        tot_sqrt = scipy.linalg.sqrtm(np.linalg.multi_dot([exact_sqrt, sigma, np.copy(exact_sqrt)]))
        fidelity = np.abs(np.trace(tot_sqrt))
        # print("[single_infidelity_sample] fidelity:", fidelity)
        return 1. - fidelity ** 2
    elif (rho.shape[1] == 1 or rho.shape[0] == 1) and (rho.shape == sigma.shape):
        return 1 - (np.abs(np.dot(rho.conj().T, sigma)).flat[0])**2
    else:
        print("[infidelity] Could not parse shapes:", rho.shape, sigma.shape)

def exact_time_evolution_density(hamiltonian_list, time, initial_rho):
    '''
    Computes the exact time evolution of a density matrix given a Hamiltonian in real time.
    '''
    if len(hamiltonian_list) == 0:
        print("[exact_time_evolution] pls give me hamiltonian")
        return 1
    exp_op = linalg.expm(1j * sum(hamiltonian_list) * time)
    return exp_op @ initial_rho @ exp_op.conj().T

# Inputs are self explanatory except simulator which can be any of 
# TrotterSim, QDriftSim, CompositeSim
# Outputs: a single shot estimate of the infidelity according to the exact output provided. 
def single_infidelity_sample(simulator, time, exact_final_state, iterations = 1, nbsamples = 1):
    sim_output = []
    # exact_output = simulator.simulate_exact_output(time)

    if type(simulator) == QDriftSim:
        sim_output = simulator.simulate(time, nbsamples)
    
    if type(simulator) == TrotterSim:
        sim_output = simulator.simulate(time, iterations)
    
    if type(simulator) == CompositeSim:
        simulator.nb = nbsamples #Fixed
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
        # print("[single_infidelity_sample] fidelity:", fidelity)
        infidelity = 1. - (np.abs(np.trace(tot_sqrt)) ** 2)
    return (infidelity, simulator.gate_count)

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
        dist = trace_distance(np.outer(sim_output, np.copy(sim_output).conj().T), np.outer(exact_final_state, np.copy(exact_final_state).conj().T))
        return (dist, simulator.gate_count)
    else:
        return (trace_distance(sim_output, exact_final_state), simulator.gate_count)
    

def multi_trace_distance_sample(simulator, time, exact_final_state, iterations=1, nbsamples=1, mc_samples=MC_SAMPLES_DEFAULT):
    if type(simulator) == TrotterSim or len(simulator.qdrift_norms) == 0:
        ret = single_trace_distance_sample(simulator, time, exact_final_state, iterations=iterations, nbsamples=nbsamples)
    else:
        # sim_out =  np.zeros(simulator.initial_state.shape, dtype='complex128')
        # for _ in range(mc_samples):
        #     sim_out += simulator.simulate(time, iterations)
        # sim_out /= mc_samples
        if type(simulator) != CompositeSim: raise Exception("Error please use composite simulators")
        simulator.set_exact_qd(True)
        sim_out = simulator.simulate(time, iterations)
        
        if simulator.use_density_matrices == False:
            dist =  trace_distance(np.outer(sim_out, np.copy(sim_out).conj().T), np.outer(exact_final_state, np.copy(exact_final_state).conj().T))
        else:
            dist = trace_distance(sim_out, exact_final_state)
        ret = (dist, simulator.gate_count)
    return ret

def exact_time_evolution(sim, time): 
    '''
    Given a simulator, compute the exact time evolution of the Hamiltonian (no channel approximation algorithms) in real or imaginary time depending on `imag_time` attribute.
    '''
    #framework did not accept a simulator previously, this may break something somewhere in the state_vec sampling pic
    if sim.use_density_matrices == True:
        if sim.imag_time == False:
            return linalg.expm(1j * sum(sim.unparsed_hamiltonian) * time) @ sim.initial_state @ linalg.expm(1j * sum(sim.unparsed_hamiltonian) * time).conj().T
        else:
            un_normed_state = linalg.expm(-1 * sum(sim.unparsed_hamiltonian) * time) @ sim.initial_state @ linalg.expm(-1 * sum(sim.unparsed_hamiltonian) * time)
            return un_normed_state / np.trace(un_normed_state)
    else:
        if sim.imag_time == False:
            return linalg.expm(1j * sum(sim.unparsed_hamiltonian) * time) @ sim.initial_state
        else:
            un_normed_state = linalg.expm(-1 * sum(sim.unparsed_hamiltonian) * time) @ sim.initial_state
            return un_normed_state / np.linalg.norm(un_normed_state)


def get_iteration_bounds(is_iteration_good, heuristic, verbose=False):
    """
    Performs exponential backoff to find iteration bounds for use in gate cost estimators.
    WARNING: With randomized composite channels there is a possibility this does not converge correctly. If you get a "good" sample at
    a heuristic that should be bad then you will search for a lower bound until you exit.\n
    Inputs
    - is_iteration_good: a callable that takes in an iteration and returns a boolean if threshold is met
    - heuristic: the guess as to where a good place to start is
    \n
    Outputs:\n
    (iteration lower bound, iteration upper bound)
    """
    if heuristic < 10:
        if is_iteration_good(10):
            return (0, 10)
        else:
            iter_lower = 5
            iter_upper = -1
    else:
        # This is to help with randomized partitions (aka qdrift heavy)
        heuristic = math.floor(0.8 * heuristic) 
        if is_iteration_good(heuristic):
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
            if is_iteration_good(new_guess):
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
            if is_iteration_good(new_guess):
                # upper bound needs to meet threshold so we are good
                iter_upper = new_guess
                continue
            else:
                curr_guess = new_guess
        else:
            return (iter_lower, iter_upper)

    print("[get_iteration_bounds] Iteration depth reached, unclear what to do.")
    raise Exception("get_iteration_bounds")



def find_optimal_cost(simulator, time, epsilon, heuristic = -1, use_infidelity = False, mc_samples=MC_SAMPLES_DEFAULT, num_state_samples=10, verbose=False):
    """
    Varies the number of iterations needed for a simulator to acheive its infidelity threshold. 
    # WARNING
    will modify the input simulator starting state to a random basis state! Probably need to change this. 
    ## Inputs
    - simulator: Either Trotter, QDrift, or Composite simulator. If a QDrift simulator is used, then the number
                 of samples is varied. If a Composite simulator is used then iterations is varied while QDrift
                 samples are held fixed.
    - mc_samples: Monte Carlo samples for wave function states, shouldn't be necessary with density matrices?
    ## Returns
    - (gate_cost, iterations) a tuple consisting of the gate cost and number of iterations needed to satisfy the epsilon
    """
    if verbose:
        print("*" * 75)
        print("[find_optimal_cost] computing cost for simulator with partitioning:")
        simulator.print_partition()
        print("[find_optimal_cost] time = ", time)
        print("epsilon =", epsilon)
        print("use_infidelity=", use_infidelity)
        print("mc_samples=", mc_samples)

    # Make sure that density matrices are turned on if we use trace distance
    if use_infidelity == False:
        simulator.set_density_matrices(True)

    # Helper function to average over random initial states and perform monte carlo averaging for infidelity.
    def get_err_avg_std_cost(iterations):
        inf_avg_tot, inf_std_tot, cost_tot = 0, 0, 0
        for _ in range(num_state_samples):
            simulator.randomize_initial_state()
            exact_final_state = simulator.exact_final_state(time)
            if use_infidelity:
                if type(simulator) == QDriftSim:
                    infs, costs = zip(*multi_infidelity_sample(simulator, time, exact_final_state, nbsamples=iterations, mc_samples=mc_samples))
                else:
                    infs, costs = zip(*multi_infidelity_sample(simulator, time, exact_final_state, iterations=iterations, mc_samples=mc_samples))
            else:
                if type(simulator) == QDriftSim:
                    infs, costs = multi_trace_distance_sample(simulator, time, exact_final_state, nbsamples=iterations, mc_samples=mc_samples)
                else:
                    infs, costs = multi_trace_distance_sample(simulator, time, exact_final_state, iterations=iterations, mc_samples=mc_samples)
            inf_avg_tot += np.mean(infs)
            inf_std_tot += np.std(infs)
            cost_tot += np.mean(costs)
        return (inf_avg_tot / num_state_samples, inf_std_tot / num_state_samples, cost_tot / num_state_samples)
            
    def is_iteration_good(iter):
        avg, std, _ = get_err_avg_std_cost(iter)
        return epsilon > avg + 2 * std
    
    iter_lower, iter_upper = get_iteration_bounds(is_iteration_good, heuristic, verbose=verbose)

    if verbose:
        print("[find_optimal_cost] found iteration bounds: lower = ", iter_lower, ", upper =", iter_upper)
    # bisection search until we find it.
    mid = 1
    count = 0
    costs = 0
    while iter_upper - iter_lower  > 1 and count < COST_LOOP_DEPTH:
        count += 1
        mid = (iter_upper + iter_lower) / 2.0
        iters = math.ceil(mid)
        
        if verbose:
            print("[find_optimal_cost] searching midpoint: ", iters)

        # upper bounds always satisfy the threshold.
        if is_iteration_good(iters):
            iter_upper = iters
        else:
            iter_lower = iters
    if count == COST_LOOP_DEPTH:
        print("[find_optimal_cost] Reached loop depth, results may be inaccurate")
    ret = get_err_avg_std_cost(iter_upper) # a tuple of inf_mean, inf_std, and cost
    if verbose:
        print("[find_optimal_cost] converged to iterations = ", iter_upper)
        print("[find_optimal_cost] final infidelity =", ret[0], " +- ", ret[1])
        print("[find_optimal_cost] final gate cost: ", ret[-1])
        # Write to intermediate file.
        json_path = os.getenv("SCRATCH")
        if json_path[-1] != '/':
            json_path += '/'
        json_path += "outputs/gate_cost_" + str(len(simulator.trotter_norms)) + "_" + str(len(simulator.qdrift_norms)) + "_" + str(time) + ".json"
        try:
            r = {}
            r["time"] = time
            r["cost"] = ret[-1]
            r["iters"] = iter_upper
            r["avg_err"] = ret[0]
            json.dump(r, open(json_path, 'w'))
        except:
            print("[find_crossover_time] tried to dump json it didn't work")
            print("file name was:", json_path)

    return (ret[-1], iter_upper)

def crossover_criteria_met(cost1, cost2):
    diff = np.abs(cost1 - cost2)
    avg = np.mean([cost1, cost2])
    if diff / avg < CROSSOVER_CUTOFF_PERCENTAGE:
        return True
    else:
        return False

def find_crossover_time(simulator, partition1, partition2, time_left, time_right, epsilon=0.05, use_infidelity=True, verbose=False, mc_samples=MC_SAMPLES_DEFAULT):
    """
    ## Computes the time where the cost between partitions is less than 5% of their difference.
    ### Inputs:
    - simulator - a Composite sim to be partitioned
    - partition1 - first partition to evaluate
    - partition2 - second partition to evaluate
    - time_left - left endpoint for search
    - time_right - right endpoint for search
    ### Returns:
    - either computed time or the best guess. Probably should fix this to indicate the cost difference between the partitions
    """
    partition_sim(simulator, partition_type=partition1)
    cost_left_1, _ = find_optimal_cost(simulator, time_left, epsilon, verbose=verbose, mc_samples=mc_samples)
    cost_right_1, _ = find_optimal_cost(simulator, time_right, epsilon, verbose=verbose, mc_samples=mc_samples)
    partition_sim(simulator, partition_type=partition2)
    cost_left_2, _ = find_optimal_cost(simulator, time_left, epsilon, verbose=verbose, mc_samples=mc_samples)
    cost_right_2, _ = find_optimal_cost(simulator, time_right, epsilon, verbose=verbose, mc_samples=mc_samples)

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
        print("t_lower = ", t_lower)
        print("t_upper = ", t_upper)
        print("t_mid = ", t_mid)
        print("start_with_1 = ", start_with_1, flush=True)
        json_path = os.getenv("SCRATCH")
        if json_path[-1] != '/':
            json_path += '/'
        json_path += "partial_result.json"
        try:
            r = {}
            r["t_lower"] = t_lower
            r["t_upper"] = t_upper
            json.dump(r, open(json_path, 'w'))
        except:
            print("[find_crossover_time] tried to dump json it didn't work")
            print("file name was:", json_path)

    for _ in range(COST_LOOP_DEPTH):
        if verbose:
            print("[find_crossover_time] evaluating midpoint: ", t_mid, flush=True)
            try:
                json_path = os.getenv("SCRATCH")
                if json_path[-1] != '/':
                    json_path += '/'
                json_path += "partial_result.json"
                r = {}
                r["midpoint_" + str(_ + 1)] = t_mid
                curr = json.load(open(json_path, 'r'))
                r.update(curr)
                print("json file name was:", json_path)
                json.dump(r, open(json_path, 'w'))
            except:
                print("[find_crossover_time] tried to dump json it didn't work")
        t_mid = np.mean([t_upper, t_lower])
        partition_sim(simulator, partition_type=partition1)
        c1, _ = find_optimal_cost(simulator, t_mid, epsilon, verbose=verbose, mc_samples=mc_samples)
        partition_sim(simulator, partition_type=partition2)
        c2, _ = find_optimal_cost(simulator, t_mid, epsilon, verbose=verbose, mc_samples=mc_samples)
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


def partition_sim(simulator, partition_type = "prob", chop_threshold = 0.5, optimize = False, nb_scaling = 0.0, time=0.01, epsilon=0.05, q_tile = 85):
    """
# Partitioning
Computes and sets a partition
## Partition schemes
### Possible strings and their function:
- "chop" - chop partition given a `chop_threshold`
- "exact_optimal_chop" - returns an optimal (upon convergence) partition given a `time`, `epsilon`, and `qtile`. See `readme.md` for more info.
- "prob" - computes the partition in lemma 2.1 given in "Composite Quantum Simulations" by Hagan and Wiebe.
- "random" - random partition
## Parameters
- simulator: A composite simulator, not type checked for flexibility later on
- partition_type: A string describing what partition method to use.
- chop_threshold: for "chop" partition, determines the spectral norm cutoff for each term to end up in QDrift
- optimize: artefact of an earlier Nelder-Mead partitioning method. Deprecated: leave false
- nb_scaling: a parametrization of nb within it's lower bound. Follows the scaling (1 + c)^2 * lower_bound. see paper for lower_bound
- time: required for probabilistic
- epsilon: required for probabilistic
- q_tile: calculates the inputed percentile of the spectral norm distribution to upper bound search space for "exact_optimal_chop". See readme.md for more info
    """
    if type(partition_type) != type("string"):
        print("[partition_sim] We only accept strings to describe the partition_type")
        return 1
    
    partition_type = partition_type.lower()

    if partition_type == "prob":
        partition_sim_prob(simulator, time, epsilon, nb_scaling, optimize)
        simulator.partition_type = "prob"

    elif partition_type == "optimize":
        partition_sim_optimize(simulator)
        simulator.partition_type = "optimize"
    
    elif partition_type == "random":
        partition_sim_random(simulator)
        simulator.partition_type = "random"
    
    elif partition_type == "chop":
        partition_sim_chop(simulator, chop_threshold)
        simulator.partition_type = "chop"
    
    elif partition_type == "optimal_chop":
        partition_sim_optimal_chop(simulator, time, epsilon)
        simulator.partition_type = "optimal_chop"

    elif partition_type=="exact_optimal_chop":
        exact_optimal_chop(simulator, time, epsilon, q_tile)
        simulator.partition_type = "exact_optimal_chop"

    elif partition_type == "trotter":
        partition_sim_trotter(simulator)
        simulator.partition_type = "trotter"
    
    elif partition_type == "first_order_trotter":
        simulator.set_trotter_order(1)
        partition_sim_trotter(simulator)
        simulator.partition_type = "trotter"
    
    elif partition_type == "second_order_trotter":
        simulator.set_trotter_order(2)
        partition_sim_trotter(simulator)
        simulator.partition_type = "trotter"

    elif partition_type == "qdrift":
        partition_sim_qdrift(simulator)
        simulator.partition_type = "qdrift"
    
    elif partition_type == "gbrt_prob":
        partition_sim_gbrt_prob(simulator, time, epsilon)
        simulator.partition_type = "gbrt_prob"
    
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
    # MATT.H - Not really sure if this is optimizing nb or how it's going about it. Need to fix. -- Squishes probability distribution
    if optimize and False:
        optimal_nb = optimize.minimize(prob_nb_optima, nb, method='Nelder-Mead', bounds = optimize.Bounds([0], [np.inf], keep_feasible = False)) #Nb attribute serves as an inital geuss in this partition
        nb_high = int(optimal_nb.x +1)
        nb_low = int(optimal_nb.x)
        prob_high = self.prob_nb_optima(nb_high) #check higher, (nb must be int)
        prob_low = self.prob_nb_optima(nb_low) #check lower 
        if prob_high > prob_low:
            self.nb = nb_low
        else:
            self.nb = nb_high
    return 0

def partition_sim_optimize(simulator, weight_threshold):
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

#MATT P- added spectral norms back to the class to make this more convinient. Did this without having to recompute norms, just keep track of them.
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

def exact_optimal_chop(simulator, time, epsilon, q_tile): 
    '''an optimizer that uses GBRT to return a tuple containing the optimized gate count and its location like so: (gate count, [nb, w])'''
    #q_tile computes the quartile of the norms and tries to prevent silly qdrift guesses 
    #with long computation times (ie placing all terms in qdrift at a large time)
    if type(simulator) != CompositeSim: raise TypeError("this partition only makes sense for simulators of the class CompositeSim")
    norm = np.linalg.norm(np.sum(simulator.unparsed_hamiltonian, axis=0), ord=2)
    dim1 = Integer(name = "nb", low=1, high = int(1/2 * len(simulator.spectral_norms)))
    if (time * norm > 1):
        dim2 = Real(name = "w", low=min(simulator.spectral_norms), high = np.percentile(simulator.spectral_norms, q_tile))
    else: 
        dim2 = Real(name = "w", low=min(simulator.spectral_norms), high = max(simulator.spectral_norms) + min(simulator.spectral_norms)/10)
    guess_point1 = [int((1/4)*len(simulator.spectral_norms)), statistics.median(simulator.spectral_norms)]
    guess_point2 = [max(int((1/20)*len(simulator.spectral_norms)), 1), statistics.median(simulator.spectral_norms)]
    guess_points = [guess_point1, guess_point2]
    dimensions = [dim1, dim2]

    @use_named_args(dimensions=dimensions)
    def obj_func(nb, w):
        simulator.nb = nb
        partition_sim(simulator, "chop", chop_threshold=w)
        return exact_cost(simulator, time, nb, epsilon)
    
    result = gbrt_minimize(func=obj_func,dimensions=dimensions, n_calls=50, n_initial_points = 20, 
                random_state=4, verbose = False, acq_func = "LCB", x0 = guess_points, n_jobs=1)
    
    opt_nb, opt_w = result.x
    opt_w = float(opt_w) #because int64 return type is not json serializable
    opt_nb = int(opt_nb)

    simulator.gate_count = (int(result.fun), [opt_nb, opt_w])
    #print("result.fun: ", result.fun)
    #print("result.x: ", result.x)

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

########################## LR Functions ###########################

def hamiltonian_localizer_1d(local_hamiltonian, sub_block_size):
    #A function to do an m=1 block decomposition of a nearest neighbour hamiltonian. Takes a tupel as input where 
    # indices 0, 1, 2 are the Hamiltonian terms, 1d lattice indices, and legnth of the lattice respectfully.
    #The funciton outputs 3 new "local" lists that can then be partitioned
    a_terms, a_index = [], []
    y_terms, y_index = [], []
    b_terms, b_index = [], []
    terms, indices, length = local_hamiltonian
    midpoint = int(length/2) 
    # start = midpoint - (int(sub_block_size/2))
    # stop = start + sub_block_size 
    # if start < 1:
    #     raise Exception("sub block is the size of or larger than the Hamiltonian")
    
    temp_norms = []
    for k in terms:
        temp_norms.append(np.linalg.norm(k, ord=2)) 
    
    h = max(temp_norms) #normalization factor (LR algorithm requires spectral norm < 1)
    
    bound = []
    upper = midpoint
    lower = midpoint
    iv = 1

    bound.append(midpoint)
    while iv < sub_block_size:
        if iv % 2 == 0:
            bound.append(upper + 1)
            upper +=1
        else:
            bound.append(lower - 1)
            lower += -1
            if midpoint - iv < 0: raise Exception("sub block is the size of or larger than the Hamiltonian")
        iv += 1
    bound = set(bound)
    print(bound)

    ix = 0
    while ix < len(terms): 
        if sub_block_size == 1:
            if set(indices[ix]) == bound:
                y_terms.append(1/h * terms[ix])
                y_index.append(indices[ix])
        else:
            if set(indices[ix]).issubset(bound):
                y_terms.append(1/h * terms[ix])
                y_index.append(indices[ix])

        if set(indices[ix]).issubset(set(arange(0, midpoint)).union(bound)): #A region
            a_terms.append(1/h * terms[ix])
            a_index.append(indices[ix])

        if set(indices[ix]).issubset(set(arange(midpoint, length)).union(bound)): #B region
            b_terms.append(1/h *terms[ix])
            b_index.append(indices[ix])

        ix += 1
        # if set(indices[ix]).issubset(arange(0, midpoint)):
        #     a_terms.append(1/h * terms[ix])
        #     a_index.append(indices[ix])
        # else:
        #     b_terms.append(1/h *terms[ix])
        #     b_index.append(indices[ix])
        
        # if not set(indices[ix]).isdisjoint(arange(start, stop)) == True: #Y region
        #     y_terms.append(1/h * terms[ix])
        #     y_index.append(indices[ix])

    #print(a_index)
    #print(y_index)
    #print(b_index)
    if ((len(a_terms) == 0) or (len(b_terms) == 0) or (len(y_terms) == 0)):
        raise Exception("poor block choice, one of the blocks is empty")
    return (np.array(a_terms), np.array(y_terms), np.array(b_terms)) #A and B here are really the AUB and BUC from the paper

def local_partition(simulator, partition, weights = None, time = 0.01, epsilon = 0.001): #weights is a list with ordering A, Y, B
        if type(simulator) != LRsim: raise TypeError("only works on LRsims")
        if partition == "chop":
            local_chop(simulator, weights)
            simulator.partition_type = "chop"
        elif partition == "optimal_chop":
            optimal_local_chop(simulator, time, epsilon)
            simulator.partition_type = "optimal_chop"
        elif partition == "trotter":
            local_trotter(simulator)
            simulator.partition_type = "trotter"
        elif partition == "qdrift":
            local_qdrift(simulator)
            simulator.partition_type = "qdrift"
        else:
            raise Exception("this is not a valid partition")
        return 0

def local_trotter(simulator):
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    for i in range(3):
        a_temp = []
        b_temp = []
        for j in range(len(simulator.local_hamiltonian[i])):
            a_temp.append(simulator.local_hamiltonian[i][j])
        simulator.internal_sims[i].set_partition(a_temp, b_temp)
    return 0

def local_qdrift(simulator):
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    for i in range(3):
        a_temp = []
        b_temp = []
        for j in range(len(simulator.local_hamiltonian[i])):
            b_temp.append(simulator.local_hamiltonian[i][j])
        simulator.internal_sims[i].set_partition(a_temp, b_temp)
    return 0

def local_chop(simulator, weights):
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    for i in range(3):
        a_temp = []
        b_temp = []
        for j in range(len(simulator.local_hamiltonian[i])):
            if simulator.spectral_norms[i][j] >= weights[i]:
                a_temp.append(simulator.local_hamiltonian[i][j]) ###should be appending terms not the spectral norms
            else:
                b_temp.append(simulator.local_hamiltonian[i][j])
            
        simulator.internal_sims[i].set_partition(a_temp, b_temp)
        
        #print("block " + str(i) + " has " + str(len(simulator.internal_sims[i].trotter_norms)) + 
            #" trotter terms and " + str(len(simulator.internal_sims[i].qdrift_norms)) + " qdrift terms")
    simulator.partition_type = "local_chop"
    return 0

def optimal_local_chop(simulator, time, epsilon): ### needs exact cost function to be operational
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    guess_points = []
    dimensions = []
    delta = min(simulator.spectral_norms[0] + simulator.spectral_norms[1] + simulator.spectral_norms[2])/10
    dim1 = Integer(name = "nb_a", low=1, high = len(simulator.spectral_norms[0]))
    dim2 = Integer(name="nb_y", low=1, high = len(simulator.spectral_norms[1]))
    dim3 = Integer(name="nb_b", low=1, high = len(simulator.spectral_norms[2]))
    dim4 = Real(name = "w_a", low=min(simulator.spectral_norms[0]), high = max(simulator.spectral_norms[0]) + delta)
    dim5 = Real(name = "w_y", low=min(simulator.spectral_norms[1]), high = max(simulator.spectral_norms[1]) + delta)
    dim6 = Real(name = "w_b", low=min(simulator.spectral_norms[2]), high = max(simulator.spectral_norms[2]) + delta) # delta is too allow everything to go into QDrift (bed on the \geq chop condition)

    for j in range(3):
        guess_points.append(int((1/2)*len(simulator.spectral_norms[j])))
    for i in range(3):
        guess_points.append(statistics.median(simulator.spectral_norms[i]))

    dimensions = [dim1, dim2, dim3, dim4, dim5, dim6]

    @use_named_args(dimensions=dimensions)
    def obj_func(nb_a, nb_y, nb_b, w_a, w_y, w_b):
        nb_list = [nb_a, nb_y, nb_b]
        weights = [w_a, w_y, w_b]
        #set_local_nb(simulator, nb_list)
        local_partition(simulator, partition = "chop", weights=weights, time=time, epsilon=epsilon)
        return exact_cost(simulator, time, nb_list, epsilon)
    
    result = gbrt_minimize(func=obj_func,dimensions=dimensions, n_calls=50, n_initial_points = 15, 
                random_state=4, verbose = False, acq_func = "LCB", x0 = guess_points, n_jobs=1)
    simulator.gate_count = result.fun
    print("(nba, nby, nbb, wa, wy, wb): " +str(result.x))
    return 0

def exact_cost(simulator, time, nb, epsilon): #relies on the use of density matrices
    '''
    Solves the search problem for the exact number of iterations of a given composite channel to achieve an error epsilon. 
    Requires monotonicity of the trace distance w.r.t iterations, so density matrices are required
    '''
    simulator.gate_count = 0
    if type(simulator.partition_type) == type(None): raise TypeError("call a partition function before calling this function")
    if type(simulator) == (CompositeSim): 
        if type(nb) == list: raise TypeError("this requires a single integer nb")
        simulator.nb = nb #redundancy
        if simulator.partition_type == "qdrift":
            get_trace_dist = lambda x : sim_trace_distance(simulator=simulator, time=time, iterations=1, nb = x)
        elif simulator.partition_type == 'trotter':
            get_trace_dist = lambda x : sim_trace_distance(simulator=simulator, time=time, iterations=x, nb = 1)
        else: 
            get_trace_dist = lambda x : sim_trace_distance(simulator=simulator, time=time, iterations=x, nb = simulator.nb)
    elif type(simulator) == (LRsim):
        if type(nb) != type([]): raise TypeError("this requires a list of nbs")
        set_local_nb(simulator, nb) #redundancy
        if simulator.partition_type == "qdrift":
            get_trace_dist = lambda x : sim_trace_distance(simulator=simulator, time=time, iterations=1, nb = [x,x,x])
        elif simulator.partition_type == 'trotter':
            get_trace_dist = lambda x : sim_trace_distance(simulator=simulator, time=time, iterations=x, nb = [1,1,1])
        else: 
            get_trace_dist = lambda x : sim_trace_distance(simulator=simulator, time=time, iterations=x, nb = simulator.nb)
    else: raise TypeError("only works on LR and Composite Sims")
    lower_bound = 1
    upper_bound = 2
    trace_dist = get_trace_dist(lower_bound)
    if trace_dist < epsilon:
        #print("[sim_channel_performance] Iterations too large, already below error threshold")
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
        raise Exception("[sim_channel_performance] function did not find a good point")

    get_trace_dist(mid)
    return simulator.gate_count

def sim_trace_distance(simulator, time, iterations, nb = None): 
    '''
    Returns the trace distance for a simulation given a simulator and other parameters. Allows one to redefine nb, even if already defined in the simulator object.
    '''
    #this fucntion has redundancy with the handelling of nb with respect to exact cost
    if type(simulator) == CompositeSim:
        if type(nb) == type(None): raise TypeError("required to set an nb")
        simulator.nb = nb
        if simulator.imag_time == False:
            return trace_distance(simulator.simulate(time, iterations), exact_time_evolution_density(simulator.unparsed_hamiltonian, 
                            time, simulator.initial_state))
        else:
            return trace_distance(simulator.simulate(time, iterations), exact_imaginary_channel(simulator.unparsed_hamiltonian, 
                            time, simulator.initial_state))

    elif type(simulator) == LRsim:
        if type(nb) != list: raise TypeError("local sims require an nb value per site")
        set_local_nb(simulator, nb)
        return trace_distance(simulator.simulate(time, iterations), exact_time_evolution_density(simulator.hamiltonian_list, 
                            time, simulator.initial_state))

    else: raise Exception("only defined for CompSim and LRsim")
    #the nb handeling here is a bit redundant when we look at the fact that it is first handeled in the objective function of
    #local optimal chop

def set_local_nb(simulator, nb_list): #is chop using this 
    if type(simulator) != LRsim: raise TypeError("only works on LRsims")
    simulator.nb = nb_list
    for i in range(3):
        simulator.internal_sims[i].nb = nb_list[i]

def normalize_hamiltonian(hamiltonian_list):
    '''
    Given a Hamiltonian as a 3d numpy array, normalizes each Hamiltonian term by the largest spectral norm.
    '''
    temp_norms = []
    norm_hamiltonian_list = []
    for k in hamiltonian_list:
        temp_norms.append(np.linalg.norm(k, ord=2)) 
    h = max(temp_norms)
    for xi in range(hamiltonian_list.shape[0]):
        norm_hamiltonian_list.append(1/h * hamiltonian_list[xi])
    return np.array(norm_hamiltonian_list)

# Imaginary time evolution -- this sectiuon is used for tools to help with imaginary time QDrift for Monte-Carlo
def exact_imaginary_channel(hamiltonian_list, i_time, initial_rho):
    '''
    Computes the exact time evolution of a density matrix given a Hamiltonian in imaginary time.
    '''
    evol_op = linalg.expm(-1 * i_time * sum(hamiltonian_list))
    ret = evol_op @ initial_rho @ evol_op
    return ret / np.trace(ret)

# Debugging purposes
# if __name__ == "__main__":
#     print("[utils.main]")
#     local_nb = [2, 2, 2]
#     heisenberg_hamiltonian_list = heisenberg_hamiltonian(8, 0.5, rng_seed=1)
#     print(heisenberg_hamiltonian_list[0].shape)
#     local_hamiltonian = hamiltonian_localizer_1d(heisenberg_hamiltonian_list, sub_block_size=2)
#     lrsim = LRsim(heisenberg_hamiltonian_list[0], local_hamiltonian, inner_order = 1, nb = local_nb, state_rand = False)
#     local_partition(lrsim, partition = "trotter", time= 0.01, epsilon = 0.01)
#     print("trace distances:")
#     for iter in range(1, 10):
#         print(sim_trace_distance(lrsim, 1e-5, iter, nb=[5, 5, 5]))
#         print('gate count per iter:', lrsim.gate_count / 51)
