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
from compilers import CompositeSim, TrotterSim, QDriftSim, DensityMatrixSim

MC_SAMPLES_DEFAULT=10

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

    sim_output = []

    if type(simulator) == QDriftSim:
        sim_output = simulator.simulate(time, nbsamples)
    
    if type(simulator) == TrotterSim:
        sim_output = simulator.simulate(time, iterations)
    
    if type(simulator) == CompositeSim:
        sim_output = simulator.simulate(time, iterations)

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

# TODO: take in a "guess" parameter which tells you a ballpark. Then search around there, avoiding the need for exponential
#       backoff and then binary search.
# Varies the number of iterations needed for a simulator to acheive its infidelity threshold. WARNING: will modify the input
# simulator starting state to a random basis state! Probably need to change this. 
# Inputs
# - simulator: Either Trotter, QDrift, or Composite simulator. If a QDrift simulator is used, then the number
#              of samples is varied. If a Composite simulator is used then iterations is varied while QDrift
#              samples are held fixed.
# - mc_samples: Monte Carlo samples for wave function states, shouldn't be necessary with density matrices?
# Output: the gate cost necessary for the simulator to meet the threshold.
def find_optimal_cost(simulator, time, infidelity_threshold, mc_samples=MC_SAMPLES_DEFAULT):
    hamiltonian_list = simulator.get_hamiltonian_list()
    
    # choose random basis state to start out with
    init = 0 * simulator.initial_state
    init[np.random.randint(0, len(init))] = 1.
    simulator.set_initial_state(init)
    exact_final_state = exact_time_evolution(hamiltonian_list, time, initial_state=init)

    # now we can simplify infidelity to nice lambda, NOTE get_inf returns a TUPLE 
    if type(simulator) == TrotterSim or type(simulator) == CompositeSim:
        get_inf = lambda x: mutli_infidelity_sample(simulator, time, exact_final_state, iterations = x, mc_samples=mc_samples)
    elif type(simulator) == QDriftSim:
        get_inf = lambda x: mutli_infidelity_sample(simulator, time, exact_final_state, nbsamples= x, mc_samples=mc_samples)

    # compute reasonable upper and lower bounds
    iter_lower = 1
    iter_upper = 2 ** 20
    # print("[find_optimal_cost] finding bounds")
    for n in range(20):
        # print("[find_optimal_cost] n: ", n)
        inf_tup, costs = zip(*get_inf(2 ** n))
        inf_list = list(inf_tup)
        # print("[find_optimal_cost] mean infidelity:", np.mean(inf_list))

        if is_threshold_met(inf_list, infidelity_threshold) == False:
            iter_lower = 2 ** n
        else:
            iter_upper = 2 ** n
            break

    # bisection search until we find it.
    mid = 1
    count = 0
    current_inf = (1., 1)
    # print("[find_optimal_cost] beginning search with lower, upper:", iter_lower, iter_upper)
    while iter_upper - iter_lower  > 1 and count < 30:
        count += 1
        mid = (iter_upper + iter_lower) / 2.0
        iters = math.ceil(mid)
        # print("[find_optimal_cost] count:", count, ", upper:",iter_upper, ", lower: ", iter_lower, ", mid:", mid)
        
        inf_tup, costs = zip(*get_inf(iters))
        infidelities = list(inf_tup)
        # print("[find_optimal_cost] current_inf:", np.mean(infidelities))
        if is_threshold_met(infidelities, infidelity_threshold):
            iter_lower = iters
        else:
            iter_upper = iters
    ret = get_inf(iter_upper)
    inf_tup, costs = zip(*ret)
    # print("[find_optimal_cost] iters:", iter_upper, ", inf_mean: ", np.mean(list(inf_tup)), " +- (", np.std(list(inf_tup)), ")")
    # print("[find_optimal_cost] Average cost:", np.mean(list(costs)))
    return np.mean(list(costs))

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

    elif partition_type == "qdrift":
        partition_sim_qdrift(simulator)
    
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

    trotter = []
    qdrift = []
    for ix in range(len(hamiltonian)):
        sample = np.random.random()
        spectral_norm = np.linalg.norm(hamiltonian[ix], ord=2)
        qdrift_prob = min(1, chi / spectral_norm) 
        if sample < qdrift_prob:
            qdrift.append(hamiltonian[ix])
        else:
            trotter.append(hamiltonian[ix])
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
    trotter, qdrift = [], []
    for ix in range(len(hamiltonian)):
        sample = np.random.random()
        if sample >= 0.5:
            trotter.append(hamiltonian[ix])
        else:
            qdrift.append(hamiltonian[ix])
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
        return find_optimal_cost(simulator, time, epsilon)
    result = gbrt_minimize(obj_fn, dimensions=dimensions, n_calls=30, n_initial_points=5, random_state=4, verbose=True, acq_func="LCB")
    print("result.fun: ", result.fun)
    print("result.x: ", result.x)
    print("result:", result)

def partition_sim_optimal_chop_2(simulator):
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

    # A function similar to that of sim_channel_performance, however, this one is defined only to be optimized not executed
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

def partition_sim_trotter(simulator):
    simulator.trotter_operators = simulator.hamiltonian_list
    simulator.trotter_norms = simulator.spectral_norms
    simulator.qdrift_operators = []
    simulator.qdrift_norms = []
    return 0

def partition_sim_qdrift(simulator):
    simulator.qdrift_operators = simulator.hamiltonian_list
    simulator.qdrift_norms = simulator.spectral_norms
    simulator.trotter_operators = []
    simulator.trotter_norms = []
    return 0

def test():
    hamiltonian = graph_hamiltonian(2, 1, 1)
    t = 0.01
    eps = 0.1
    compsim = CompositeSim(hamiltonian_list=hamiltonian, inner_order=2)
    trottsim = TrotterSim(hamiltonian_list=hamiltonian)
    qsim = QDriftSim(hamiltonian_list=hamiltonian)
    partition_sim(compsim, "prob", nb_scaling=.01)
    compsim.print_partition()
    partition_sim(compsim, "prob", nb_scaling=.05)
    compsim.print_partition()
    partition_sim(compsim, "prob", nb_scaling=.1)
    compsim.print_partition()
    partition_sim(compsim, "prob", nb_scaling=.5)
    compsim.print_partition()
    partition_sim(compsim, "prob", nb_scaling=0.05)
    compsim.reset_init_state()
    print("Check to see if the hamiltonian has remained the same after many partitionings.")
    hamiltonian_copy = compsim.get_hamiltonian_list()
    print("Difference norm:", np.linalg.norm(sum(hamiltonian) - sum(hamiltonian_copy)))
    compsim.print_partition()
    print("testing trotter")
    print(find_optimal_cost(trottsim, .01, .1))
    print(find_optimal_cost(compsim, 0.01, .1))
    print("#" * 25)
    print("Now testing optimal chop")
    partition_sim(compsim, "optimal_chop")

test()