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
from compilers import *


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


def partition_sim(simulator, partition_type = "prob", weight_threshold = 0.5, optimize = False):
    if type(partition_type) != type("string"):
        print("[partition_sim] We only accept strings to describe the partition_type")
        return 1
    
    partition_type = partition_type.lower()

    if partition_type == "prob":
        partition_sim_prob(simulator)
    
    elif partition_type == "optimize":
        partition_sim_optimize(simulator)
    
    elif partition_type == "random":
        partition_sim_random(simulator)
    
    elif partition_type == "chop":
        partition_sim_chop(simulator, weight_threshold)
    
    elif partition_type == "optimal_chop":
        partition_sim_optimal_chop(simulator)

    elif partition_type == "trotter":
        partition_sim_trotter(simulator)

    elif partition_type == "qdrift":
        partition_sim_qdrift(simulator)
    
    else:
        print("[partition_sim] Did not recieve valid partition. Valid options are: 'prob', 'optimize', 'random', 'chop', 'optimal_chop', 'trotter', and 'qdrift'.")
        return 1

def partition_sim_prob(simulator):
    if simulator.trotter_sim.order > 1:
         k = simulator.trotter_sim.order/2
    else: 
        print("partition not defined for this order") 
        return 1
    
    upsilon = 2*(5**(k -1))
    lamb = simulator.get_lambda()

    # TODO: how to optimize this quantity? not sure what optimal is without computing gate counts?
    if simulator.nb_optimizer == True and False:
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
    
    for i in range(len(simulator.spectral_norms)):
        num = np.random.random()
        prob = (1- min((1/simulator.spectral_norms[i])*chi, 1))
        if prob >= num:
            self.a_norms.append(([i, simulator.spectral_norms[i]]))
        else:
            self.b_norms.append(([i, simulator.spectral_norms[i]]))
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
    for i in range(len(simulator.spectral_norms)):
        sample = np.random.random()
        if sample >= 0.5:
            simulator.a_norms.append([i, simulator.spectral_norms[i]])
        elif sample < 0.5:
            simulator.b_norms.append([i, simulator.spectral_norms[i]])
    simulator.a_norms = np.array(simulator.a_norms, dtype='complex')
    simulator.b_norms = np.array(simulator.b_norms, dtype='complex')
    return 0

def partition_sim_chop(simulator, weight_threshold):
    for i in range(len(simulator.spectral_norms)):
        if simulator.spectral_norms[i] >= weight_threshold:
            simulator.a_norms.append(([i, simulator.spectral_norms[i]]))
        else:
            simulator.b_norms.append(([i, simulator.spectral_norms[i]]))
    simulator.a_norms = np.array(simulator.a_norms, dtype='complex')
    simulator.b_norms = np.array(simulator.b_norms, dtype='complex')
    return 0


def partitioning(simulator, weight_threshold):
        #if ((self.partition == "trotter" or self.partition == "qdrift" or self.partition == "random" or self.partition == "optimize")): #This condition may change for the optimize scheme later
            #print("This partitioning method does not require repartitioning") #Checks that our scheme is sane, prevents unnecessary time wasting
            #return 1 maybe work this in again later?

        if self.partition == "prob":
            
        
        #Nelder-Mead optimization protocol based on analytic cost function for first order
        elif self.partition == "optimize": 
            
        
        elif self.partition == "random":
            
        
        elif self.partition == "chop": #cutting off at some value defined by the user
            

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