import numpy as np
from scipy import linalg
import multiprocessing as mp
import cProfile, pstats, io
from itertools import repeat
from copy import deepcopy

# from .utils import MC_SAMPLES_DEFAULT, graph_hamiltonian
# import utils

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

#A function to generate a random initial state that is normalized MP-- sampled from gauss to avoid measure concentration
def initial_state_randomizer(hilbert_dim): #changed to sample each dimension from a gaussain
     initial_state = []
     x = np.random.normal(size=(hilbert_dim, 1))
     y = np.random.normal(size=(hilbert_dim, 1))
     initial_state = x + (1j * y) 
     initial_state_norm = initial_state / np.linalg.norm(initial_state)
     return initial_state_norm.reshape((hilbert_dim, 1))
     
FLOATING_POINT_PRECISION = 1e-10

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
        raise Exception('[compute_trotter_timesteps] trotterOrder input is not an int')

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
        raise Exception("[compute_trotter_timesteps] trotterOrder seems to be bad")

# A basic trotter simulator organizer.
# Inputs
# - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
#                     in the list is numpy matrix (preferably sparse, no guarantee on data struct
#                     actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
#                     ASSUME SQUARE MATRIX INPUTS
# - iterations: "r" parameter. This object will handle repeating the channel r times and dividing 
#               overall simulation into t/r chunks.
# - order: The trotter order, represented as "2k" in literature. 
class TrotterSim:
    def __init__(self, hamiltonian_list = [], order = 1, use_density_matrices = False, imag_time = False):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.order = order
        self.gate_count = 0
        self.exp_op_cache = dict()
        self.conj_cache = dict()
        self.imag_time = imag_time

        # Use the first computational basis state as the initial state until the user specifies.
        if len(hamiltonian_list) == 0:
            self.initial_state = np.zeros((1,1))
        else:
            self.initial_state = np.zeros((hamiltonian_list[0].shape[0]))
        self.initial_state[0] = 1
        
        self.use_density_matrices = use_density_matrices
        if use_density_matrices == True:
            self.initial_state = np.outer(self.initial_state, self.initial_state.conj())

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
    
    def set_trotter_order(self, order):
        self.order = order

    # Assumes terms are already normalized
    def set_hamiltonian(self, mat_list = [], norm_list = []):
        if len(mat_list) != len(norm_list):
            print("[Trott - set_hamiltonian] Incorrect length arrays")
            return 1
        self.hamiltonian_list = mat_list
        self.spectral_norms = norm_list
    
    def get_hamiltonian_list(self):
        ret = []
        for ix in range(len(self.hamiltonian_list)):
            ret.append(np.copy(self.hamiltonian_list[ix]) * self.spectral_norms[ix])
        return ret

    def clear_hamiltonian(self):
        self.hamiltonian_list = []
        self.spectral_norms = []
        return 0

    # Do some sanity checking before storing. Check if input is proper dimensions and an actual
    # quantum state.
    # TODO: Check that the input matches shape? aka density matrix vs statevector
    def set_initial_state(self, psi_init):
        if len(self.hamiltonian_list) == 0:
            return
        hilbert_dim = self.hamiltonian_list[0].shape[0]
        is_input_density_matrix = (psi_init.shape[0] == hilbert_dim) and (psi_init.shape[1] == hilbert_dim)
        if self.use_density_matrices and is_input_density_matrix:
            self.initial_state = psi_init
        elif self.use_density_matrices and (is_input_density_matrix == False):
            self.initial_state = np.outer(psi_init, np.copy(psi_init).conj().T)
        elif (self.use_density_matrices == False) and (is_input_density_matrix == False):
            self.initial_state = psi_init
        else:
            print("[TrotterSim.set_initial_state] Tried to set a density matrix to initial state for non-density matrix sim. Doing nothing.")

    def reset_init_state(self):
        if len(self.hamiltonian_list) > 0:
            self.initial_state = np.zeros((self.hamiltonian_list[0].shape[0], 1))
            self.initial_state[0] = 1.
            if self.use_density_matrices == True:
                self.initial_state = np.outer(self.initial_state, self.initial_state.conj())
        #else:
            #print("[TrotterSim.reset_init_state] I don't have any dimensions")

    def simulate(self, time, iterations):
        self.gate_count = 0
        if len(self.hamiltonian_list) == 0:
            return np.copy(self.initial_state)

        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["iterations"] != iterations):
                self.exp_op_cache.clear()
        
        if len(self.exp_op_cache) == 0:
            self.exp_op_cache["time"] = time
            self.exp_op_cache["iterations"] = iterations

        op_time = time/iterations
        steps = compute_trotter_timesteps(len(self.hamiltonian_list), op_time, self.order)
        matrix_mul_list = []
        for (ix, timestep) in steps:
            if (ix, timestep) not in self.exp_op_cache:
                if self.imag_time == False:
                    self.exp_op_cache[(ix, timestep)] = linalg.expm(1j * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep)
                else:
                    self.exp_op_cache[(ix, timestep)] = linalg.expm(-1 * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep)
            exp_h = self.exp_op_cache[(ix, timestep)]
            matrix_mul_list.append(exp_h)

        final_state = np.copy(self.initial_state)
        if len(matrix_mul_list) == 1:
                scrunch_op = matrix_mul_list[0]
        else: scrunch_op = np.linalg.multi_dot(matrix_mul_list)
        if self.use_density_matrices:
            if self.imag_time == False:
                scrunch_op_iters = np.linalg.matrix_power(scrunch_op, iterations)
                final_state = scrunch_op_iters @ self.initial_state @ scrunch_op_iters.conj().T
            else:
                scrunch_op_iters = np.linalg.matrix_power(scrunch_op, iterations)
                channel_out = scrunch_op_iters @ self.initial_state @ scrunch_op_iters
                final_state = channel_out / np.trace(channel_out)
            if np.abs(np.abs(np.trace(final_state)) - np.abs(np.trace(self.initial_state))) > 1e-12:
                print("[Trotter_sim.simulate] Error: significant non-trace preserving operation was done.")
        else:
            if self.imag_time == False:
                final_state = np.linalg.matrix_power(scrunch_op, iterations) @ self.initial_state
            else:
                imag_out = np.linalg.matrix_power(scrunch_op, iterations) @ self.initial_state
                final_state = imag_out / np.linalg.norm(imag_out, ord=2)

        self.gate_count = len(matrix_mul_list) * iterations

        return final_state        

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
    def __init__(self, hamiltonian_list = [], rng_seed = 1, use_density_matrices=False, exact_qd=False, imag_time=False):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.rng_seed = rng_seed
        self.gate_count = 0
        self.exp_op_cache = dict()
        self.conj_cache = dict()
        self.exact_qd=exact_qd #adding this to choose to build E[]
        self.imag_time = imag_time

        # Use the first computational basis state as the initial state until the user specifies.
        self.prep_hamiltonian_lists(hamiltonian_list)
        if len(hamiltonian_list) == 0:
            self.initial_state = np.zeros((1,1))
        else:
            self.initial_state = np.zeros((hamiltonian_list[0].shape[0]))
        self.initial_state[0] = 1

        self.use_density_matrices = use_density_matrices
        if use_density_matrices == True:
            self.initial_state = np.outer(self.initial_state, self.initial_state.conj())
        self.final_state = np.copy(self.initial_state)
        
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
    # TODO: check inputs match? aka density matrix vs not
    def set_initial_state(self, psi_init):
        if len(self.hamiltonian_list) == 0:
            return 
        hilbert_dim = self.hamiltonian_list[0].shape[0]
        is_input_density_matrix = (psi_init.shape[0] == hilbert_dim) and (psi_init.shape[1] == hilbert_dim)
        if self.use_density_matrices and is_input_density_matrix:
            self.initial_state = psi_init
        elif self.use_density_matrices and (is_input_density_matrix == False):
            self.initial_state = np.outer(psi_init, np.copy(psi_init).conj().T)
        elif (self.use_density_matrices == False) and (is_input_density_matrix == False):
            self.initial_state = psi_init
        else:
            print("[QDriftSim.set_initial_state] Tried to set a density matrix to initial state for non-density matrix sim. Doing nothing.")

    def reset_init_state(self):
        if len(self.hamiltonian_list) > 0:
            self.initial_state = np.zeros((self.hamiltonian_list[0].shape[0]))
            self.initial_state[0] = 1.
        if self.use_density_matrices:
            self.initial_state = np.outer(self.initial_state, self.initial_state.conj())

    # Assumes terms are already normalized
    def set_hamiltonian(self, mat_list = [], norm_list = []):
        if len(mat_list) != len(norm_list):
            print("[QD - set_hamiltonian] Incorrect length arrays")
            return 1
        self.hamiltonian_list = mat_list
        self.spectral_norms = norm_list

    def clear_hamiltonian(self):
        self.hamiltonian_list = []
        self.spectral_norms = []
        return 0

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
        if self.exact_qd == True:
            return self.construct_density(time, samples)
        else:
            self.gate_count = 0
            if samples == 0:
                return np.copy(self.initial_state)
            
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
            obtained_samples = self.draw_hamiltonian_samples(samples)

            op_list = []
            for ix in obtained_samples:
                if ix not in self.exp_op_cache:
                    if self.imag_time == False:
                        self.exp_op_cache[ix] = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
                    else: 
                        self.exp_op_cache[ix] = linalg.expm(-1 * tau * self.hamiltonian_list[ix])
                op_list.append(self.exp_op_cache[ix])

            final_state = np.copy(self.initial_state)
            if self.use_density_matrices:
                if self.imag_time == False:
                    reversed = op_list.copy()
                    reversed.reverse()
                    for ix in range(len(reversed)):
                        reversed[ix] = reversed[ix].conj().T
                    final_state = np.linalg.multi_dot(op_list + [self.initial_state] + reversed)

                else:
                    channel_out = np.linalg.multi_dot(op_list + [self.initial_state] + op_list)
                    final_state =  channel_out / np.trace(channel_out)

                if np.abs(np.abs(np.trace(final_state)) - np.abs(np.trace(self.initial_state))) > 1e-12:
                    print("[Trotter_sim.simulate] Error: significant non-trace preserving operation was done.")

            else: 
                if self.imag_time == False:
                    final_state = np.linalg.multi_dot(op_list + [self.initial_state])
                else: 
                    imag_out = np.linalg.multi_dot(op_list + [self.initial_state])
                    final_state = imag_out / np.linalg.norm(imag_out, ord=2)

            self.final_state = final_state
            self.gate_count = samples
            return final_state

            # if self.use_density_matrices:
            #     channel_out = np.linalg.multi_dot(op_list + [self.initial_state] + op_list)
            #     final_state =  channel_out / np.trace(channel_out)
            #     if np.abs(np.abs(np.trace(final_state)) - np.abs(np.trace(self.initial_state))) > 1e-12:
            #         print("[Trotter_sim.simulate] Error: significant non-trace preserving operation was done.")

            # else:
            #     evol_out = np.linalg.multi_dot(op_list + [self.initial_state])
            #     final_state = evol_out / np.linalg.norm(evol_out, ord=2)

    def construct_density(self, time, samples):
        if self.use_density_matrices == False:
            print("[QDSim.construct_density] You're trying to construct a density matrix with vector initial state. try again")
            return np.copy(self.initial_state)
        self.gate_count = 0
        lamb = np.sum(self.spectral_norms)
        tau = time * lamb / (samples * 1.0)
        if "time" in self.exp_op_cache:
            if (self.exp_op_cache["time"] != time) or (self.exp_op_cache["samples"] != samples) or (len(self.conj_cache) != len(self.spectral_norms)): #incase partition changes
                self.exp_op_cache.clear()
                self.conj_cache.clear() #based on code will follow the conditions above

        if (len(self.hamiltonian_list) == 0): # or (len(self.hamiltonian_list) == 1) caused issues in comp sim
            return np.copy(self.initial_state) #make the choice not to sample a lone qdrift term
        
        if len(self.exp_op_cache) == 0:
            self.exp_op_cache["time"] = time
            self.exp_op_cache["samples"] = samples
            for k in range(len(self.spectral_norms)):
                if self.imag_time == False: #only need if time is real
                    self.exp_op_cache[k] = linalg.expm(1.j * tau * self.hamiltonian_list[k])
                    self.conj_cache[k] = self.exp_op_cache.get(k).conj().T
                else:
                    self.exp_op_cache[k] = linalg.expm(-1 * tau * self.hamiltonian_list[k])

        rho = np.copy(self.initial_state)
        if self.imag_time == False:
            for i in range(samples):
                channel_output = np.zeros((self.hamiltonian_list[0].shape[0], self.hamiltonian_list[0].shape[0]), dtype = 'complex')
                for j in range(len(self.spectral_norms)):
                    channel_output += (self.spectral_norms[j]/lamb) * self.exp_op_cache.get(j) @ rho @ self.conj_cache.get(j) #an error is creeping in here (I think for the case len(b) = 1)
                rho = channel_output
            
        else:
            for i in range(samples):
                channel_output = np.zeros((self.hamiltonian_list[0].shape[0], self.hamiltonian_list[0].shape[0]), dtype = 'complex')
                for j in range(len(self.spectral_norms)):
                    channel_output += (self.spectral_norms[j]/lamb) * self.exp_op_cache.get(j) @ rho @ self.exp_op_cache.get(j) #an error is creeping in here (I think for the case len(b) = 1)
                rho = channel_output / np.trace(channel_output)
        if self.imag_time==False:
            self.final_state = rho
        else: 
            print("normalized by " + str(np.trace(rho)))
            self.final_state = rho / np.trace(rho)
        self.gate_count = samples
        return np.copy(self.final_state)


###############################################################################################################################################################
class CompositeSim:
    """
    # Composite Simulator
    A simulator whose sole purpose is to store hamiltonians, a cache of operator exponentials, an initial state, and then simulate it's time dynamics.
    ## Inputs
    - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
                        in the list is numpy matrix (preferably sparse, no guarantee on data struct
                        actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
                        ASSUME SQUARE MATRIX INPUTS
    - rng_seed: Seed for the random number generator so results are reproducible.
    - inner_order: The Trotter-Suzuki product formula order for the Trotter channel
    - outter_order: The decomposition order / "outer-loop" order. Dictates which channels to simulate for how long
    - nb: number of samples to use in the QDrift channel.
    - state_rand: use random initial states if true, use computational |0> if false. 
    """
    def __init__(
                self,
                hamiltonian_list = [],
                inner_order = 1,
                outer_order = 1,
                rng_seed = 1,
                nb = 1,
                state_rand = False,
                use_density_matrices = False,
                exact_qd = False,
                verbose = False,
                imag_time = False
                ):
        self.trotter_operators = []
        self.trotter_norms = []
        self.qdrift_operators = []
        self.qdrift_norms = []
        self.spectral_norms = []
        if len(hamiltonian_list) > 0:
            self.hilbert_dim = hamiltonian_list[0].shape[0] 
        else:
            self.hilbert_dim = 0

        self.rng_seed = rng_seed
        self.outer_order = outer_order 
        self.inner_order = inner_order
        self.imag_time = imag_time

        self.state_rand = state_rand

        self.qdrift_sim = QDriftSim(use_density_matrices=use_density_matrices, exact_qd=exact_qd, imag_time=imag_time)
        self.trotter_sim = TrotterSim(order = inner_order, use_density_matrices=use_density_matrices, imag_time=imag_time)

        self.nb = nb #number of Qdrift channel samples. Useful to define as an attribute if we are choosing whether or not to optimize over it.

        self.gate_count = 0 
        self.partition_type = None

        #Choose to randomize the initial state or just use computational |0>
        #Should probably add functionality to take an initial state as input at some point
        if self.hilbert_dim > 0:
            if self.state_rand == True:
                self.initial_state = initial_state_randomizer(self.hilbert_dim)
            else:
                # Use the first computational basis state as the initial state until the user specifies.
                self.initial_state = np.zeros((self.hilbert_dim, 1))
                self.initial_state[0] = 1.
        else:
            self.initial_state = np.zeros((1,1))
        self.use_density_matrices = use_density_matrices
        if use_density_matrices == True:
            self.initial_state = np.outer(self.initial_state, self.initial_state.conj())
        self.trotter_sim.set_initial_state(self.initial_state)
        self.qdrift_sim.set_initial_state(self.initial_state)
        self.final_state = np.copy(self.initial_state)
        self.unparsed_hamiltonian = np.copy(hamiltonian_list) #the unmodified input matrix

        # NOTE: This sets the default behavior to be a fully Trotter channel!
        self.set_partition(hamiltonian_list, [])
        np.random.seed(self.rng_seed)

        # if nb_optimizer == True:
        #     print("Nb is equal to " + str(self.nb))

    # Returns a dictionary representing the simulator object, designed for multiprocessing.
    def to_pickle(self):
        d = {}

    def get_hamiltonian_list(self):
        ret = []
        for ix in range(len(self.trotter_norms)):
            ret.append(self.trotter_norms[ix] * self.trotter_operators[ix])
        for jx in range(len(self.qdrift_norms)):
            ret.append(self.qdrift_norms[jx] * self.qdrift_operators[jx])
        return ret
    
    def get_lambda(self):
        return sum(self.qdrift_norms) + sum(self.trotter_norms)

    def reset_initial_state(self):
        self.initial_state = np.zeros((self.hilbert_dim, 1))
        self.initial_state[0] = 1.
        if self.use_density_matrices == True:
            self.initial_state = np.outer(self.initial_state, self.initial_state.conj())
        self.trotter_sim.reset_init_state()
        self.qdrift_sim.reset_init_state()

    def randomize_initial_state(self):
        if self.hilbert_dim == 0:
            self.set_initial_state(np.zeros((1,1)))
        elif self.hilbert_dim >= 1:
            initial_state = []
            x = np.random.normal(size = (self.hilbert_dim, 1))
            y = np.random.normal(size = (self.hilbert_dim, 1))
            initial_state = x + (1j * y) 
            # np.linalg.norm defaults to frobenius, or L2 norm which is what we want
            initial_state_normalized = initial_state / np.linalg.norm(initial_state)
            self.set_initial_state(initial_state_normalized.reshape((self.hilbert_dim, 1)))

    def set_density_matrices(self, use_density_matrices):
        self.use_density_matrices = use_density_matrices
        self.trotter_sim.use_density_matrices = use_density_matrices
        self.qdrift_sim.use_density_matrices = use_density_matrices
        self.reset_initial_state()

    def set_trotter_order(self, inner_order, outer_order=1):
        self.inner_order = inner_order
        self.trotter_sim.set_trotter_order(inner_order)
        self.outer_order = outer_order

    def set_initial_state(self, state):
        is_input_density_matrix = (state.shape[0] == self.hilbert_dim) and (state.shape[1] == self.hilbert_dim)
        if self.use_density_matrices and (is_input_density_matrix == False):
            density_matrix = np.outer(state, np.copy(state).conj().T)
            self.initial_state = density_matrix
            self.trotter_sim.set_initial_state(density_matrix)
            self.qdrift_sim.set_initial_state(density_matrix)
        elif self.use_density_matrices and (is_input_density_matrix == True):
            self.initial_state = np.copy(state)
            self.trotter_sim.set_initial_state(state)
            self.qdrift_sim.set_initial_state(state)
        elif (self.use_density_matrices == False) and (is_input_density_matrix == False):
            self.initial_state = state
            self.trotter_sim.set_initial_state(state)
            self.qdrift_sim.set_initial_state(state)
        else:
            print("[CompositeSim.set_initial_state] trying to set density matrix input to non-density matrix simulator.")

    def set_hamiltonian(self, hamiltonian_list):
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.set_partition(hamiltonian_list, [])
        self.reset_initial_state()

    def set_exact_qd(self, val):
        self.qdrift_sim.exact_qd = val
    
    # Inputs: trotter_list - a python list of numpy arrays, each element is a single term in a hamiltonian
    #         qdrift_list - same but these terms go into the qdrift simulator. 
    # Note: each of these matrices should NOT be normalized, all normalization should be done internally
    def set_partition(self, trotter_list, qdrift_list):
        self.gate_count = 0
        self.trotter_sim.gate_count = 0
        self.qdrift_sim.gate_count = 0
        self.trotter_norms, self.trotter_operators = [], []
        self.qdrift_norms, self.qdrift_operators = [], []
        
        for matrix in trotter_list:
            temp_norm = np.linalg.norm(matrix, ord = 2)
            self.trotter_norms.append(temp_norm)
            self.trotter_operators.append(matrix / temp_norm)
        
        for matrix in qdrift_list:
            temp_norm = np.linalg.norm(matrix, ord = 2)
            self.qdrift_norms.append(temp_norm)
            self.qdrift_operators.append(matrix / temp_norm)
        
        self.spectral_norms = []
        self.spectral_norms = self.trotter_norms + self.qdrift_norms

        # TODO check clear and then set
        if len(qdrift_list) > 0:
            self.qdrift_sim.set_hamiltonian(norm_list=self.qdrift_norms, mat_list=self.qdrift_operators)
        elif len(qdrift_list) == 0:
            self.qdrift_sim.clear_hamiltonian()

        if len(trotter_list) > 0:
            self.trotter_sim.set_hamiltonian(norm_list=self.trotter_norms, mat_list=self.trotter_operators)
        elif len(trotter_list) == 0:
            self.trotter_sim.clear_hamiltonian()
        
        return 0

    def print_partition(self):
        """
        prints out the partitioning.
        """
        print("[CompositeSim] # of Trotter terms:", len(self.trotter_norms), ", # of Qdrift terms: ", len(self.qdrift_norms), ", and Nb = ", self.nb)

    def simulate(self, time, iterations):
        """
        Implements a single instance of the composite channel for the specified input time and interations. Specifically, the QDrift channel
        represents a single sampled value. 
        """
        self.gate_count = 0
        outer_loop_timesteps = compute_trotter_timesteps(2, time / (1. * iterations), self.outer_order)

        current_state = np.copy(self.initial_state)
        for i in range(iterations):
            for (ix, sim_time) in outer_loop_timesteps:
                if ix == 0 and len(self.trotter_norms) > 0:
                    self.trotter_sim.set_initial_state(current_state)
                    current_state = self.trotter_sim.simulate(sim_time, 1)
                    self.gate_count += self.trotter_sim.gate_count
                if ix == 1 and len(self.qdrift_norms) > 0:
                    self.qdrift_sim.set_initial_state(current_state)
                    current_state = self.qdrift_sim.simulate(sim_time, self.nb)
                    self.gate_count += self.qdrift_sim.gate_count
        # return nested simulator initial states to normal
        self.trotter_sim.set_initial_state(self.initial_state)
        self.qdrift_sim.set_initial_state(self.initial_state)

        if self.imag_time == True and self.use_density_matrices == True:
            if np.abs(np.trace(current_state) - 1) > FLOATING_POINT_PRECISION:
                raise Exception("A non-trace preserving operation was excecuted")

        return current_state

    def to_multiprocessing_dictionary(self, time, iterations):
        # now we need a simple pickle-able function to return an ndarray representing the simulated output for time and iterations.
        # what do we provide as input?
        trotter_ham = self.trotter_sim.get_hamiltonian_list()
        qdrift_ham = self.qdrift_sim.get_hamiltonian_list()
        if len(trotter_ham) > 0:
            processed_trotter_ham = [mat.tolist() for mat in trotter_ham]
            processed_trotter_ham.append(trotter_ham[0].shape)
        else:
            processed_trotter_ham = []
        if len(qdrift_ham) > 0:
            processed_qdrift_ham = [mat.tolist() for mat in qdrift_ham]
            processed_qdrift_ham.append(qdrift_ham[0].shape)
        else:
            processed_qdrift_ham = []
        d = {}
        state = np.copy(self.initial_state).tolist()
        state.append(self.initial_state.shape)
        d["initial_state"] = state
        d["trotter_hamiltonian"] = processed_trotter_ham
        d["qdrift_hamiltonian"] = processed_qdrift_ham
        d["inner_order"] = self.inner_order
        d["outer_order"] = self.outer_order
        d["nb"] = self.nb
        d["time"] = time
        d["iterations"] = iterations
        d["use_density_matrices"] = self.use_density_matrices
        d["rng_seed"] = self.rng_seed
        return d

    def simulate_mc(self, time, iterations, mc_samples=100):
        """
        Represents a monte-carlo'd implementation of the QDrift channel to simulate the overall composite channel. This is specifically implemented as a 
        starting point for allowing multiprocessing.
        """
        print("[simulate_mc] monte carlo samples:", mc_samples)
        self.gate_count = 0

        final_state = np.zeros(self.initial_state.shape, dtype=self.initial_state.dtype)
        dict_state = self.to_multiprocessing_dictionary(time, iterations)
        argument_iterator = []
        for _ in range(mc_samples):
            argument_iterator.append((deepcopy(dict_state), np.random.randint(1)))
        with mp.Pool() as pool:
            states, shapes, gate_counts = zip(*pool.starmap(simulate_worker_thread, iter(argument_iterator)))
            for ix in range(len(states)):
                final_state += np.array(states[ix], dtype=final_state.dtype).reshape(shapes[ix])
            final_state /= len(states)
        self.gate_count = np.mean(gate_counts)
        return final_state

    # Computes time evolution exactly. Returns the final state and makes no internal changes.
    def exact_final_state(self, time):
        h_trott = [self.trotter_norms[ix] * self.trotter_operators[ix] for ix in range(len(self.trotter_norms))]
        h_qd = [self.qdrift_norms[ix] * self.qdrift_operators[ix] for ix in range(len(self.qdrift_norms))]
        h = sum(h_qd + h_trott)
        u = linalg.expm(1j * h * time)
        if self.use_density_matrices:
            return u @ self.initial_state @ u.conj().T
        else:
            return u @ self.initial_state

def composite_sim_from_dictionary(d):
    processed_trotter = d.get("trotter_hamiltonian", [])
    processed_qdrift = d.get("qdrift_hamiltonian", [])
    if len(processed_trotter) > 0:
        shape = processed_trotter.pop(-1)
        final_trotter = [np.array(mat).reshape(shape) for mat in processed_trotter]
    else:
        final_trotter = []
    if len(processed_qdrift) > 0:
        shape = processed_qdrift.pop(-1)
        final_qdrift = [np.array(mat).reshape(shape) for mat in processed_qdrift]
    else:
        final_qdrift = []
    ham_list = final_qdrift + final_trotter
    sim = CompositeSim(hamiltonian_list=ham_list, inner_order=d.get("inner_order", 1), outer_order=d.get("outer_order", 1), verbose=True, nb=d.get("nb", 1), use_density_matrices=d.get("use_density_matrices", True), rng_seed=d.get("rng_seed", 1))
    sim.set_partition(final_trotter, final_qdrift)
    state = d.get("initial_state")
    shape = state.pop(-1)
    sim.set_initial_state(np.array(state).reshape(shape))
    return sim

def simulate_worker_thread(dict_state, rng_seed):
    dict_state["rng_seed"] = rng_seed
    simulator = composite_sim_from_dictionary(dict_state)
    final_state = simulator.simulate(dict_state.get("time", 1e-3), iterations=dict_state.get("iterations", 1))
    shape = final_state.shape
    return (final_state.tolist(), shape, simulator.gate_count)

# A Lieb-Robinson local composite simulator
class LRsim: 
    def __init__(
        self,
        hamiltonian_list, 
        local_hamiltonian, #a tuple of lists of H terms: each index of the tuple contains a list representing a local block
        inner_order,
        nb = [], #should be a list in this case
        state_rand = True,
        rng_seed = 1
    ):
        self.gate_count = 0

        self.hamiltonian_list = []

        #normalize the incoming hamiltonian
        temp_norms = []
        for k in hamiltonian_list:
            temp_norms.append(np.linalg.norm(k, ord=2)) 
        h = max(temp_norms)
        for xi in range(hamiltonian_list.shape[0]):
            self.hamiltonian_list.append(1/h * hamiltonian_list[xi])
        self.hamiltonian_list = np.array(self.hamiltonian_list)

        self.local_hamiltonian = local_hamiltonian
        self.inner_order = inner_order
        self.spectral_norms = [] # a list of lists of the spectral norms of each local bracket
        self.state_rand = state_rand
        self.hilbert_dim = hamiltonian_list[0].shape[0] 
        self.rng_seed = rng_seed
        self.partition_type = None

        self.comp_sim_A = CompositeSim(hamiltonian_list = self.local_hamiltonian[0], inner_order=inner_order, outer_order=1, use_density_matrices=True, exact_qd=True)
        self.comp_sim_Y = CompositeSim(hamiltonian_list = self.local_hamiltonian[1], inner_order=inner_order, outer_order=1, use_density_matrices=True, exact_qd=True)
        self.comp_sim_B = CompositeSim(hamiltonian_list = self.local_hamiltonian[2], inner_order=inner_order, outer_order=1, use_density_matrices=True, exact_qd=True)

        self.internal_sims = [self.comp_sim_A, self.comp_sim_Y, self.comp_sim_B]

        np.random.seed(self.rng_seed)
        #Set the nb for each sim
        self.nb = nb
        if type(self.nb) != type([]): raise TypeError("nb is a list that requires input for each local block")
        for l in range(len(self.nb)):
            self.internal_sims[l].nb = self.nb[l]

        #create a list of lists for spectral norms
        for i in range(len(self.local_hamiltonian)):
            temp = []
            for j in range(len(self.local_hamiltonian[i])):
                temp.append(np.linalg.norm(self.local_hamiltonian[i][j], ord = 2))
            self.spectral_norms.append(temp)

        #Choose to randomize the initial state or just use computational |0>
        #Should probably add functionality to take an initial state as input at some point
        if self.state_rand == True:
            self.initial_state = initial_state_randomizer(self.hilbert_dim)
        else:
            # Use the first computational basis state as the initial state until the user specifies.
            self.initial_state = np.zeros((self.hilbert_dim, 1))
            self.initial_state[0] = 1.
        
        self.initial_state = np.outer(self.initial_state, self.initial_state.conj())

        self.comp_sim_A.set_initial_state(self.initial_state)
        self.comp_sim_Y.set_initial_state(self.initial_state)
        self.comp_sim_B.set_initial_state(self.initial_state)
        self.final_state = np.copy(self.initial_state)

    def simulate(self, time, iterations):
        self.gate_count=0
        current_state = np.copy(self.initial_state)

        self.comp_sim_A.set_initial_state(current_state)
        current_state = self.comp_sim_A.simulate(time, iterations)
        self.gate_count += self.comp_sim_A.gate_count

        self.comp_sim_Y.set_initial_state(current_state)
        current_state = self.comp_sim_Y.simulate(-1*time, iterations)
        self.gate_count += self.comp_sim_Y.gate_count

        self.comp_sim_B.set_initial_state(current_state)
        current_state = self.comp_sim_B.simulate(time, iterations)
        self.gate_count += self.comp_sim_B.gate_count

        self.final_state = current_state
        return np.copy(self.final_state)

if __name__ == "__main__":
    from utils import graph_hamiltonian
    ham = graph_hamiltonian(7,1,1)
    sim = CompositeSim(hamiltonian_list=ham, use_density_matrices=True, verbose=True)
    sim.randomize_initial_state()
    sim.simulate_mc(1e-3, 2, mc_samples=8)
