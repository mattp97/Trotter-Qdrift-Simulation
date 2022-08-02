import numpy as np
import statistics
from scipy import linalg
from scipy import optimize
from scipy import interpolate
from numpy import inner, mat, random
# import time as time_this
from sympy import S, symbols, printing
from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import cProfile, pstats, io

# from utils import initial_state_randomizer


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
# - time: Floating point (no size guarantees) representing time for TOTAL simulation, NOT per
#         iteration. "t" parameter in the literature.
# - iterations: "r" parameter. This object will handle repeating the channel r times and dividing 
#               overall simulation into t/r chunks.
# - order: The trotter order, represented as "2k" in literature. 

class TrotterSim:
    def __init__(self, hamiltonian_list = [], order = 1, use_density_matrices = False):
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
        self.initial_state = psi_init

    def reset_init_state(self):
        if len(self.hamiltonian_list) > 0:
            self.initial_state = np.zeros((self.hamiltonian_list[0].shape[0]))
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
                self.exp_op_cache[(ix, timestep)] = linalg.expm(1j * self.hamiltonian_list[ix] * self.spectral_norms[ix] * timestep)
            exp_h = self.exp_op_cache[(ix, timestep)]
            matrix_mul_list.append(exp_h)

        # compute final output, let multi_dot figure out the cheapest way to perform all the matrix-matrix
        # or matrix-vec multiplications. ALSO ITERATIONS IS HANDLED HERE in a really slick/sneaky way.
        final_state = np.copy(self.initial_state)
        if self.use_density_matrices:
            reversed = matrix_mul_list.copy()
            reversed.reverse()
            for ix in range(len(reversed)):
                reversed[ix] = reversed[ix].conj().T
            final_state = np.linalg.multi_dot(matrix_mul_list * iterations + [self.initial_state] + reversed * iterations)
            if np.abs(np.abs(np.trace(final_state)) - np.abs(np.trace(self.initial_state))) > 1e-12:
                print("[Trotter_sim.simulate] Error: significant non-trace preserving operation was done.")
        else:
            final_state = np.linalg.multi_dot(matrix_mul_list * iterations + [self.initial_state])
        
        # TODO: This only uses the gates used for one side if we use density matrix, is this reasonable?
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
    def __init__(self, hamiltonian_list = [], rng_seed = 1, use_density_matrices=False):
        self.hamiltonian_list = []
        self.spectral_norms = []
        self.rng_seed = rng_seed
        self.gate_count = 0
        self.exp_op_cache = dict()
        self.conj_cache = dict()

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
        self.initial_state = psi_init

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
                self.exp_op_cache[ix] = linalg.expm(1.j * tau * self.hamiltonian_list[ix])
            op_list.append(self.exp_op_cache[ix])

        final_state = np.copy(self.initial_state)
        if self.use_density_matrices:
            reversed = op_list.copy()
            reversed.reverse()
            for ix in range(len(reversed)):
                reversed[ix] = reversed[ix].conj().T
            final_state = np.linalg.multi_dot(op_list + [self.initial_state] + reversed)
            if np.abs(np.abs(np.trace(final_state)) - np.abs(np.trace(self.initial_state))) > 1e-12:
                print("[Trotter_sim.simulate] Error: significant non-trace preserving operation was done.")
        else:
            final_state = np.linalg.multi_dot(op_list + [self.initial_state])

        self.final_state = final_state
        self.gate_count = samples
        return final_state

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

    

# Composite Simulator
# Inputs
# - hamiltonian_list: List of terms that compose your overall hamiltonian. Data type of each entry
#                     in the list is numpy matrix (preferably sparse, no guarantee on data struct
#                     actually used). Ex: H = A + B + C + D --> hamiltonian_list = [A, B, C, D]
#                     ASSUME SQUARE MATRIX INPUTS
# - rng_seed: Seed for the random number generator so results are reproducible.
# - inner_order: The Trotter-Suzuki product formula order for the Trotter channel
# - outter_order: The decomposition order / "outer-loop" order. Dictates which channels to simulate for how long
# - nb: number of samples to use in the QDrift channel.
# - state_rand: use random initial states if true, use computational |0> if false. 

###############################################################################################################################################################
class CompositeSim:
    def __init__(
                self,
                hamiltonian_list = [],
                inner_order = 1,
                outer_order = 1,
                rng_seed = 1,
                nb = 1,
                state_rand = False,
                use_density_matrices = False,
                verbose = False
                ):
        self.trotter_operators = []
        self.trotter_norms = []
        self.qdrift_operators = []
        self.qdrift_norms = []
        if len(hamiltonian_list) > 0:
            self.hilbert_dim = hamiltonian_list[0].shape[0] 
        else:
            self.hilbert_dim = 0

        # self.hilbert_dim = hamiltonian_list[0].shape[0] 
        self.rng_seed = rng_seed
        self.outer_order = outer_order 
        self.inner_order = inner_order

        self.state_rand = state_rand

        self.qdrift_sim = QDriftSim(use_density_matrices=use_density_matrices)
        self.trotter_sim = TrotterSim(order = inner_order, use_density_matrices=use_density_matrices)

        self.nb = nb #number of Qdrift channel samples. Useful to define as an attribute if we are choosing whether or not to optimize over it. Only used in analytic cost optimization

        self.gate_count = 0 

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
            rng_ix = np.random.randint(0, self.hilbert_dim)
            init = np.zeros((self.hilbert_dim, 1))
            init[rng_ix] = 1.
            if self.use_density_matrices == True:
                init = np.outer(init, init.conj())
            self.set_initial_state(init)
    
    def set_trotter_order(self, inner_order, outer_order=1):
        self.inner_order = inner_order
        self.trotter_sim.set_trotter_order(inner_order)
        self.outer_order = outer_order

    def set_initial_state(self, state):
        self.initial_state = state
        self.trotter_sim.set_initial_state(state)
        self.qdrift_sim.set_initial_state(state)

    def set_hamiltonian(self, hamiltonian_list):
        self.hilbert_dim = hamiltonian_list[0].shape[0]
        self.set_partition(hamiltonian_list, [])
        self.reset_initial_state()

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
        print("[CompositeSim] # of Trotter terms:", len(self.trotter_norms), ", # of Qdrift terms: ", len(self.qdrift_norms), ", and Nb = ", self.nb)

    # Simulate time evolution approximately. returns the final state and stores the number of gates executed as self.gate_count
    def simulate(self, time, iterations): 
        self.gate_count = 0
        outer_loop_timesteps = compute_trotter_timesteps(2, time / (1. * iterations), self.outer_order)

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
        # return nested simulator initial states to normal
        self.trotter_sim.set_initial_state(self.initial_state)
        self.qdrift_sim.set_initial_state(self.initial_state)
        return current_state
    
    # Computes time evolution exactly. Returns the final state and makes no internal changes.
    def simulate_exact_output(self, time):
        h_trott = [self.trotter_norms[ix] * self.trotter_operators[ix] for ix in range(len(self.trotter_norms))]
        h_qd = [self.qdrift_norms[ix] * self.qdrift_operators[ix] for ix in range(len(self.qdrift_norms))]
        h = sum(h_qd + h_trott)
        u = linalg.expm(1j * h * time)
        if self.use_density_matrices:
            return u @ self.initial_state @ u.conj().T
        else:
            return u @ self.initial_state


class LRsim: 
    def __init__(
        self,
        hamiltonian_list, 
        local_hamiltonian, #a tuple of lists of H terms: each index of the tuple contains a list representing a local block
        inner_order,
        partition,
        nb = [], #should be a list in this case
        state_rand = True,
        rng_seed = 1
    ):
        self.gate_count = 0
        self.hamiltonian_list = hamiltonian_list
        self.local_hamiltonian = local_hamiltonian
        self.inner_order = inner_order
        self.spectral_norms = [] # a list of lists of the spectral norms of each local bracket
        self.partition = partition 
        self.state_rand = state_rand
        self.hilbert_dim = hamiltonian_list[0].shape[0] 
        self.rng_seed = rng_seed

        self.comp_sim_A = CompositeSim(inner_order=inner_order, outer_order=1, use_density_matrices=True)
        self.comp_sim_Y = CompositeSim(inner_order=inner_order, outer_order=1, use_density_matrices=True)
        self.comp_sim_B = CompositeSim(inner_order=inner_order, outer_order=1, use_density_matrices=True)

        self.internal_sims = [self.comp_sim_A, self.comp_sim_Y, self.comp_sim_B]

        np.random.seed(self.rng_seed)
        self.nb = nb
        if type(self.nb) != type(list): raise TypeError("nb is a list that requires input for each local block")

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
        current_state = np.copy(self.initial_state)
        for i in range(iterations):
            self.comp_sim_A.set_initial_state(current_state)
            current_state = self.comp_sim_A.simulate(time/iterations, 1)
            self.gate_count += self.comp_sim_A.gate_count

            self.comp_sim_Y.set_initial_state(current_state)
            current_state = self.comp_sim_Y.simulate(time/iterations, 1)
            self.gate_count += self.comp_sim_A.gate_count

            self.comp_sim_B.set_initial_state(current_state)
            current_state = self.comp_sim_B.simulate(time/iterations, 1)
            self.gate_count += self.comp_sim_A.gate_count

        self.final_state = current_state
        return np.copy(self.final_state)
        
    
