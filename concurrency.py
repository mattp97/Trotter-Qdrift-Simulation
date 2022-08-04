import numpy as np
import multiprocessing
from compilers import *

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

# Here is the simulate code for composite sim
def simulate(time, iterations): 
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

# Here it is for trotter sim
def simulate(time, iterations):
    self.gate_count = 0
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
    

# Because multiprocessing does not like non-pickleable objects we need to put all the necessary information into a dictionary.
def picklable_infidelity_sample(data_dict, time, rng_seed):


