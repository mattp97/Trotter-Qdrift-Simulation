import numpy as np
import scipy
import multiprocessing as mp
import cProfile, pstats, io

class ImaginaryQdrift: #this class will only use the exact density matrix. It will not have state vector functionality or sampling

    def __init__(self, hamiltonian_list=[], rng_seed=1, state_rand = False):

        self.hamiltonian_list = hamiltonian_list
        self.unparsed_hamiltonian_list = np.copy(hamiltonian_list)
        self.rng_seed = rng_seed

        self.spectral_norms = []
        self.normed_hamiltonian_list = []
        self.hilbert_dim = self.hamiltonian_list[0].shape[0]
        self.exp_cache = {}
        self.gate_count = 0

        self.initial_state = np.zeros(self.hilbert_dim)
        self.initial_state[0] = 1
        self.initial_state = np.outer(self.initial_state, self.initial_state)

        self.final_state = np.copy(self.initial_state)

        for i in hamiltonian_list:
            h_norm = np.linalg.norm(i, ord=2)
            h_term = i / h_norm

            self.spectral_norms.append(h_norm)
            self.normed_hamiltonian_list.append(h_term)



    def construct_density(self, time, samples):
        if 'time' in self.exp_cache:
            if self.exp_cache['time'] != time or self.exp_cache['samples'] != samples:
                self.exp_cache.clear()
        
        self.exp_cache['time'] = time
        self.exp_cache['samples'] = samples

        lamb = sum(self.spectral_norms)
        tau  = (time * lamb) / samples

        for i in range(len(self.spectral_norms)):
            self.exp_cache[i] = scipy.linalg.expm(-1 * tau * self.normed_hamiltonian_list[i])

        rho = np.copy(self.initial_state)

        for n in range(samples):
            interm_rho = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype = 'complex')
            for l in range(len(self.spectral_norms)): 
                interm_rho += (self.spectral_norms[l]/lamb) * self.exp_cache.get(l) @ rho @ self.exp_cache.get(l)
            rho = interm_rho 

        rho = rho / np.trace(rho)
        self.final_state = np.copy(rho)

        self.gate_count = samples
        
        return self.final_state
        

    def simulate(self, time, samples):
        return self.construct_density(time, samples)

    def trace_dist(self, time, samples):
        final_state = self.simulate(time, samples)

        exact_evol_op = scipy.linalg.expm(-1 * time * sum(self.unparsed_hamiltonian_list))
        exact_evolution = exact_evol_op @ self.initial_state @ exact_evol_op

        exact_evolution = exact_evolution / np.trace(exact_evolution)

        diff = final_state - exact_evolution

        dist = 1/2 * np.abs(np.trace(scipy.linalg.sqrtm(diff.conj().T @ diff)))
        return dist




