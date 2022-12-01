from compilers import *
from utils import *
from lattice_hamiltonians import *
import json
import multiprocessing as mp
if __name__ == "__main__":

#time evolution
    t_i = 0.05
    t_f=1
    t_steps = 20
    times = np.geomspace(t_i, t_f, t_steps)
    epsilon=0.001
    print(times)

    heisenberg_hamiltonian_list = heisenberg_hamiltonian(length = 9, b_field=1, rng_seed=1, b_rand=True)
    print("local model shape: " + str(heisenberg_hamiltonian_list[0].shape))

    data = {}
    for m in range(2, 8):
        local_hamiltonian_list = hamiltonian_localizer_1d(heisenberg_hamiltonian_list, sub_block_size=m)
        local_sim = LRsim(heisenberg_hamiltonian_list[0], local_hamiltonian_list, inner_order=1, state_rand=True)
        local_partition(local_sim, "trotter")
        
        std_trotter = CompositeSim(heisenberg_hamiltonian_list[0], inner_order=1, use_density_matrices = True)

        if m == 1:
            data["local_trotter"] = {}
        else:
            data["block_{0}".format(m)] = {}
        
        for t in times:
            if m == 1:
                data["local_trotter"][t] = exact_cost(std_trotter, t, 1, epsilon = epsilon)
            else:
                data["block_{0}".format(m)][t] = exact_cost(local_sim, t, [1,1,1], epsilon=epsilon)

    outfile = open("Block data, 9 spin")
    json.dump(data, outfile)
    outfile.close()