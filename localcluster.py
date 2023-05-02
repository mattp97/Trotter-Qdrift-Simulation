from compilers import *
from utils import *
from lattice_hamiltonians import *
import json
from joblib import Parallel, delayed
from datetime import datetime
from itertools import product

if __name__ == "__main__":
    #sim params
    cluster = False
    n_proc = 10
    job_name = 'local_heisenberg_10'
    h_params = local_heisenberg_hamiltonian(length=8, b_field=5, rng_seed=1, b_rand=False)
    hamiltonian = (normalize_hamiltonian(h_params[0]), h_params[1], h_params[2])
    norm = np.linalg.norm(np.sum(hamiltonian[0], axis=0), ord=2)
    print("local model shape: " + str(hamiltonian[0].shape))

    #time evolution
    t_i = 0.01 / norm
    t_f= (3/2) * math.pi /norm
    t_steps = 20
    times = np.geomspace(t_i, t_f, t_steps)
    blocks = np.linspace(2, 6, 2)
    block_nb = [2, 2, 2]
    epsilon=0.001

    partitions = ["chop"]
    
    data = {}
    data['partitions'] = partitions
    data["time"] = times
    data["norm"] = norm
    data["epsilon"] = epsilon
    data["blocks"] = blocks
    data["block_nb"] = block_nb

    sim_params = list(product(blocks, partitions, times))
        
    def cost_t(params):
        blocked_hamiltonian = hamiltonian_localizer_1d(hamiltonian, sub_block_size=params[0])
        local_sim = LRsim(hamiltonian[0], blocked_hamiltonian, inner_order=1, nb=block_nb)
        local_partition(local_sim, params[1], weights = [0.1, 0.1, 0.1])
        return exact_cost(local_sim, time=params[2], nb=block_nb, epsilon=epsilon)
    
    results = Parallel(n_jobs=n_proc)(delayed(cost_t)(p) for p in sim_params)

    #unpack results -- this will have to be modified if we include more partitions (one if statement should do it)
    n = len(blocks) # number of elements to group together
    count = 0
    for j in results:
        if count % n == 0:
            # create a new list for every n elements
            current_list = []
            # associate the new list with a dictionary key
            data['block ' + str(count)] = current_list
        # add the current element to the current list
        current_list.append(j)
        count += 1
    print(data)


    # for m in range(2, 8):
    #     local_hamiltonian_list = hamiltonian_localizer_1d(heisenberg_hamiltonian_list, sub_block_size=m)
    #     local_sim = LRsim(heisenberg_hamiltonian_list[0], local_hamiltonian_list, inner_order=1, state_rand=True)
    #     local_partition(local_sim, "trotter")
        
    #     std_trotter = CompositeSim(heisenberg_hamiltonian_list[0], inner_order=1, use_density_matrices = True)

    #     if m == 1:
    #         data["local_trotter"] = {}
    #     else:
    #         data["block_{0}".format(m)] = {}
        
    #     for t in times:
    #         if m == 1:
    #             data["local_trotter"][t] = exact_cost(std_trotter, t, 1, epsilon = epsilon)
    #         else:
    #             data["block_{0}".format(m)][t] = exact_cost(local_sim, t, [1,1,1], epsilon=epsilon)

    # outfile = open("Block data, 9 spin")
    # json.dump(data, outfile)
    # outfile.close()

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")

    if cluster == False:
        filepath = os.path.join(os.path.expanduser('~'), 'Desktop', 'sim_runs', job_name + dt_string)
    else:
        filepath = os.path.join(os.path.expanduser('~'), 'compsim_data', job_name + dt_string)
    # open the file for writing and specify the file path
    with open(filepath, 'x') as f:
        # use json.dump to write the dictionary to the file
        json.dump(data, f)