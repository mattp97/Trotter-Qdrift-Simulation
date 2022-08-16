from compilers import *
from utils import *
import json
import multiprocessing as mp

if __name__ == "__main__":
    graph_hamiltonian_list = graph_hamiltonian(7, 1, 1)
    print("graph hamiltonian shape: " +str(graph_hamiltonian_list.shape))
    sim = CompositeSim(graph_hamiltonian_list, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter1 = CompositeSim(graph_hamiltonian_list, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter2 = CompositeSim(graph_hamiltonian_list, inner_order=2, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    qdrift = CompositeSim(graph_hamiltonian_list, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)

    t_i = 0.05
    t_f=1.75
    t_steps = 20
    times = np.geomspace(t_i, t_f, t_steps)
    epsilon=0.001
    print(times)

    CompSim_results = dict()
    local_trot_results = dict()
    TrotSim1_results = dict()
    TrotSim2_results = dict()
    QDSim_results = dict()

    #local_partition(local_trot, "trotter")
    partition_sim(trotter1, "trotter")
    partition_sim(trotter2, "trotter")

    for t in times:
        #local_partition(local_sim, "optimal_chop", time=t, epsilon=epsilon)
        #CompSim_results[t] = int(local_sim.gate_count)
        #local_trot_results[t] = int(exact_cost(local_trot, time=t, nb=[1,1,1], epsilon=epsilon))
        partition_sim(sim, "exact_optimal_chop", time =t, epsilon=epsilon)
        CompSim_results[t] = int(sim.gate_count)

        TrotSim1_results[t] = int(exact_cost(trotter1, time=t, nb=1, epsilon=epsilon))

        TrotSim2_results[t] = int(exact_cost(trotter2, time=t, nb=1, epsilon=epsilon))