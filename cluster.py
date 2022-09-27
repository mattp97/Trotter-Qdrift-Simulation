from compilers import *
from utils import *
import json
import multiprocessing as mp

if __name__ == "__main__":
    graph_hamiltonian_list = graph_hamiltonian(8, 1, 1)
    print("graph hamiltonian shape: " +str(graph_hamiltonian_list.shape))
    sim = CompositeSim(graph_hamiltonian_list, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter1 = CompositeSim(graph_hamiltonian_list, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    trotter2 = CompositeSim(graph_hamiltonian_list, inner_order=2, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)
    qdrift = CompositeSim(graph_hamiltonian_list, inner_order=1, outer_order=1, nb=1, state_rand=True, exact_qd=True, use_density_matrices=True)

    t_i = 0.01
    t_f=2
    t_steps = 30
    times = np.geomspace(t_i, t_f, t_steps)
    epsilon=0.001
    
    CompSim_results = dict()
    TrotSim1_results = dict()
    TrotSim2_results = dict()
    QDSim_results = dict()
    
    partition_sim(trotter1, "trotter")
    partition_sim(trotter2, "trotter")
    partition_sim(qdrift, "qdrift")
    
    for t in times:
        print("At time :" +str(t))

        partition_sim(sim, "exact_optimal_chop", time =t, epsilon=epsilon)
        CompSim_results[t] = int(sim.gate_count)

        TrotSim1_results[t] = int(exact_cost(trotter1, time=t, nb=1, epsilon=epsilon))

        TrotSim2_results[t] = int(exact_cost(trotter2, time=t, nb=1, epsilon=epsilon))

        if t <= times[20]:
            QDSim_results[t] = int(exact_cost(qdrift, time=t, nb=1, epsilon=epsilon))

        if t == times[20]:
            int_out = open("intermediate_opchop_trot_trot2_qd", "w")
            json.dump(CompSim_results, int_out)
            json.dump(TrotSim1_results, int_out)
            json.dump(TrotSim2_results, int_out)
            json.dump(QDSim_results, int_out)
            int_out.close()

    outfile = open("8_graph_rand_states_opchop1_trot1_trot2_qd", "w")

    json.dump(CompSim_results, outfile)
    json.dump(TrotSim1_results, outfile)
    json.dump(TrotSim2_results, outfile)
    json.dump(QDSim_results, outfile)
        
    outfile.close()