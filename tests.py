#!/usr/bin/env python

from utils import *
from compilers import *
import numpy as np
import sys
import pickle
import os
import shutil
import matplotlib.pyplot as plt

INFIDELITY_TEST_TYPE = "infidelity"
TRACE_DIST_TEST_TYPE = "trace_distance"
GATE_COST_TEST_TYPE = "gate_cost"
CROSSOVER_TEST_TYPE = "crossover"
OPTIMAL_PARTITION_TEST_TYPE = "optimal_partition"
LAUNCHPAD = "launchpad"
EXPERIMENT_LABEL = "experiment_label"
DEFAULT_EXPERIMENT_LABEL = "default_experiment_label"
MINIMAL_SETTINGS = [EXPERIMENT_LABEL,
                    "verbose",
                    'use_density_matrices',
                    't_start',
                    't_stop',
                    't_steps',
                    'partitions',
                    'infidelity_threshold',
                    'num_state_samples',
                    'base_directory',
                    'test_type'
]

# For coordinating runs. 
class Experiment:
    def __init__(
        self,
        hamiltonian_list = [],
        base_directory="./",
        experiment_label="default_experiment_label",
        t_start=1e-3,
        t_stop=1e-1,
        t_steps=50,
        num_state_samples=5,
        partitions=["first_order_trotter", "qdrift"],
        test_type=INFIDELITY_TEST_TYPE,
        infidelity_threshold=0.05,
        use_density_matrices=False,
        verbose=False
        ):
        self.sim = CompositeSim(hamiltonian_list=hamiltonian_list, use_density_matrices=use_density_matrices, verbose=verbose)
        self.use_density_matrices = use_density_matrices
        self.times = np.geomspace(t_start, t_stop, t_steps)
        self.partitions = partitions
        self.test_type=test_type.lower()
        self.infidelity_threshold = infidelity_threshold
        self.verbose = verbose
        self.num_state_samples = num_state_samples
        if type(os.getenv("SCRATCH")) != type(None):
            self.base_directory = os.getenv("SCRATCH")
        elif len(sys.argv) > 1:
            self.base_directory = sys.argv[1]
        else:
            self.base_directory = base_directory
        if base_directory[-1] != '/':
            self.base_directory = base_directory + '/'
        self.experiment_label = experiment_label
    
    
    def run_gate_cost(self):
        results = {}
        for partition in self.partitions:
            if self.verbose:
                print("[run_gate_cost] evaluating partition:", partition)
            outputs = []
            heuristic = 1
            partition_sim(self.sim, partition)
            for t in self.times:
                if self.verbose:
                    print("[run_gate_cost] evaluating time:", t)
                out = 0
                for _ in range(self.num_state_samples):
                    self.sim.randomize_initial_state()
                    cost, iters = find_optimal_cost(self.sim, t, self.infidelity_threshold, heuristic=heuristic, verbose=self.verbose)
                    heuristic = iters
                    out += cost
                outputs.append(out / self.num_state_samples)
            results[partition] = outputs
        return results
    
    def run_infidelity(self):
        results = {}
        for partition in self.partitions:
            if self.verbose:
                print("[run_infidelity] evaluating partition:", partition)
            time_inf_tups = []
            partition_sim(self.sim, partition)
            print("[run_infidelity] confirming density_matrices are set? self.sim.use_density_matrices=", self.sim.use_density_matrices)
            for t in self.times:
                if self.verbose:
                    print("[run_infidelity] evaluating time:", t)
                per_state_out = []
                for _ in range(self.num_state_samples):
                    if self.verbose:
                        print("[run_infidelity] on state sample: ", _)
                    self.sim.randomize_initial_state()
                    exact_final_state = self.sim.simulate_exact_output(t)
                    mc_inf, _ = zip(*multi_infidelity_sample(self.sim, t, exact_final_state, mc_samples=10))
                    if self.verbose:
                        print("[run_infidelity] observed monte carlo avg inf: ", np.mean(mc_inf), " +- ", np.std(mc_inf))
                    per_state_out.append(np.mean(mc_inf))
                time_inf_tups.append((t, np.mean(per_state_out)))
                print("[run_infidelity] average inf: ", np.mean(per_state_out))
            results[partition] = time_inf_tups
        return results

    def run_crossover(self):
        results = {}
        if len(self.partitions) < 2:
            print("[run_crossover] Error: trying to compute crossover with less than two partitions. Bail.")
            return
        p1 = self.partitions[0]
        p2 = self.partitions[1]
        if len(self.times) < 2:
            print("[run_crossover] Error: trying to compute crossover without enough endpoints. Bail.")
            return
        t1 = self.times[0]
        t2 = self.times[-1]
        results["crossover"] = find_crossover_time(self.sim, p1, p2, t1, t2, verbose=self.verbose)
        return results

    def run_optimal_partition(self):
        results = {}
        prob_vec, nb, cost = find_optimal_partition(self.sim, self.times[0], self.infidelity_threshold)
        results["optimal_probabilities"] = prob_vec
        results["optimal_nb"] = nb
        results["optimal_cost"] = cost
        return results

    def run_trace_distance(self):
        results = {}
        for partition in self.partitions:
            if self.verbose:
                print("[run_trace_distance] evaluating partition:", partition)
            time_dist_tups = []
            partition_sim(self.sim, partition)
            print("[run_trace_distance] confirming density_matrices are set? self.sim.use_density_matrices=", self.sim.use_density_matrices)
            for t in self.times:
                if self.verbose:
                    print("[run_trace_distance] evaluating time:", t)
                per_state_out = []
                for _ in range(self.num_state_samples):
                    if self.verbose:
                        print("[run_trace_distance] on state sample: ", _)
                    self.sim.randomize_initial_state()
                    exact_final_state = self.sim.simulate_exact_output(t)
                    mc_dist = multi_trace_distance_sample(self.sim, t, exact_final_state, mc_samples=5000)
                    if self.verbose:
                        print("[run_trace_distance] observed monte carlo avg dist: ", np.mean(mc_dist), " +- ", np.std(mc_dist))
                    per_state_out.append(np.mean(mc_dist))
                time_dist_tups.append((t, np.mean(per_state_out)))
                print("[run_trace_distance] average dist: ", np.mean(per_state_out))
            results[partition] = time_dist_tups
        return results

    # TODO: implement multithreading?
    # TODO: How to handle probabilistic partitionings?
    def run(self):
        final_results = {}
        final_results["times"] = np.copy(self.times).tolist()
        final_results["test_type"] = self.test_type
        if self.test_type == INFIDELITY_TEST_TYPE:
            out = self.run_infidelity()
        elif self.test_type == GATE_COST_TEST_TYPE:
            out = self.run_gate_cost()
        elif self.test_type == CROSSOVER_TEST_TYPE:
            out = self.run_crossover()
        elif self.test_type == OPTIMAL_PARTITION_TEST_TYPE:
            out = self.run_optimal_partition()
        elif self.test_type == TRACE_DIST_TEST_TYPE:
            out = self.run_trace_distance()
        final_results.update(out)
        self.results = final_results
        if self.base_directory[-1] != '/':
            self.base_directory += '/'
        if os.path.exists(self.base_directory + "outputs") == False:
            try:
                os.mkdir(self.base_directory + "outputs")
                output_dir = self.base_directory + "outputs/"
            except:
                print("[Experiment.run] no output directory and I couldn't make one. storing in base directory")
                output_dir = self.base_directory
        else:
            output_dir = self.base_directory + "outputs/"
        try:
            pickle.dump(final_results, open(output_dir + self.experiment_label + ".pickle", 'wb'))
            print("[Experiment.run] successfully wrote output to: ", output_dir + self.experiment_label + '.pickle')
        except:
            print("[Experiment.run] ERROR: could not save output to:", self.base_directory + 'results.pickle')



    # TODO: This isn't really necessary anymore with the setup scripts
    def pickle_settings(self, output_path):
        settings = {}
        settings["experiment_label"] = self.experiment_label.get(EXPERIMENT_LABEL, DEFAULT_EXPERIMENT_LABEL)
        settings["verbose"] = self.verbose
        settings["use_density_matrices"] = self.use_density_matrices
        settings["t_start"] = self.times[0]
        settings["t_stop"] = self.times[-1]
        settings["t_steps"] = len(self.times)
        settings["partitions"] = self.partitions
        settings["infidelity_threshold"] = self.infidelity_threshold
        settings["num_state_samples"] = self.num_state_samples
        settings["base_directory"] = self.base_directory
        settings["test_type"] = self.test_type
        pickle.dump(settings, open(output_path, 'wb'))

    # given an exact absoulute path to a settings file, loads them into the experiment object. Note that we ensure defaults and formatting
    # are handled on write, so loading should not have to use these defaults but do this out of precaution. 
    def load_settings(self, settings_path):
        settings = pickle.load(open(settings_path, 'rb'))
        self.experiment_label     = settings.get(EXPERIMENT_LABEL, "default_experiment_label")
        self.verbose              = settings.get("verbose", True)
        self.use_density_matrices = settings.get("use_density_matrices", False)
        self.times                = np.geomspace(settings.get("t_start", 1e-4), settings.get("t_stop", 1e-2), settings.get("t_steps", 10))
        self.partitions           = settings.get("partitions", ["first_order_trotter", "qdrift"])
        self.infidelity_threshold = settings.get("infidelity_threshold", 0.05)
        self.num_state_samples    = settings.get("num_state_samples", 5)
        self.test_type            = settings.get("test_type", GATE_COST_TEST_TYPE)
        # Drop self.output_directory? if we can find the settings then just output there

    # Inputs:
    # - An absoluate path to a pickled hamiltonian file. converts to numpy array and makes sure simulator is loaded with that hamiltonian.
    def load_hamiltonian(self, input_path):
        unpickled = pickle.load(open(input_path, 'rb'))
        output_shape = unpickled[-1]
        ham_list = []
        for ix in range(len(unpickled) - 1):
            ham_list.append(np.array(unpickled[ix]).reshape(output_shape))
        print("[load_hamiltonian] loaded this many terms: ", len(ham_list))
        self.sim.set_hamiltonian(ham_list)

def pickle_hamiltonian(output_path, unparsed_ham_list):
    shape = unparsed_ham_list[0].shape
    # convert to pickle'able datatype
    to_pickle = [mat.tolist() for mat in unparsed_ham_list]
    to_pickle.append(shape)
    pickle.dump(to_pickle, open(output_path, 'wb'))

def hamiltonian_entry_point():
    if len(sys.argv) == 3:
        ham_path = sys.argv[2]
    else:
        print("[hamiltonian_entry_point] No hamiltonian store path provided. quitting.")
        sys.exit()
    graph_path = ham_path + "/graph_4_2_1.pickle"
    if os.path.exists(graph_path):
        print("[hamiltonian_entry_point] graph_4_2_1.pickle exists")
        sys.exit()
    else:
        print("[hamiltonian_entry_point] no file found, using this as file path: ", graph_path)
    ham_list = graph_hamiltonian(4,2,1)
    pickle_hamiltonian(graph_path, ham_list)
