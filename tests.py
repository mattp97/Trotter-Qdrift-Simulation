import sys

from utils import *
from compilers import *
import numpy as np

import pickle
import os
import shutil
import matplotlib.pyplot as plt

INFIDELITY_TEST_TYPE = "infidelity"
TRACE_DIST_TEST_TYPE = "trace_distance"
GATE_COST_TEST_TYPE = "gate_cost"
CROSSOVER_TEST_TYPE = "crossover"
LAUNCHPAD = "launchpad"

# For coordinating runs. 
class Experiment:
    def __init__(
        self,
        hamiltonian_list = [],
        output_directory="./",
        experiment_label="default",
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
        if output_directory[-1] != '/':
            self.output_directory = output_directory + '/'
        else:
            self.output_directory = output_directory
        self.experiment_label = experiment_label
    
    
    # TODO: implement multithreading?
    # TODO: How to handle probabilistic partitionings?
    def run(self):
        results = {}
        results["times"] = np.copy(self.times).tolist()
        for partition in self.partitions:
            if self.verbose:
                print("[Experiment.run] evaluating partition type:", partition)
            outputs = []
            heuristic = -1
            for t in results["times"]:
                if self.verbose:
                    print("[Experiment.run] evaluating time:", t)
                out = 0
                for _ in range(self.num_state_samples):
                    self.sim.randomize_initial_state()
                    if self.test_type == "infidelity":
                        inf_temp, _ = single_infidelity_sample(self.sim, t)
                        out += inf_temp
                    elif self.test_type == "gate_cost":
                        cost, iters = find_optimal_cost(self.sim, t, self.infidelity_threshold, heuristic=heuristic, verbose=self.verbose)
                        heuristic = iters
                        out += cost
                outputs.append(out / self.num_state_samples)
            results[partition] = outputs
        self.results = results
        pickle.dump(self.results, open(self.output_directory + "results.pickle", 'wb'))

    def pickle_settings(self, output_path):
        settings = {}
        settings["experiment_label"] = self.experiment_label
        settings["verbose"] = self.verbose
        settings["use_density_matrices"] = self.use_density_matrices
        settings["t_start"] = self.times[0]
        settings["t_stop"] = self.times[-1]
        settings["t_steps"] = len(self.times)
        settings["partitions"] = self.partitions
        settings["infidelity_threshold"] = self.infidelity_threshold
        settings["num_state_samples"] = self.num_state_samples
        settings["output_directory"] = self.output_directory
        settings["test_type"] = self.test_type
        pickle.dump(settings, open(output_path, 'wb'))

    # TODO: Replace dictionary access with get and defaults
    def load_settings(self, settings_path):
        settings = pickle.load(open(settings_path, 'rb'))
        self.experiment_label     = settings.get('experiment_label', "default_experiment_label")
        self.verbose              = settings.get("verbose", True)
        self.use_density_matrices = settings.get("use_density_matrices", False)
        self.times                = np.geomspace(settings.get("t_start", 1e-4), settings.get("t_stop", 1e-2), settings.get("t_steps", 10))
        self.partitions           = settings.get("partitions", ["first_order_trotter", "qdrift"])
        self.infidelity_threshold = settings.get("infidelity_threshold", 0.05)
        self.num_state_samples    = settings.get("num_state_samples", 5)
        self.test_type            = settings.get("test_type", GATE_COST_TEST_TYPE)
        # Drop self.output_directory? if we can find the settings then just output there

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


def find_launchpad():
    if len(sys.argv) == 3:
        base_dir = sys.argv[-1]
        if base_dir[-1] != '/':
            base_dir += '/'
    else:
        scratch_path = os.getenv("SCRATCH")
        if type(scratch_path) != type("string"):
            print("[find_launchpad] Error, could not find SCRATCH environment variable and no base directory provided")
            return None, None, None
        if scratch_path[-1] != '/':
            scratch_path += '/'
        base_dir = scratch_path
    launchpad = base_dir + LAUNCHPAD + '/'
    if os.path.exists(launchpad + 'hamiltonian.pickle'):
        ham_path = launchpad + 'hamiltonian.pickle'
    else:
        ham_path = None
    if os.path.exists(launchpad + 'settings.pickle'):
        settings_path = launchpad + 'settings.pickle'
    else:
        settings_path = None
    return ham_path, settings_path, launchpad
    
# RETURNS : path to selected hamiltonian.pickle file
def setup_manage_hamiltonians(base_dir):
    if os.path.exists(base_dir + "hamiltonians"):
        print("[setup] found existing hamiltonians directory")
    else:
        os.mkdir(base_dir + "hamiltonians")
        print("[setup] created hamiltonian directory at: ", base_dir + "hamiltonians")
    hamiltonian_base = base_dir + "hamiltonians/"
    hamiltonians = [f for f in os.listdir(hamiltonian_base) if os.path.isfile(hamiltonian_base + f)]
    print("[setup] found the following hamiltonian files")
    print(hamiltonians)
    user_input = input('[setup] which hamiltonian would you like to use? (Do not type \'.pickle\' extension): ')
    return hamiltonian_base + user_input + '.pickle'

# RETURNS : path to the final configured settings file. 
def setup_manage_settings(base_dir):
    if os.path.exists(base_dir + "settings"):
        print("[setup] found existing settings directory")
    else:
        os.mkdir(base_dir + "settings")
        print("[setup] created settings directory at: ", base_dir + "settings")
    settings_base = base_dir + "settings/"
    settings_files = [f for f in os.listdir(settings_base) if os.path.isfile(settings_base + f)]
    print("[setup] found the following settings files")
    print(settings_files)
    settings_file = input("[setup] enter a settings name to start modifying (do not type \'.pickle\' extension) or leave empty for new: ")
    settings_file += '.pickle'
    use_new_settings = (settings_file == "")
    settings = {}
    if use_new_settings == False:
        settings_path = base_dir + "settings/" + settings_file
        if os.path.exists(settings_path):
            loaded_settings = pickle.load(open(settings_path, 'rb'))
            settings.update(loaded_settings)
        else:
            print("[setup] could not find specified settings file: " + settings_path + " so using empty")

    print("[setup] These are the currently existing settings:")
    print(settings)

    need_to_set = ["experiment_label", "verbose", 'use_density_matrices','t_start', 't_stop', 't_steps', 'partitions', 'infidelity_threshold']
    need_to_set += ['num_state_samples', 'output_directory', 'test_type']
    print("[setup] Make sure the followings keys are set: ", need_to_set)
        
    print("[setup] you can modify a setting by writing \'setting=value\' after the \'>\' prompt. Only use one = sign. Enter q to quit.")
    for _ in range(100):
        user_input = input("> ")
        if user_input == "q":
            # TODO: Need to check that each required setting has been set.
            print("[setup] configuration completed. final settings:")
            print(settings)
            break
        try:
            key, val = user_input.split("=")
            settings[key] = val
        except:
            print("[setup] incorrect input. format is \'key=val\'. Only use one = sign.")
    save_to_new = input("[setup] Enter the filename you'd like to save to. WARNING - can overwrite existing settings, do not write \'.pickle\': ")
    pickle.dump(settings, open(base_dir + "settings/" + save_to_new + '.pickle', 'wb'))
    print("[setup] settings completed.")
    return base_dir + "settings/" + save_to_new + '.pickle'

def prep_launchpad(base_dir, settings_path, hamiltonian_path):
    if base_dir[-1] != "/":
        base_dir += '/'
    launchpad = base_dir + LAUNCHPAD
    if os.path.exists(launchpad) == False:
        os.mkdir(launchpad)
    print("[prep_launchpad] settings_path:", settings_path)
    print("[prep_launchpad] launchpad:", launchpad)
    shutil.copyfile(settings_path, launchpad + "/settings.pickle")
    try:
        shutil.copyfile(hamiltonian_path, launchpad + "/hamiltonian.pickle")
    except:
        print("[prep_launchpad] Could not copy hamiltonian.")
    if os.path.exists(launchpad + "/settings.pickle") and os.path.exists(launchpad + "/hamiltonian.pickle"):
        return True

def setup_entry_point():
    if len(sys.argv) == 3:
        output_dir = sys.argv[-1]
    else:
        print("[setup_entry_point] No output directory given, using SCRATCH")
        scratch_path = os.getenv("SCRATCH")
        if type(scratch_path) != type("string"):
            print("[setup_entry_point] No directory given and no scratch path")
        output_dir = scratch_path
    if output_dir[-1] != '/':
        output_dir += '/'
    print("[setup] base directory: ", output_dir)
    hamiltonian_file_path = setup_manage_hamiltonians(output_dir)
    settings_file_path = setup_manage_settings(output_dir)
    prep_launchpad(output_dir, settings_file_path, hamiltonian_file_path)

def compute_entry_point():
    ham_path, settings_path, launchpad = find_launchpad()
    if type(ham_path) != type("string") or type(settings_path) != type("string") or type(launchpad) != type("string"):
        print("[tests.py] Error: could not find hamiltonian.pickle or settings.pickle")
        sys.exit()
    ham_list = pickle.load(open(ham_path, 'rb'))
    settings = pickle.load(open(settings_path, 'rb'))
    print("[compute_entry_point] hamiltonian found with this many terms:", len(ham_list))
    print("#" * 50)
    print("settings found:")
    print(settings)
    working_dir = launchpad + 'output'
    if os.path.exists(working_dir) == False:
        os.mkdir(working_dir)
    exp = Experiment(output_directory=working_dir)
    exp.load_hamiltonian(ham_path)
    exp.load_settings(settings_path)
    exp.run()

# Analyze the results from the results.pickle file of a previous run
def analyze_entry_point():
    if len(sys.argv) == 3:
        base_dir = sys.argv[-1]
    else:
        print("[analyze_entry_point] No results.pickle file path provided. quitting.")
        sys.exit()
    try:
        results = pickle.load(open(base_dir + '/' + LAUNCHPAD + '/output/results.pickle', 'rb'))
    except:
        print("[analyze_entry_point] no results were found at: ", base_dir + '/' + LAUNCHPAD + '/output/results.pickle')
        sys.exit()
    print("results:")
    print(results)
    times = results["times"]
    for k,v in results.items():
        if k != "times":
            plt.plot(times, v, label=k)
    plt.show()

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


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup_entry_point()
    if sys.argv[1] == "compute":
        compute_entry_point()
    if sys.argv[1] == "analyze":
        analyze_entry_point()
    if sys.argv[1] == "hamiltonian":
        hamiltonian_entry_point()
