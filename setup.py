#!/usr/bin/env python

from utils import *
from compilers import *
from tests import *
import numpy as np
import sys
import pickle
import os
import shutil

def setup_entry_point():
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        print("[setup_entry_point] No output directory given, using SCRATCH")
        scratch_path = os.getenv("SCRATCH")
        if type(scratch_path) != type("string"):
            print("[setup_entry_point] No directory given and no scratch path")
            sys.exit()
        output_dir = scratch_path
    if output_dir[-1] != '/':
        output_dir += '/'
    print("[setup] base directory: ", output_dir)
    hamiltonian_file_path = setup_manage_hamiltonians(output_dir)
    settings_file_path = setup_manage_settings(output_dir)
    prep_launchpad(output_dir, settings_file_path, hamiltonian_file_path)

# RETURNS : path to selected hamiltonian.pickle file
def setup_manage_hamiltonians(base_dir):
    if os.path.exists(base_dir + "hamiltonians"):
        print("[setup] found existing hamiltonians directory")
    else:
        os.mkdir(base_dir + "hamiltonians")
        print("[setup] created hamiltonian directory at: ", base_dir + "hamiltonians")
    hamiltonian_base = base_dir + "hamiltonians/"
    hamiltonians = [f for f in os.listdir(hamiltonian_base) if os.path.isfile(hamiltonian_base + f)]
    hamiltonians.sort()
    print("[setup] found the following hamiltonian files")
    for i in range(len(hamiltonians)):
        print("(" + str(i + 1) +") ", hamiltonians[i])
    user_input = input('[setup] which hamiltonian would you like to use? >')
    try:
        return hamiltonian_base + hamiltonians[int(user_input) -1]
    except:
        print("[manage_hamiltonians] error: could not parse integer. Failing.")
        sys.exit()

# Assumes an input dictionary mapping strings of inputs to strings of values. converts them to proper data types for pickling.
def process_settings_to_save(unprocessed_settings):
    ret = {}
    for setting in MINIMAL_SETTINGS:
        if setting == EXPERIMENT_LABEL:
            val = unprocessed_settings.get(setting, DEFAULT_EXPERIMENT_LABEL)
            ret[setting] = val
        if setting == "verbose":
            val = unprocessed_settings.get(setting, "True")
            ret[setting] = bool(val)
        if setting == 'use_density_matrices':
            if unprocessed_settings.get(setting, "False") == "False":
                ret['use_density_matrices'] = False
            else:
                ret['use_density_matrices'] = True
        if setting == 't_start':
            val = unprocessed_settings.get(setting, "1e-4")
            ret[setting] = float(val)
        if setting == 't_stop':
            val = unprocessed_settings.get(setting, "1e-2")
            ret[setting] = float(val)
        if setting == 't_steps':
            val = unprocessed_settings.get(setting, "10")
            ret[setting] = int(val)
        if setting == 'partitions':
            val = unprocessed_settings.get(setting, ["first_order_trotter", "qdrift"])
            if type(val) == type([]):
                all_items_strings = True
                for v in val:
                    if type(v) != type("string"):
                        all_items_strings = False
                if all_items_strings == False:
                    print("[process_settings] found partition list with non string types. using default.") 
                    val = ["first_order_trotter", "qdrift"]
                ret[setting] = val
        if setting == 'infidelity_threshold':
            val = unprocessed_settings.get(setting, "0.05")
            ret[setting] = float(val)
        if setting == 'num_state_samples':
            val = unprocessed_settings.get(setting, "5")
            ret[setting] = int(val)
        if setting == 'base_directory':
            val = unprocessed_settings.get(setting, ".")
            ret[setting] = val
        if setting == 'test_type':
            val = unprocessed_settings.get(setting, "gate_cost")
            ret[setting] = val
        if setting == 'mc_samples':
            print("in mc_samples????")
            val = unprocessed_settings.get(setting, '1000')
            ret[setting] = int(val)
    return ret

# RETURNS : path to the final configured settings file. 
# GUARANTEES THAT EACH PICKLED SETTINGS FILE HAS MINIMUM SETTINGS WITH SANE DEFAULTS. 
def setup_manage_settings(base_dir):
    if os.path.exists(base_dir + "settings"):
        print("[setup] found existing settings directory")
    else:
        os.mkdir(base_dir + "settings")
        print("[setup] created settings directory at: ", base_dir + "settings")
    settings_base = base_dir + "settings/"
    settings_files = [f for f in os.listdir(settings_base) if os.path.isfile(settings_base + f)]
    settings_files.sort()
    print("[setup] found the following settings files")
    for ix in range(len(settings_files)):
        print("(" + str(ix + 1) + ")", settings_files[ix])
    try:
        index = int(input("[setup] enter a number or leave empty for new: "))
        settings_file = settings_files[index - 1]
    except:
        settings_file = ''
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
    for k,v in settings.items():
        print(k, "= ", v)

    need_to_set = ["experiment_label", "verbose", 'use_density_matrices','t_start', 't_stop', 't_steps', 'partitions', 'infidelity_threshold']
    need_to_set += ['num_state_samples', 'base_directory', 'test_type']
    print("[setup] Make sure the followings keys are set: ", need_to_set)
        
    print("[setup] you can modify a setting by writing \'setting=value\' after the \'>\' prompt. Only use one = sign. Enter q to quit.")
    for _ in range(100):
        user_input = input("> ")
        if user_input == "q":
            # TODO: Need to check that each required setting has been set.
            break
        elif user_input.startswith("clear"):
            try:
                var = user_input.split(" ")[-1]
                del settings[var]
            except:
                print("usage is 'clear <varname>' to clear, or you tried to clear a variable is not in settings.")
                continue
        elif user_input == "partitions":
            print("How many partitions?")
            num_partitions = input("> ")
            partitions = []
            try:
                num_partitions = int(num_partitions)
            except:
                print("try again with integer.")
                continue
            for i in range(num_partitions):
                p = input("enter partition number " + str(i + 1) + " > ")
                partitions.append(p)
            settings["partitions"] = partitions
        else:
            try:
                key, val = user_input.split("=")
                settings[key] = val
            except:
                print("[setup] incorrect input. format is \'key=val\'. Only use one = sign.")
    processed_settings = process_settings_to_save(settings)
    print("[setup] configuration completed. final settings:")
    print("*" * 60)
    for k,v in processed_settings.items():
        print(k, "=", v)
    save_to_new = input("[setup] Enter the filename (without \'.pickle\' extension) you'd like to save to (empty keeps name and rewrites). WARNING - can overwrite existing settings: ")
    if save_to_new == "":
        print("[setup] overwriting existing file: ", settings_file)
        save_file = base_dir + "settings/" + settings_file
    else:
        save_file = base_dir + "settings/" + save_to_new + ".pickle"
        print("[setup] saving to:", save_file)
    try:
        pickle.dump(processed_settings, open(save_file, 'wb'))
    except:
        print("[setup] error while pickling.")
    print("[setup] settings completed.")
    return save_file

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

if __name__ == "__main__":
    setup_entry_point()
