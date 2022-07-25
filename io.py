import os
import pickle
from datetime import datetime

def dump_data(data):
    file_path = os.getenv("SCRATCH")
    if file_path[-1] != '/':
        file_path += "/"
    dt = datetime.now()
    file_path += dt.strftime("%m_%d_%H_%M")
    file_path += ".pickle"
    pickle(data, open(file_path, 'wb'))
