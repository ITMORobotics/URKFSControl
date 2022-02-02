import numpy as np
import json

def read_poses_json(filename, key_name, key_absrel) -> np.ndarray:
    with open(filename) as pose_data:
        data = json.load(pose_data)

    return np.reshape(data[key_name][key_absrel], [4,4])
