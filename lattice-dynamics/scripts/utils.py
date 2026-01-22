import yaml
import numpy as np

def load_config(path="config/parameters.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_k(a, points):
    return np.linspace(0, np.pi/a, points)
