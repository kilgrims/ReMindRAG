import numpy as np
import math

def cosine_similarity(a:np.array, b:np.array) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

def edge_weight_coefficient(vector_mode:float):
    argument = (math.pi / 2) * vector_mode
    return (2 / math.pi) * math.cos(argument)