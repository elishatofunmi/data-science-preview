import numpy as np
import pandas as pd
import tensorflow as tf


class Uniform_distribution:
    def __init__(self):
        return
    
    def logistic(n_inputs, n_outputs):
        return np.square(6/(n_inputs + n_outputs))
    
    def ReLU(n_inputs, n_outputs):
        return np.dot(np.square(2), np.square(6/(n_inputs, n_outputs)))
    
    def hyperbolic_tangent(n_inputs, n_outputs):
        return 4 * np.square(6/(n_inputs + n_outputs))
    
    
    
class Normal_distribution:
    def __init__(self):
        return
    
    
    def logistic(n_inputs, n_outputs):
        return np.square(2/(n_inputs + n_outputs))
    
    def hyperbolic_tangent(n_inputs, n_outputs):
        return 4 * np.square(2/(n_inputs + n_outputs))
    
    
    def ReLU(n_inputs , n_outputs):
        return np.dot(np.square(2), np.2/(n_inputs + n_outputs))
    
    
class ELU:
    def __init__(self, learning_param, z):
        self.apply_elu(learning_param, z)
        return
    
    def apply_elu(learning_param, z):
        if z < 0:
            value = learning_param * (np.exp(z)- 1)
        else:
            value = z
        return value
    
    def leaky_relu(z,name = None):
        return tf.maximum(0.01 * z , z, name = name)
    
    
    
    