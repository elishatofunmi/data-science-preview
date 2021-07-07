import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# disables all tensorflow logs/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf



class LinearRegression(x,y):
    def __init__(self,x,y):
        # initialize the variables and constants
        self.x = x
        self.y = y
        return

    def find_theta(self):
        var_x = tf.constant(self.x, dtypes = tf.float32, name = 'var_x')
        var_y = tf.constant(self.y.reshape(-1,1), dtypes = tf.float32, name = 'var_y')
        x_transpose = tf.transpose(var_x)
        theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(x_transpose, var_x)),x_transpose),
                         var_y)
        
        with tf.Session() as sess:
            theta_val = theta.eval()
            
        return theta_val
    
    def hypothesis(self,x,m,y):
        y_val = 0
        m_val = tf.Constant(m, dtypes = tf.float32, name = 'slope_m')
        y_int = tf.Constant(y, dtypes = tf.float32, name = 'y_intercept')
        x_var = tf.Variable(x, dtypes = tf.float32, name = 'input_x')
        y = tf.add(tf.matmul(m_val,x_var),y_int)
        
        with tf.Session() as sess:
            y_val = y.eval()
        return y_val
    
    
    
    
    

        
    