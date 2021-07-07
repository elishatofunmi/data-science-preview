mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# disables all tensorflow logs/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf




class Grad_descent(x,y,theta,m, learning_rate, epochs):
    # initialization of constants and variables
    self.X = tf.constant(x, dtype = tf.float32, name = 'x_tensor')
    self.y= tf.constant(y.reshape(-1,1), dtype = tf.float32, name = 'y_tensor')
    self.theta = tf.Variable(theta.reshape(-1,1), name = 'theta')
    self.slope = tf.constant(m, dtype = tf.flaot32, name = 'gradient slope')
    self.learning_rate = tf.constant(learning_rate, dtype = tf.float32, name = 'learning_rate')
    
    # predict y from x and y input
    y_pred = tf.matmul(self.X, self.theta, name = 'predict_y')
    
    # compute error difference
    find_error = self.Error(y_pred, self.y)
    self.tensor_error = tf.Constant(find_error, name = 'error_difference')
    
    # compute MSE
    self.mean_square_error = self.MSE(self.tensor_error, name = 'mean_square_error')
    self.tensor_MSE = tf.Variable(self.mean_square_error, name = 'tensor MSE')
    
    # compute gradient
    self.compute_grad = self.Gradient(self.slope, self.X,self.tensor_error)
    self.tensor_grad = tf.Variable(self.compute_grad, name = 'gradient')
    
    
    # autodiff computation of gradient
    #gradients = tf.gradients(self.MSE, [theta])[0]
    
    
    # update theta
    training_op = tf.assign(self.theta, self.theta - self.learning_rate * self.tensor_grad)
    
    # initialize global variables
    init = tf.global_variables_initializer()
    
    # run session
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(epochs):
            if epoch% 100 == 0:
                print('Epoch', epoch, 'MSE = ', self.mean_square_error)
            sess.run(training_op)
        best_theta = self.theta.eval()
            
    
    def Gradient(m_val, x_, y_):
        gradient = m_val/ tf.matmul(tf.transpose(x_), y_)
        with tf.Session() as sess:
            grad_val = gradient.eval()
        return grad_val
    
    def Error(self, predicted_y, real_y):
        # define error value as an empty array
        error_val = np.empty([])
        Error = tf.math.subtract(real_y, predicted_y, name = 'error')
        with tf.Session() as sess:
            error_val = Error.eval()
        return error_val
    
    
    def MSE(self,error):
        # compute mean
        mean = tf.reduce_mean(tf.square(error), name = 'MSE')
        with tf.Session() as sess:
            mean_sess = mean.eval()
        return mean_sess
    
    
Class Opimizers(Grad_descent):
    def __init__(self):
        super().__init__()
        
    def gradient_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        training_op = optimizer.minimize(self.mse)
        # returns a tensor training_op
        return training_op
    
    
    def momentum_optimizer(self):
        optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9)
        training_op = optimizer.minimize(self.mse)
        # returns a tensor training_op
    
    

        