import numpy as np
import tensorflow as tf


class DeepONet(tf.keras.Model):
    ### DeepONet class ###
    def __init__(self, hyper_params: dict, regular_params: dict): 
        """
        regular_params: dict
            - internal_model: model for the internal basis-coefficient learning function, (input functions at grid points) R^d_p -> R^d_v (coefficients)
            - external_model: model for the basis learning function, R^d_v -> R^d_v
            
        
        hyper_params: dict
            ### Model parameters ###
            - d_p: dimension of the encoder space for input function, number of grid points in the encoder
            - d_V: dimension of the decoder space for output function, number of basis functions to be learned
            
            ### Training parameters ###
            - learning_rate: learning rate for the optimizer
            - optimizer: optimizer for the model
            - n_layers: number of layers in the network
        """
        self.params = {
            "hyper_params": hyper_params,
            "regular_params": regular_params
        }
        

    def build_model(self):
        
        # Build the 
        pass
    

    def fit(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
