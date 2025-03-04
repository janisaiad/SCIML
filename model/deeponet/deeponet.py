import numpy as np
import tensorflow as tf


class DeepONet(tf.keras.Model):
    ### DeepONet class ###
    def __init__(self, hyper_params: dict, regular_params: dict): 
        """
        regular_params: dict
            - internal_model: tensorflow model for the internal basis-coefficient learning function, (input functions at grid points) R^d_p -> R^d_v (coefficients)
            - external_model: tensorflow model for the basis learning function, R^d_v -> R^d_v
            
        
        hyper_params: dict
            ### Model parameters ###
            - d_p: dimension of the encoder space for input function, number of grid points in the encoder
            - d_V: dimension of the decoder space for output function, number of basis functions to be learned
            
            ### Training parameters ###
            - learning_rate: learning rate for the optimizer
            - optimizer: optimizer for the model
            - n_layers: number of layers in the network
            - n_epochs: number of epochs to train the model
            - batch_size: batch size for the training
            - verbose: verbosity for the training
            - loss_function: loss function for the model
            
        Remarks : regular params have their own (hyper)parameters, which are not passed to the model, it is constructed in the build_model function
        """
        self.params = {
            "hyper_params": hyper_params,
            "regular_params": regular_params
        }
        
        self.internal_model = None
        self.external_model = None
        self.build_model()
        

    def build_model(self):
        
        # build the model using regular params
        self.internal_model = self.regular_params["internal_model"]
        self.external_model = self.regular_params["external_model"]
        
        
        pass
    
    def predict(self,mu:tf.Tensor,x:tf.Tensor): # mu is the input function, x is the pointwise evaluation points
        return tf.tensordot(self.internal_model(mu),self.external_model(x),axes=1) # just dot product
    
    
    def fit(self):
        
        # get the data
        mu,x,y = self.get_data()
        
        # fit the model
        self.internal_model.fit(mu,y)
        self.external_model.fit(x,y)
        
        pass
    
    def save(self,save_path:str):
        tf.saved_model.save(self,self.save_path)

    def load_to_gpu(self):
        self.internal_model.to_gpu()
        self.external_model.to_gpu()
    
    