import numpy as np
import tensorflow as tf
from typing import Tuple
from tqdm import tqdm
import logging # janis sauce to debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os


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
            - learning_rate: learning rate for the optimizer, default is 0.001
            - optimizer: tensorflow optimizer for the model, default is Adam
            - n_epochs: number of epochs to train the model, default is 100
            - batch_size: batch size for the training, default is 32
            - verbose: verbosity for the training, default is 1
            - loss_function: tensorflow loss function for the model, default is MSE
            - device: device for inference and training the model on, default is 'cpu'
            
        Remarks : regular params have their own (hyper)parameters, which are not passed to the model, it is constructed in the build_model function
        """
        
        
        super().__init__()
        
        required_params = ["internal_model","external_model"]
        for param in required_params:
            if param not in regular_params:
                logger.error(f"Required parameter {param} not found in regular_params")
                raise ValueError(f"Required parameter {param} not found in regular_params")
        
        self.params = {
            "hyper_params": hyper_params,
            "regular_params": regular_params
        }
        
        
        self.internal_model = self.regular_params["internal_model"]
        self.external_model = self.regular_params["external_model"]
        
        self.learning_rate = hyper_params["learning_rate"] if "learning_rate" in hyper_params else 0.001
        self.optimizer = hyper_params["optimizer"] if "optimizer" in hyper_params else tf.optimizers.Adam(self.learning_rate) # AdamW to be added after
        self.n_epochs = hyper_params["n_epochs"] if "n_epochs" in hyper_params else 100
        self.batch_size = hyper_params["batch_size"] if "batch_size" in hyper_params else 32
        self.verbose = hyper_params["verbose"] if "verbose" in hyper_params else 1
        self.loss_function = hyper_params["loss_function"] if "loss_function" in hyper_params else tf.losses.MeanSquaredError()
        
        self.trainable_variables = self.internal_model.trainable_variables + self.external_model.trainable_variables
    
        
        
        
        logger.info(f"Model initialized with {self.n_epochs} epochs, {self.batch_size} batch size, {self.learning_rate} learning rate")
    
        
    def predict(self,mu:tf.Tensor,x:tf.Tensor): # mu is the input function, x is the pointwise evaluation points
        return tf.tensordot(self.internal_model(mu),self.external_model(x),axes=1) # just an easy dot product
    
    def set_internal_model(self,internal_model:tf.keras.Model): # for user experience to tune something
        self.internal_model = internal_model
        
    def set_external_model(self,external_model:tf.keras.Model): # for user experience to tune something
        self.external_model = external_model
    
    
    ### Data loading ### Be careful with the data format, we can have various sensor points for parameters : for instance a specified mu function can require to get many more points to compute the exact solution
    def get_data(self,folder_path:str) -> tuple[tf.Tensor,tf.Tensor]: # typing is important
        
        try: # error handling because it's critical
            mus = np.load(folder_path + "mus.npy")
            xs = np.load(folder_path + "xs.npy") 
            ys = np.load(folder_path + "ys.npy")    
        except:
            logger.error(f"Data not found in {folder_path}")
            raise ValueError(f"Data not found in {folder_path}")
        
        return mus, xs, ys
    
    def call(self,mu:tf.Tensor,x:tf.Tensor)->tf.Tensor:
        return self.predict(mu,x)
    
    
    def fit(self,device:str='cpu')->np.ndarray:
        
        # Get the functions and pointwise evaluation points
        mus, xs, ys = self.get_data()
        loss_history = []
        if device != 'cpu': # not the best way to do it, but it works
            mus = tf.convert_to_tensor(mus,dtype=tf.float32)
            xs = tf.convert_to_tensor(xs,dtype=tf.float32)
            ys = tf.convert_to_tensor(ys,dtype=tf.float32)
            self.load_to_gpu()
            
            dataset = tf.data.Dataset.from_tensor_slices((mus,xs,ys)) # batching the data with batch size
        else :
            dataset = tf.data.Dataset.from_tensor_slices((mus,xs,ys))
        
        dataset = dataset.batch(self.batch_size) # batching method from tensorflow
        # Training loop
        for epoch in tqdm(range(self.n_epochs),desc="Training progress"):
            for batch in dataset:
                loss = self.optimizer.train_step(batch)
                loss_history.append(loss)
            logger.info(f"Epoch {epoch} completed")
            
        return loss_history
        
            
        
    def train_step(self,batch:tuple[tf.Tensor,tf.Tensor,tf.Tensor])->None:
        mu,x,y = batch
        
        with tf.GradientTape() as tape: # gradient tape to compute the gradients
            y_pred = self.predict(mu,x)
            loss = self.loss_function(y_pred,y)
            
        # Backprop
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables)) # apply the gradients to the model, which means updating the weights
        
        return loss
            
            
    def save(self,save_path:str):  # error handling because it's also critical out there
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        
        try:
            tf.saved_model.save(self,save_path)
        except:
            logger.error(f"Failed to save model in {save_path}")
            raise ValueError(f"Failed to save model in {save_path}")
        

    def load_to_gpu(self):
        
        if tf.test.is_gpu_available():
            with tf.device('/GPU:0'):
                self.internal_model.to_gpu()
                self.external_model.to_gpu()
        else:
            logger.warning("No GPU available, training will be slow")
        
    
    