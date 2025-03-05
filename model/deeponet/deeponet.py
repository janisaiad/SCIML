import numpy as np
import tensorflow as tf
from typing import Tuple
from tqdm import tqdm
import logging # janis sauce to debug
import os
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ROOT = os.getenv("PROJECT_ROOT")


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
        
        
        self.params = {
            "hyper_params": hyper_params,
            "regular_params": regular_params
        }
        
        self.hyper_params = hyper_params
        self.regular_params = regular_params
        
        self.internal_model = self.regular_params["internal_model"]
        self.external_model = self.regular_params["external_model"]
        
        
        self.d_p = hyper_params["d_p"]
        self.d_V = hyper_params["d_V"]
        self.learning_rate = hyper_params["learning_rate"] if "learning_rate" in hyper_params else 0.001
        self.optimizer = hyper_params["optimizer"] if "optimizer" in hyper_params else tf.optimizers.Adam(self.learning_rate) # AdamW to be added after
        self.n_epochs = hyper_params["n_epochs"] if "n_epochs" in hyper_params else 100
        self.batch_size = hyper_params["batch_size"] if "batch_size" in hyper_params else 32
        self.verbose = hyper_params["verbose"] if "verbose" in hyper_params else 1
        self.loss_function = hyper_params["loss_function"] if "loss_function" in hyper_params else tf.losses.MeanSquaredError()
        self.device = hyper_params["device"] if "device" in hyper_params else 'cpu'
        self.folder_path = None  
    
        
        
        
        logger.info(f"Model initialized with {self.n_epochs} epochs, {self.batch_size} batch size, {self.learning_rate} learning rate")
    
    
    @property
    def trainable_variables(self): # for user experience to get the trainable variables, doesn't required by tf
        return self.internal_model.trainable_variables + self.external_model.trainable_variables
        
        
        
        
    def predict(self, mu: tf.Tensor, x: tf.Tensor):
        with tf.device(self.device):
            # mu: [batch_size, d_p]
            # x: [batch_size, n_points, 2] ou [batch_size, nx, ny, 2]
            tf.print(mu.shape)
            tf.print(x.shape)
            coefficients = self.internal_model(mu)
    
            # reshape to get [batch_size, 40]
            coefficients = tf.reshape(coefficients, [tf.shape(coefficients)[0], -1])[:, :self.d_V]
            tf.print(coefficients.shape)
            basis_evaluation = self.external_model(x)  # [batch_size, n_points, 40]
            tf.print(basis_evaluation.shape)

            # Produit tensoriel
            output = tf.einsum('bi,bji->bj', coefficients, basis_evaluation)
            tf.print(output.shape)
            # coefficients: [batch_size, d_V]
            # basis: [batch_size, n_points, d_V]
        return output
    
    
    # in case you want to modify those models but not the other
    def set_internal_model(self,internal_model:tf.keras.Model): # for user experience to tune something
        self.internal_model = internal_model
        
    def set_external_model(self,external_model:tf.keras.Model): # for user experience to tune something
        self.external_model = external_model
        
        
        
        
        
    
    
    ### Data loading ### Be careful with the data format, we can have various sensor points for parameters : for instance a specified mu function can require to get many more points to compute the exact solution
    def get_data(self,folder_path:str) -> tuple[tf.Tensor,tf.Tensor]: # typing is important
        
        true_path = os.path.join(PROJECT_ROOT,folder_path)
        self.folder_path = true_path
        
        try: # error handling because it's critical
            
            mu_files = [np.load(os.path.join(true_path,f"mu_{i}.npy")) for i in tqdm(range(len(os.listdir(os.path.join(true_path)))//3), desc="Loading mu data")]
            x_files = [np.load(os.path.join(true_path,f"xs_{i}.npy")) for i in tqdm(range(len(os.listdir(os.path.join(true_path)))//3), desc="Loading x data")]
            sol_files = [np.load(os.path.join(true_path,f"sol_{i}.npy")) for i in tqdm(range(len(os.listdir(os.path.join(true_path)))//3), desc="Loading y data")]
            
            
            mus = tf.convert_to_tensor(mu_files, dtype=tf.float32)
            xs = tf.convert_to_tensor(x_files, dtype=tf.float32) 
            sol = tf.convert_to_tensor(sol_files, dtype=tf.float32)
            
        except:
            logger.error(f"Data not found in {true_path}")
            raise ValueError(f"Data not found in {true_path}")
        
        return mus, xs, sol
    
    
    
    # mandatory methods to be implemented for keras
    def call(self,mu:tf.Tensor,x:tf.Tensor)->tf.Tensor:
        return self.predict(mu,x)
    
    
    def compile(self): # apparently mandatory to compile the model
        self.optimizer = self.hyper_params["optimizer"] if "optimizer" in self.hyper_params else tf.optimizers.Adam(self.learning_rate)
        self.loss_function = self.hyper_params["loss_function"] if "loss_function" in self.hyper_params else tf.losses.MeanSquaredError()
    
    def build(self): # apparently also mandatory to build the model for tensorflow
        self.internal_model.build(input_shape=(None,self.d_p))
        self.external_model.build(input_shape=(None,self.d_V))
    
    
    
    ### managing model training methods ###
    def fit(self,device:str='cpu',mus=None,xs=None,sol=None,folder_path=None)->np.ndarray:
        
        # Get the functions and pointwise evaluation points
        mus, xs, sol = self.get_data(self.folder_path)
        loss_history = []
        with tf.device(device):
            dataset = tf.data.Dataset.from_tensor_slices((mus,xs,sol)) # batching the data with batch size
        
            dataset = dataset.batch(self.batch_size) # batching method from tensorflow
            # Training loop
            for epoch in tqdm(range(self.n_epochs),desc="Training progress"):
                for batch in dataset:
                    loss = self.train_step(batch)
                    loss_history.append(loss)
                logger.info(f"Epoch {epoch} completed")
            
        return loss_history
        
            
        
    def train_step(self,batch:tuple[tf.Tensor,tf.Tensor,tf.Tensor])->None:
        mu,x,sol = batch
        
        sol = tf.reshape(sol, [tf.shape(sol)[0], -1])[:, :self.d_V]
        with tf.GradientTape() as tape: # gradient tape to compute the gradients
            y_pred = self.predict(mu,x)
            loss = self.loss_function(y_pred,sol)
            
        # Backprop
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables)) # apply the gradients to the model, which means updating the weights
        
        return loss
            
            
            
    ## managing model saving methods        
    def save(self,save_path:str):  # error handling because it's also critical out there, we save a tensorflow model as a keras file
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        
        try:
            tf.saved_model.save(self,save_path)
        except:
            logger.error(f"Failed to save model in {save_path}")
            raise ValueError(f"Failed to save model in {save_path}")
        


    def load_weights(self,save_path:str): # just loading some other weights if we want to compare, but not the entire model
        if not os.path.exists(save_path):
            logger.error(f"Weights not found in {save_path}")
            raise ValueError(f"Weights not found in {save_path}")
        
        self.load_weights(save_path)
        
    def save_weights(self,save_path:str): 
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        
        self.save_weights(save_path)
