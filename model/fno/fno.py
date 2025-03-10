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




import tensorflow as tf
import numpy as np


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self,n_modes:int,initializer:str='normal',device:str='GPU'):
        super().__init__()
        self.n_modes = n_modes
        self.initializer = initializer
        self.device = device
    
    def call(self,inputs:tf.Tensor)->tf.Tensor:
        with tf.device(self.device):
            if len(inputs.shape) == 2:
                return inputs[:,:self.n_modes]*self.linear_weights
            else:
                raise ValueError(f"Format des inputs incorrect. Attendu [batch, n_points, dim_coords], reçu {inputs.shape}")

    def build(self,input_shape:tf.TensorShape):
        self.linear_weights = self.add_weight(shape=(self.n_modes,),initializer=self.initializer,trainable=True,name="linear_weights")
        
        
        

class FourierLayer(tf.keras.layers.Layer): # just a simple fourier layer with pointwise multiplication with a linear thing in parallel
    def __init__(self, n_modes: int, activation: str = "relu", kernel_initializer: str = "he_normal", device:str='GPU', linear_initializer:str='normal'):
        super().__init__()
        self.n_modes = n_modes
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.linear_initializer = linear_initializer
        self.device = device
    
    def build(self, input_shape: tf.TensorShape):
        
        self.fourier_weights = self.add_weight(
            shape=(self.n_modes,),
            initializer=self.kernel_initializer,
            trainable=True,
            name="fourier_weights"
        )
        
        
        self.linear_layer = LinearLayer(self.n_modes, self.linear_initializer, self.device)
        self.linear_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor: # the fft is spatial so the last coordinate is the time and should not be taken into account
        with tf.device(self.device):
        
            x = tf.rfftnd(tf.cast(inputs, tf.complex64),axes=1,fft_length=self.n_modes) # here the dimension of x after this operation is [batch, n_points, dim_coords] with dim_coords = 2
            # x = x[:, :self.n_modes,:] # because  it becomes unuseful
            x = x * tf.cast(self.fourier_weights, tf.complex64)
            x = tf.irfftnd(x,axes=1,fft_length=self.n_modes) # keep in mind that here the dimension is [batch, n_points, dim_coords]
            # to be made complex after
            
        
            z = self.linear_layer(inputs) #  the dimension is [batch, n_points, dim_coords]
            
            
            return self.activation(x + z)

    
class FourierNetwork(tf.keras.Model): # we consider a network of fourier layers, ie concatenation of fourier layers
    def __init__(self, n_layers: int, n_modes: int, activation: str = "relu", kernel_initializer: str = "he_normal"):
        super().__init__()
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.fourier_layers = [FourierLayer(n_modes, activation, kernel_initializer) for _ in range(n_layers)]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        for layer in self.fourier_layers:
            inputs = layer(inputs)
        return inputs
    
    def build(self, input_shape: tf.TensorShape):
        for layer in self.fourier_layers:
            layer.build(input_shape)
            


    
class FNO(tf.keras.Model):
    ### FNO class ###
    def __init__(self, hyper_params: dict, regular_params: dict,fourier_params:dict): 
        """
        regular_params: dict
                - first_network : tensorflow model for the encoder before fourier layers, (input functions at grid points) R^p_1 -> R^p_2 (coefficients)
                - last_network: tensorflow model for the decoder after fourier layers, R^p_2 -> R^p_3
        
        fourier_params: dict
            - n_layers: number of fourier layers in the model, default is 3
            - n_modes: number of modes in the fourier layers, default is 16, for future implementation I will make it a tuple, ie a number for each layer
            - activation: activation function for the fourier layers, default is ReLU
            - kernel_initializer: kernel initializer for the fourier layers, default is HeNormal
            
            fourier_network: tensorflow model for the fourier layers, default is None, built in the build_fourier_network function
            
        hyper_params: dict
            ### Model parameters ###
            - p_1: dimension of the input space for input function, number of grid points in the encoder
            - p_2: dimension of the encoder space for internal fourier layers, number of basis functions to be learned
            - p_3: dimension of the decoder space for output function, number of modes in the fourier layers
            - index: time index for the training, default is 1
            
            
            
            ### Training parameters ###
            - learning_rate: learning rate for the optimizer, default is 0.001
            - optimizer: tensorflow optimizer for the model, default is Adam
            - n_epochs: number of epochs to train the model, default is 100
            - batch_size: batch size for the training, default is 32
            - verbose: verbosity for the training, default is 1
            - loss_function: tensorflow loss function for the model, default is MSE
            - device: device for inference and training the model on, default is 'cpu'
            
        remarks : regular params have their own (hyper)parameters, which are not passed to the model, it is constructed in the build_model function
        """
        
        super().__init__()
        
        required_params = ["first_network","last_network"]
        for param in required_params:
            if param not in regular_params:
                logger.error(f"Required parameter {param} not found in regular_params")
        
        self.params = {
            "hyper_params": hyper_params,
            "regular_params": regular_params,
            "fourier_params": fourier_params
        }
        
        self.hyper_params = hyper_params
        self.regular_params = regular_params
        self.fourier_params = fourier_params
        
        self.first_network = self.regular_params["first_network"] # its a tensorflow model
        self.last_network = self.regular_params["last_network"] # same
        self.fourier_network = self.fourier_params["fourier_network"] if "fourier_network" in self.fourier_params else None # we can specify it or build it after
        
        self.p_1 = hyper_params["p_1"]
        self.p_2 = hyper_params["p_2"]    
        self.p_3 = hyper_params["p_3"]
        
        
        
        # optimisation stuff
        self.learning_rate = hyper_params["learning_rate"] if "learning_rate" in hyper_params else 0.001
        self.optimizer = hyper_params["optimizer"] if "optimizer" in hyper_params else tf.optimizers.Adam(self.learning_rate) # AdamW to be added after
        self.n_epochs = hyper_params["n_epochs"] if "n_epochs" in hyper_params else 100
        self.batch_size = hyper_params["batch_size"] if "batch_size" in hyper_params else 32
        self.verbose = hyper_params["verbose"] if "verbose" in hyper_params else 1
        self.loss_function = hyper_params["loss_function"] if "loss_function" in hyper_params else tf.losses.MeanSquaredError()
        self.device = hyper_params["device"] if "device" in hyper_params else 'cpu'
        self.folder_path = None  
    
        self.output_shape = hyper_params["output_shape"] if "output_shape" in hyper_params else None
        
        
        self.build() # most important to build the model
        
        
        logger.info(f"Model initialized with {self.n_epochs} epochs, {self.batch_size} batch size, {self.learning_rate} learning rate")
    
    @property
    def trainable_variables(self): # for user experience to get the trainable variables, doesn't required by tf
        return self.first_network.trainable_variables + self.last_network.trainable_variables+self.fourier_network.trainable_variables
    
    
    
    
    
    def predict(self, mu: tf.Tensor, x: tf.Tensor):
       
        
        batch_size = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        with tf.device(self.device):
            #  first network
            features = self.first_network(mu)  # [batch, p_1] -> [batch, p_2]
            
            #  fourier network,  [batch, n_points, p_2] -> [batch, p_3]
            fourier_features = self.fourier_network(features)  # [batch, p_2]

            # last network, [batch, n_points, p_3] -> [batch, n_points]
            output_tensor = self.last_network(fourier_features)
            
            return output_tensor
    
    
    
    
    # in case you want to modify those models but not the other
    def set_first_network(self,first_network:tf.keras.Model): # for user experience to tune something
        self.first_network = first_network
        
    def set_last_network(self,last_network:tf.keras.Model): # for user experience to tune something
        self.last_network = last_network
        
    def set_fourier_network(self,fourier_network:tf.keras.Model): # for user experience to tune something
        self.fourier_network = fourier_network
        
        
        
        
    ### Data loading ### Be careful with the data format, we can have various sensor points for parameters : for instance a specified mu function can require to get many more points to compute the exact solution
    def get_data(self, folder_path: str):
        true_path = os.path.join(PROJECT_ROOT, folder_path)
        self.folder_path = true_path
        
        try:
           
            mu_files = [np.load(os.path.join(true_path, f"mu_{i}.npy")) for i in tqdm(range(len(os.listdir(os.path.join(true_path)))//3), desc="Loading mu data")]
            x_files = [np.load(os.path.join(true_path, f"xs_{i}.npy")) for i in tqdm(range(len(os.listdir(os.path.join(true_path)))//3), desc="Loading x data")]
            sol_files = [np.load(os.path.join(true_path, f"sol_{i}.npy")) for i in tqdm(range(len(os.listdir(os.path.join(true_path)))//3), desc="Loading y data")]
            
           
            mus = tf.convert_to_tensor(mu_files, dtype=tf.float32)
            mus = tf.reshape(mus, [tf.shape(mus)[0], -1])  # [batch, 400]
            
            
            xs = tf.convert_to_tensor(x_files, dtype=tf.float32)
            if len(xs.shape) > 2:
                
                n_samples = xs.shape[0]
                xs = tf.reshape(xs, [n_samples, -1, xs.shape[-1]])
                
                
                time_index = self.hyper_params.get("index", 0)
                n_points_per_time = 400  # 20×20 points
                
                
                start_idx = time_index * n_points_per_time
                end_idx = start_idx + n_points_per_time
                
                
                xs_at_time = xs[:, start_idx:end_idx, :]
                
                
                xs_spatial = xs_at_time[:, :, :2]  # [batch, 400, 2]
                xs = xs_spatial
            
            
            sol = tf.convert_to_tensor(sol_files, dtype=tf.float32)
            
            
            sol = tf.reshape(sol, [tf.shape(sol)[0], 20, 400])
            
            
            time_index = self.hyper_params.get("index", 0)
            sol = sol[:, time_index, :]  # [batch, 400]
            
            print(f"Dimensions après extraction du temps {time_index}:")
            print(f"mus: {mus.shape}")
            print(f"xs: {xs.shape}")
            print(f"sol: {sol.shape}")
            
            return mus, xs, sol
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise ValueError(f"Échec du chargement des données: {str(e)}")
    

    
    # mandatory, we have an inference point of view
    def call(self,mu:tf.Tensor,x:tf.Tensor)->tf.Tensor:
        return self.predict(mu,x)
    
    def compile(self): # apparently mandatory to compile the model
        self.optimizer = self.hyper_params["optimizer"] if "optimizer" in self.hyper_params else tf.optimizers.Adam(self.learning_rate)
        self.loss_function = self.hyper_params["loss_function"] if "loss_function" in self.hyper_params else tf.losses.MeanSquaredError()
    
    
    
    
    def build(self) -> None: # apparently also mandatory to build the model for tensorflow, typed to understand
        self.first_network.build(input_shape=(None,self.p_1))
        self.last_network.build(input_shape=(None,self.p_2))
        self.build_fourier_network()
    
    
    
    def build_fourier_network(self) -> None:  # main function to build the fourier network and allow CuFFT for fourier efficient training (with my rtx 4060)
        if self.fourier_network is None:
            self.fourier_network = FourierNetwork(self.fourier_params["n_layers"],self.fourier_params["n_modes"],self.fourier_params["activation"],self.fourier_params["kernel_initializer"])
            
        self.fourier_network.build(input_shape=(None,self.p_2))
    
    
    
    ### managing model training methods ###
    def fit(self,device:str='GPU',mus=None,xs=None,sol=None)->np.ndarray:
        
        mus, xs, sol = self.get_data(self.folder_path)
        loss_history = []
        with tf.device(device):
            dataset = tf.data.Dataset.from_tensor_slices((mus,xs,sol)) # batching the data with batch size
        
            dataset = dataset.batch(self.batch_size) # batching method from tensorflow
       
            for epoch in tqdm(range(self.n_epochs),desc="Training progress"):
                for batch in dataset:
                    loss = self.train_step(batch)
                    loss_history.append(loss)
                logger.info(f"Epoch {epoch} completed")
            
        return loss_history
        
    def train_step(self, batch: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Étape d'entraînement simplifiée - minimum de code
        """
        mu, x, sol = batch
        
        with tf.GradientTape() as tape:
            
            y_pred = self.predict(mu, x)
            
           
            loss = self.loss_function(y_pred, sol)
        
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
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
