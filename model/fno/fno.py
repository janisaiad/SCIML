import numpy as np
import tensorflow as tf
from typing import Tuple
from tqdm import tqdm
import logging # janis sauce to debug
import os
import dotenv
import json


dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.getenv("PROJECT_ROOT")




import tensorflow as tf
import numpy as np

class LinearLayer(tf.keras.layers.Layer):
    # [batch, p_1, p_1, n_coords] -> [batch, p_1, p_1, n_modes]
    def __init__(self,n_modes:int,initializer:str='normal',device:str='GPU',p_1:int=50): # attention ici
        super().__init__()
        self.n_modes = n_modes
        self.initializer = initializer
        self.device = device
        self.p_1 = p_1
        
    def call(self,inputs:tf.Tensor)->tf.Tensor:
        with tf.device(self.device):
            if len(inputs.shape) == 3:
                return inputs * self.linear_weights # [batch, p_1, p_1, n_modes]
            else:
                raise ValueError(f"Expected shape [batch, p_1, p_1, n_coords], got {inputs.shape}")

    def build(self,input_shape:tf.TensorShape):
        # [p_1, p_1]
        self.linear_weights = self.add_weight(shape=(self.p_1,self.p_1,),initializer=self.initializer,trainable=True,name="linear_weights")
        

class FourierLayer(tf.keras.layers.Layer): # just a simple fourier layer with pointwise multiplication with a linear thing in parallel
    def __init__(self, n_modes: int, dim_coords: int, activation: str = "relu", kernel_initializer: str = "he_normal", device:str='GPU', linear_initializer:str='normal',p_1:int=20):
        super().__init__()
        self.n_modes = n_modes
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.linear_initializer = linear_initializer
        self.device = device
        self.dim_coords = dim_coords
        
        self.p_1 = p_1
        
        
    def build(self, input_shape: tf.TensorShape):
        # this is a 3d tensor of shape [n_modes,n_coords,n_coords]
        
        
        self.fourier_weights = self.add_weight(
            shape=(self.p_1,self.p_1,),
            initializer=self.kernel_initializer,
            trainable=True,
            name="fourier_weights"
        )
        
        
        self.linear_layer = LinearLayer(self.n_modes, self.linear_initializer, self.device,self.p_1)
        self.linear_layer.build(input_shape)
        super().build(input_shape)
        
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs est de taille [batch, p_2,p_2,1] - représente les valeurs de la fonction
        print("inputs shape",inputs.shape)
        with tf.device(self.device):
            casted_data = tf.cast(inputs, tf.complex64) # to real, [batch, p_2,p_2,1]
            function_fft = tf.signal.fft2d(casted_data) # [batch, p_2,p_2, n_modes]
            
            print("function_fft shape",function_fft.shape)
            # keep in mind fourier_weights is a tensor of shape [n_modes,dim_coords]

            fourier_casted = tf.cast(self.fourier_weights,tf.complex64)
            
            print("fourier_weights shape",fourier_casted.shape)
            try:
                function_fft = function_fft * fourier_casted  # with broadcasting
            except:
                print("function_fft shape",function_fft.shape)
                print("fourier_casted shape",fourier_casted.shape)
                raise ValueError("Shape mismatch")
            
            x_spatial = tf.signal.ifft2d(function_fft)
            x_spatial = tf.cast(x_spatial, tf.float32)
            z = self.linear_layer(inputs)
            print("x_spatial shape",x_spatial.shape)
            print("z shape",z.shape)
            return self.activation(x_spatial + z)

    
class FourierNetwork(tf.keras.Model): # we consider a network of fourier layers, ie concatenation of fourier layers
    def __init__(self, n_layers: int, n_modes: int, dim_coords: int, activation: str = "relu", kernel_initializer: str = "he_normal"):
        super().__init__()
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.fourier_layers = [FourierLayer(n_modes, dim_coords, activation, kernel_initializer) for _ in range(n_layers)]

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
            - dim_coords: dimension of the coordinates, default is 2
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
        
        self.dim_coords = fourier_params["dim_coords"]
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
    
    
    
    
    
    def predict(self, inputs: tf.Tensor):
        
        batch_size = tf.shape(inputs)[0]
        n_points = tf.shape(inputs)[1]
        with tf.device(self.device):
            #  first network
            features = self.first_network(inputs)  # [batch, p_1, p_1, 3] -> [batch, p_1, p_1, p_2]
            
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
        
        
    def get_data(self, folder_path: str):
        true_path = os.path.join(PROJECT_ROOT, folder_path)
        self.folder_path = true_path

        try:
            with open(os.path.join(true_path, "params.json"), "r") as f:
                params = json.load(f)
        
            nx = params["nx"]  # [scalar]
            ny = params["ny"]  # [scalar] 
            nt = params["nt"]  # [scalar]
            n_mu = params["n_mu"]  # [scalar]
        
            
            mu = np.load(os.path.join(true_path, "mu.npy"))  # [N, nx, ny, 1]
            sol = np.load(os.path.join(true_path, "sol.npy"))  # [N, nx, ny, nt]
            xs = np.load(os.path.join(true_path, "xs.npy"))  # [N, nx, ny, 2]

            time_index = self.hyper_params.get("index", -1)  # [scalar], this means that we take the last time step for the training in the worst case
                
            mu = tf.convert_to_tensor(mu, dtype=tf.float32)  # [N, nx, ny, 1]
            xs = tf.convert_to_tensor(xs, dtype=tf.float32)  # [n_mu, nx, ny, 2]
            sol = tf.convert_to_tensor(sol[:, time_index, :], dtype=tf.float32)  # [N, nx*ny]
            
            print("mu shape",mu.shape)
            inputs = tf.squeeze(mu,axis=-1)
            print("inputs shape",inputs.shape)
            #inputs = tf.concat([mu, xs], axis=-1)  # [N, nx, ny, 3]
            return inputs, sol
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise ValueError(f"Échec du chargement des données: {str(e)}")
    
    def get_data_partial(self, folder_path: str,alpha:float=0.01):
        true_path = os.path.join(PROJECT_ROOT, folder_path)
        self.folder_path = true_path

        try:
            with open(os.path.join(true_path, "params.json"), "r") as f:
                params = json.load(f)
        
            nx = params["nx"]  # [scalar]
            ny = params["ny"]  # [scalar] 
            nt = params["nt"]  # [scalar]
            n_mu = params["n_mu"]  # [scalar]
            
            mu_list = []
            sol_list = []
            xs_list = []
            
            for i in range(int(n_mu*alpha)):
                mu_list.append(np.load(os.path.join(true_path, f"mu/mu_{i}.npy")))  # [nx, ny, 1]
                sol_list.append(np.load(os.path.join(true_path, f"sol/sol_{i}.npy")))  # [nx, ny, nt]
                xs_list.append(np.load(os.path.join(true_path, f"xs/xs_{i}.npy")))  # [nx, ny, 2]
            
            mu = np.stack(mu_list, axis=0)  # [N, nx, ny, 1]
            sol = np.stack(sol_list, axis=0)  # [N, nx, ny, nt] 
            xs = np.stack(xs_list, axis=0)  # [N, nx, ny, 2]

            time_index = self.hyper_params.get("index", -1)  # [scalar], this means that we take the last time step for the training in the worst case
                
            mu = tf.convert_to_tensor(mu, dtype=tf.float32)  # [N, nx, ny, 1]
            xs = tf.convert_to_tensor(xs, dtype=tf.float32)  # [n_mu, nx, ny, 2]
            sol = tf.convert_to_tensor(sol[:, time_index, :], dtype=tf.float32)  # [N, nx*ny]
            
            print("mu shape",mu.shape)
            inputs = tf.squeeze(mu,axis=-1)
            print("inputs shape",inputs.shape)
            return inputs, sol
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise ValueError(f"Échec du chargement des données: {str(e)}")

    
    # mandatory, we have an inference point of view
    def call(self,inputs:tf.Tensor)->tf.Tensor:
        return self.predict(inputs)
    
    def compile(self): # apparently mandatory to compile the model
        self.optimizer = self.hyper_params["optimizer"] if "optimizer" in self.hyper_params else tf.optimizers.Adam(self.learning_rate)
        self.loss_function = self.hyper_params["loss_function"] if "loss_function" in self.hyper_params else tf.losses.MeanSquaredError()
    
    
    
    def build(self) -> None:
        self.first_network.build(input_shape=(None, self.p_1, self.p_1))
        self.last_network.build(input_shape=(None, self.p_1, self.p_1))
        self.build_fourier_network()

    
    
    def build_fourier_network(self) -> None:  # main function to build the fourier network and allow CuFFT for fourier efficient training (with my rtx 4060)
        if self.fourier_network is None:
            self.fourier_network = FourierNetwork(dim_coords=self.dim_coords,n_layers=self.fourier_params["n_layers"],n_modes=self.fourier_params["n_modes"],activation=self.fourier_params["activation"],kernel_initializer=self.fourier_params["kernel_initializer"])
            
        self.fourier_network.build(input_shape=(None,self.p_2)) # it takes a tuple 
    
    
    
    ### managing model training methods ###
    def fit(self,device:str='GPU',inputs=None,sol=None)->np.ndarray:
        
        inputs, sol = self.get_data(self.folder_path)
        loss_history = []
        with tf.device(device):
            dataset = tf.data.Dataset.from_tensor_slices((inputs,sol)) # batching the data with batch size
        
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
        inputs, sol = batch
        
        with tf.GradientTape() as tape:
            
            y_pred = self.predict(inputs)
            
           
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
