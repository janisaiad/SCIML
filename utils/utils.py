import tensorflow as tf
import numpy as np


def fourier_building(fourier_params: dict, dim: int) -> tf.keras.Model:   
    # fourier pointwise multiplication
    fourier_weights = tf.Variable(
        tf.keras.initializers.get(fourier_params.get("kernel_initializer", "he_normal"))(shape=(fourier_params["n_modes"],),dtype=tf.float32),trainable=True,name="fourier_weights")
    
    inputs = tf.keras.Input(shape=(dim,))
    
    # fft to be trainable and multiplication
    x = tf.signal.fft(tf.cast(inputs, tf.complex64))
    x = x[:, :fourier_params["n_modes"]]  # Keep only n_modes frequencies    
    x = x * tf.cast(fourier_weights, tf.complex64)
    x = tf.signal.ifft(x)
    x = tf.cast(tf.math.real(x), tf.float32)  # Take real part

    # downward = linear + bias
    outputs = tf.keras.layers.Dense(
        dim,
        activation=tf.keras.activations.linear,
        use_bias=True
    )(x)
    
    # final model is sum of all and activation
    outputs = tf.keras.layers.Add()([outputs, x])
    outputs = tf.keras.layers.Activation(activation=fourier_params["activation"])(outputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
