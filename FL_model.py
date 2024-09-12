import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import f1_score, roc_auc_score, auc, precision_score, recall_score


import os
import warnings
warnings.filterwarnings("ignore")

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed = 42
fix_seed(seed)

# input_shape = (38,) # for RRT
# input_shape = (23,) # for mimic
input_shape = (21,) # for eicu
# latent_dim = 16
# hidden_units = [128, 64, 32]
# epsilon = 1e-7

############### Loss and metrics ###############

def f1_metric(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    return tf.py_function(
        lambda y_true, y_pred_binary: f1_score(
            y_true.numpy(), y_pred_binary.numpy(), average='weighted'
        ),
        [y_true, y_pred_binary],
        tf.float64
    )

def precision_metric(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    return tf.py_function(
        lambda y_true, y_pred_binary: precision_score(
            y_true.numpy(), y_pred_binary.numpy(), average='weighted'
        ),
        [y_true, y_pred_binary],
        tf.float64
    )

def recall_metric(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    return tf.py_function(
        lambda y_true, y_pred_binary: recall_score(
            y_true.numpy(), y_pred_binary.numpy(), average='weighted'
        ),
        [y_true, y_pred_binary],
        tf.float64
    )

class SupervisedContrastiveLoss(keras.losses.Loss):

    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(encoder):
    inputs = Input(shape=input_shape)
    features = encoder(inputs)
    outputs = Dense(16, activation='relu')(features)
    model = Model(inputs=inputs, outputs=outputs)
    return model



#### DNN - AE ####

def create_encoder(input_shape, latent_dim, hidden_units=[128, 64, 32], activation='relu'):
    """
    Create a neural network model using the functional API.

    Parameters:
    - input_shape: Tuple, the shape of the input data.
    - latent_dim: Integer, the dimension of the latent representation.
    - hidden_units: List, the number of units in each hidden layer.
    - activation: String, the activation function to use in hidden layers.

    Returns:
    - model: Keras Model, the created neural network model.
    """

    # Define input layer
    inputs = Input(shape=input_shape)

    # Hidden layers
    x = inputs
    for units in hidden_units:
        x = Dense(units, activation=activation)(x)

    # Output layer representing the latent features
    latent_representation = Dense(latent_dim, activation='relu', name='latent_representation')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=latent_representation)

    return model



# create classification decoder

def create_decoder(encoder, trainable=True):
    for layer in encoder.layers:
        layer.trainable = trainable
    
    inputs = Input(shape=input_shape)
    x = encoder(inputs)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001),  metrics=['accuracy', f1_metric, precision_metric, recall_metric])
    # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001),  metrics=['accuracy'])
    
    return model

# create single classification head
def create_clf_head(input_shape):
    inputs = Input(shape=input_shape)
    x = Dropout(0.2)(inputs)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001),  metrics=['accuracy', f1_metric, precision_metric, recall_metric])

    return model

### DNN - VAE ###

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# build encoder
def create_vae_encoder(input_dim, latent_dim, hidden_dim, activation='tanh'):
    inputs = Input(shape=input_dim)

    # hidden layers
    x = inputs
    for units in hidden_dim:
        x = Dense(units, activation=activation)(x)

    # mean and variance
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])

    z = Dense(16, activation='relu')(z)

    encoder = Model(inputs, z, name="encoder")

    return encoder

# build decoder
def create_vae_decoder(input_dim, latent_dim, hidden_dim, activation='relu'):
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for units in hidden_dim:
        x = Dense(units, activation=activation)(x)

    outputs = Dense(input_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name="decoder")

    return decoder

# build vae
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

###### FL ######
    
#### local training encoder ####
def local_enc_train(create_enc, x_local, y_local, batch_size, epochs):
    enc_model = create_enc
    enc_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SupervisedContrastiveLoss(temperature=0.1)
    )

    # train encoder
    enc_model.fit(
        x=x_local,
        y=y_local,
        batch_size=batch_size,
        epochs=epochs
    )

    # get local weights
    local_weights = enc_model.get_weights()

    return local_weights

#### local training classifier ####
def local_clf_train(create_clf, x_local, y_local, batch_size, epochs):
    clf_model = create_clf
    clf_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        # metrics=['accuracy']
        metrics=['accuracy', f1_metric, precision_metric, recall_metric]
    )

    # train encoder
    clf_model.fit(
        x=x_local,
        y=y_local,
        batch_size=batch_size,
        epochs=epochs
    )

    # get local weights
    local_weights = clf_model.get_weights()

    return local_weights



#### aggregate local weights ####
def aggregate_weights(weight_list):
    # Simple average aggregation for demonstration purposes
    avg_weights = np.mean(weight_list, axis=0)
    return avg_weights

#### federated training encoder ####
def federated_enc_training(num_round, create_enc, client_data_list,
                           client_target_list, weight_dir, batch_size, epochs):
    weight_dir = os.path.join(weight_dir, 'enc_weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    global_enc = create_enc
    global_weights = global_enc.get_weights()

    for round in range(num_round):
        print(f'Starting training round {round}')
        
        client_weights_list = []

        # local traning encoder
        for client_id, (x_client, y_client) in enumerate(zip(client_data_list, client_target_list)):
            
            print(f'Client {client_id} training')
            
           
            client_enc = create_enc

            local_weights = local_enc_train(create_enc, x_client, y_client, batch_size, epochs)
            

            # save local weights
            client_weights_list.append(local_weights)

            client_enc_path = os.path.join(weight_dir, f'client_{client_id}_enc_weights.h5')
            client_enc.save(client_enc_path)

        # aggregate local weights
            
        global_weights = aggregate_weights(client_weights_list)
        global_enc.set_weights(global_weights)

        # save global weights
        global_enc.save(os.path.join(weight_dir, f'global_enc_weights_{round}.h5'))

    return global_enc

# example use
# global_enc = federated_enc_training(num_rounds, create_encoder(input_shape, latent_dim, hidden_units), client_data_list, client_target_list, weight_dir, 32, 10)

#### federated training classifier ####
def federated_clf_training(num_round, create_clf, client_data_list,
                           client_target_list, weight_dir, batch_size, epochs):
    weight_dir = os.path.join(weight_dir, 'clf_weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    global_clf = create_clf
    global_weights = global_clf.get_weights()

    for round in range(num_round):
        print(f'Starting training round {round}')
        
        client_weights_list = []

        # local traning encoder
        for client_id, (x_client, y_client) in enumerate(zip(client_data_list, client_target_list)):
            
            print(f'Client {client_id} training')
            
           
            client_clf = create_clf

            local_weights = local_clf_train(create_clf, x_client, y_client, batch_size, epochs)
            

            # save local weights
            client_weights_list.append(local_weights)

            client_clf_path = os.path.join(weight_dir, f'client_{client_id}_clf_weights.h5')
            client_clf.save(client_clf_path)

        # aggregate local weights
            
        global_weights = aggregate_weights(client_weights_list)
        global_clf.set_weights(global_weights)

        # save global weights
        global_clf.save(os.path.join(weight_dir, f'global_clf_weights_{round}.h5'))

    return global_clf

# example use
# global_clf = federated_clf_training(num_rounds, create_decoder(global_enc), client_data_list, client_target_list, weight_dir, 32, 10)


#### CL training ####
def CL_training(model, client_data_list, client_target_list, weight_dir, batch_size, epochs):
    weight_dir = os.path.join(weight_dir, 'CL_weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    global_model = model
    global_weights = global_model.get_weights()

    for x_client, y_client in zip(client_data_list, client_target_list):
        print('start data concatenation')
        # concatenate client data
        x_train = np.concatenate(client_data_list)
        y_train = np.concatenate(client_target_list)

        print('start training')
        # train global model
        global_model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs
        )

        # save global weights
        global_model.save(os.path.join(weight_dir, f'global_model_weights.h5'))

    return global_model

### Local training ###
def ll_clf_training(num_round, create_clf, client_data_list,
                           client_target_list, weight_dir, batch_size, epochs):
    weight_dir = os.path.join(weight_dir, 'clf_weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)


    for round in range(num_round):
        print(f'Starting training round {round}')
        
        client_weights_list = []

        # local traning encoder
        for client_id, (x_client, y_client) in enumerate(zip(client_data_list, client_target_list)):
            
            print(f'Client {client_id} training')
            
           
            client_clf = create_clf

            local_weights = local_clf_train(create_clf, x_client, y_client, batch_size, epochs)
            

            # save local weights
            client_weights_list.append(local_weights)

            client_clf_path = os.path.join(weight_dir, f'client_{client_id}_clf_weights.h5')
            client_clf.save(client_clf_path)

        return client_clf

        