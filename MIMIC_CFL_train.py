import sys
import os
import warnings
import torch
import numpy as np
import random
import pandas as pd
from FL_utils import *  # assuming these are custom modules
from FL_model import (SupervisedContrastiveLoss, create_encoder, add_projection_head, 
                      create_decoder, create_vae_encoder, Sampling, local_enc_train, 
                      local_clf_train, aggregate_weights, federated_enc_training, 
                      federated_clf_training)
from FL_evaluation import *

# Ignore warnings
warnings.filterwarnings("ignore")

# Seed fixing function
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Main function
def main():
    # Set seed for reproducibility
    seed = 42
    fix_seed(seed)
    
    # Load the data
    mimic_dir = r'/media/nghia/DATA/DATA/MIMICIII_Grouped'
    exclude_var_list = ['icustay_id', 'label_death_icu']
    
    train_data_mimic, train_target_mimic = get_data(mimic_dir, 'train_24h(min24h).csv',
                                                    exclude_var_list=exclude_var_list, outcome_var='label_death_icu')
    test_data_mimic, test_target_mimic = get_data(mimic_dir, 'test_24h(min24h).csv',
                                                    exclude_var_list=exclude_var_list, outcome_var='label_death_icu')
    
    # Split the data for federated learning
    train_data_mimic_list = []
    train_target_mimic_list = []
    num_clients = 9
    
    for i in range(num_clients):
        train_data_mimic_list.append(train_data_mimic.iloc[int(i*len(train_data_mimic)/num_clients):int((i+1)*len(train_data_mimic)/num_clients)])
        train_target_mimic_list.append(train_target_mimic.iloc[int(i*len(train_target_mimic)/num_clients):int((i+1)*len(train_target_mimic)/num_clients)])
    
    # Create client-specific data
    client_data_list = []
    client_target_list = []
    
    for i in range(num_clients):
        client_data_list.append(pd.DataFrame(train_data_mimic_list[i]))
        client_target_list.append(pd.DataFrame(train_target_mimic_list[i]))
    
    # Model parameters
    input_shape = (23,)
    latent_dim = 16
    hidden_units = [128, 64, 32]
    num_rounds = 10
    weight_dir = "weights_vae_9Clients_multi"
    batch_size = 32
    epochs = 10

    # Train the global encoder using federated learning
    global_enc = federated_enc_training(num_rounds, create_encoder(input_shape, latent_dim, hidden_units), 
                                        client_data_list, client_target_list, weight_dir, batch_size, epochs)
    
    # Train the global classifier
    global_clf = federated_clf_training(num_rounds, create_decoder(global_enc), 
                                        client_data_list, client_target_list, weight_dir, batch_size, epochs)
    
    # Make predictions on the test data
    y_pred_global, y_label_global = get_predictions(global_clf, test_data_mimic)
    
    # Evaluate the model
    bnr_rp(test_target_mimic.values, y_pred_global, y_label_global, labels=[0, 1])

# Entry point
if __name__ == "__main__":
    main()
