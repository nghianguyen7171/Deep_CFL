import numpy as np
import pandas as pd
import os


def get_data(data_dir, data_name, exclude_var_list , outcome_var):
    data = pd.read_csv(os.path.join(data_dir, data_name))

    features_list = [x for x in data.columns if x not in exclude_var_list]
    input_var = features_list

    return data[input_var], data[outcome_var]

def client_split(data, target_data ,num_clients):
    train_data_list = []
    train_target_list = []

    for i in range(num_clients):
        train_data_list.append(data.iloc[int(i*len(data)/num_clients):int((i+1)*len(data)/num_clients)])
        train_target_list.append(target_data.iloc[int(i*len(data)/num_clients):int((i+1)*len(data)/num_clients)])

    for i in range(num_clients):
        globals()[f'client_data_{i}'] = pd.DataFrame(train_data_list[i])
        globals()[f'client_target_{i}'] = pd.DataFrame(train_target_list[i])

    

    

