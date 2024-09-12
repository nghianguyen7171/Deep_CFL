import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from plot_metric.functions import BinaryClassification
from os import listdir
from sklearn.preprocessing import StandardScaler
import tqdm

import warnings
warnings.filterwarnings("ignore")



def ts_dataloader(df, time_list, dem_list, window_len, scaler, stride, task):
    # rename alternative number to alternative_number
    df = df.rename(columns={'alternative number': 'alternative_number'})
    

    features_list = [x for x in df.columns if x not in ['ev_w_dec', 'is_detection', 'alternative_number', 'is_event', 'label']]
    
    # Normalization for sign_list only
    if scaler is not None:
        scaler = StandardScaler()
        scaler.fit(df[features_list])
        features = scaler.transform(df[features_list])

        for idx, features_name in enumerate(features_list):
            df[features_name] = features[:, idx]

    
    # id label
    patient_cnts = np.unique(df["alternative_number"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids  = list(patient_cnts.keys())

    

    # window setting
    data_info = {}
    idx_debug = 0
    for patient_id in tqdm.tqdm(patient_ids):
        
        df_patient = df.query(f'alternative_number=="{patient_id}"')
        #print(df_patient["is_detection"])
        for idx in range(len(df_patient) - window_len + 1 - stride):
            row_info = {}
            from_idx = idx
            to_idx   = idx + window_len - 1
            to_target = to_idx + stride

            ############# row ################
            row_info["pid"] = df_patient["alternative_number"].iloc[from_idx: to_idx + 1].values
            row_info["x_t"] = df_patient[time_list].iloc[from_idx: to_idx + 1].values
            row_info["x_d"] = df_patient[dem_list].iloc[from_idx: to_idx + 1].values
            
            # label for abnormal or normal
            row_info["y"] = df_patient["label"].iloc[from_idx: to_target].values
            row_info["seq_y"] = df_patient["label"].iloc[to_target]

            # label for detection or not
            row_info["y_dec"] = df_patient["is_detection"].iloc[from_idx: to_target].values
            row_info["seq_y_dec"] = df_patient["is_detection"].iloc[to_target]

            # label for event or not
            row_info["y_event"] = df_patient["is_event"].iloc[from_idx: to_target].values
            row_info["seq_y_event"] = df_patient["is_event"].iloc[to_target]

            # label for event with detection or not
            row_info["y_event_w_dec"] = df_patient["ev_w_dec"].iloc[from_idx: to_target].values
            row_info["seq_y_event_w_dec"] = df_patient["ev_w_dec"].iloc[to_target]

            ############ append ##############
            for key in row_info:
                if data_info.get(key) is None: data_info[key] = []
                data_info[key].append(row_info[key])
                pass # key

            pass # row

        pass # data
        # break

    for key in row_info:
        data_info[key] = np.array(data_info[key])
        pass
    
    #print(patient_id)
    
    x_t = data_info["x_t"]
    x_d = data_info["x_d"]

    y = data_info["seq_y"]
    y_onehot = np.zeros((len(data_info["seq_y"]),2), dtype=np.float32)
    for idx in range(len(y)):
       y_onehot[idx, y.astype(np.int64)[idx]] = 1.0

    y_dec = data_info["seq_y_dec"]
    y_dec_onehot = np.zeros((len(data_info["seq_y_dec"]),2), dtype=np.float32)
    for idx in range(len(y_dec)):
       y_dec_onehot[idx, y_dec.astype(np.int64)[idx]] = 1.0

    y_event = data_info["seq_y_event"]
    y_event_onehot = np.zeros((len(data_info["seq_y_event"]),2), dtype=np.float32)
    for idx in range(len(y_event)):
       y_event_onehot[idx, y_event.astype(np.int64)[idx]] = 1.0

    y_event_w_dec = data_info["seq_y_event_w_dec"]
    y_event_w_dec_onehot = np.zeros((len(data_info["seq_y_event_w_dec"]),2), dtype=np.float32)
    for idx in range(len(y_event_w_dec)):
        y_event_w_dec_onehot[idx, y_event_w_dec.astype(np.int64)[idx]] = 1.0

    if task == "abnormal":
        return x_t, x_d, y, y_onehot
    
    elif task == "detection":
        return x_t, x_d, y_dec, y_dec_onehot
    
    elif task == "event":
        return x_t, x_d, y_event, y_event_onehot
    
    elif task == "event_w_dec":
        return x_t, x_d, y_event_w_dec, y_event_w_dec_onehot
