import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#import torch
#import torch.nn as nn

# import tensorflow as tf
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Flatten, Dense, Embedding
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras import metrics
# from tensorflow.keras.callbacks import Callback, EarlyStopping



def mews_sbp(sbp: float) -> int:
    score = 0
    # SBP score
    if sbp <= 70:
        score += 3
    elif sbp <= 80:
        score += 2
    elif sbp <= 100:
        score += 1
    elif sbp >= 200:
        score += 2

    return score


def mews_hr(hr: float) -> int:
    score = 0

    # hr
    if hr <= 40:
        score += 2
    elif hr <= 50:
        score += 1
    elif hr <= 100:
        score += 0
    elif hr <= 110:
        score += 1
    elif hr <= 130:
        score += 2
    else:
        score += 3

    return score


def mews_rr(rr: float) -> int:
    score = 0

    if rr <= 8:
        score += 2
    elif rr <= 14:
        score += 0
    elif rr <= 20:
        score += 1
    elif rr <= 29:
        score += 2
    else:
        score += 3

    return score


def mews_bt(bt: float) -> int:
    score = 0

    if bt <= 35:
        score += 1
    elif bt <= 38.4:
        score += 0
    else:
        score += 2

    return score


def mews(hr: float, rr: float, sbp: float, bt: float) -> int:
    s_hr = mews_hr(hr)
    s_rr = mews_rr(rr)
    s_sbp = mews_sbp(sbp)
    s_bt = mews_bt(bt)

    score = s_hr + s_rr + s_sbp + s_bt

    return score




