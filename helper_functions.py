import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
# import constants
from constants import FEATURE_COLUMNS, TARGET_COLUMNS


def tf_r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def tanh_normalize(data):
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data_normalized = np.tanh(data_standardized)
    return data_normalized


def preprocess_and_feature_engineer(
    file_path, output_file_path, 
    include_percentage=100):
    
    data = pd.read_csv(file_path, on_bad_lines='skip')
    
    predata = data[FEATURE_COLUMNS + TARGET_COLUMNS]
        
    reduced_predata = predata.sample(frac=include_percentage/100.0)
    reduced_predata.to_csv(output_file_path, index=False)
    print(f"Preprocessed data ({include_percentage}% of original) has been saved to {output_file_path}")




