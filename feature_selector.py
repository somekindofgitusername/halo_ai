# feature_selector.py
from sklearn.feature_selection import VarianceThreshold

def select_features_by_variance(dataframe, feature_columns, threshold=0.0):
    """
    Selects features based on variance threshold.
    
    Parameters:
        dataframe (pd.DataFrame): The dataset containing the features.
        feature_columns (list): List of feature column names to consider.
        threshold (float): The variance threshold for feature selection.
        
    Returns:
        list: The list of selected feature column names.
    """
    selector = VarianceThreshold(threshold)
    selector.fit(dataframe[feature_columns])
    # Get the selected features based on the variance threshold
    features = [feature_columns[i] for i in selector.get_support(indices=True)]
    return features
