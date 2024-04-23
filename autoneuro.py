# Imports grouped by functionality
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split

#from sklearn.feature_selection import SelectFromModel
#from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor if it's a regression problem
#from sklearn.preprocessing import MinMaxScaler

from helper_functions import preprocess_and_feature_engineer
#from feature_selector import select_features_by_variance
from constants import FEATURE_COLUMNS, TARGET_COLUMNS
from data_handling import save_model
from reporting import print_evaluation_metrics
from model import build_model
from tuner_options import initialize_tuner
from best_model import build_best_model


# Constants for user interface
TASK_PROMPT = "Train a model (1) or Refine an existing model(2)"
TUNER_PROMPT = "Select a tuner:\n1. RandomSearch\n2. Hyperband\n3. BayesianOptimization\nEnter the number of your choice (default '1'): "
DATA_PERCENTAGE_PROMPT = "Enter the percentage of data to use (1-100): "
INVALID_INPUT_MESSAGE = "Invalid input. Defaulting to treating color components as separate features."
INVALID_PERCENTAGE_MESSAGE = "Percentage must be between 1 and 100. Using 100% of the data."
REFINE_MODEL_PROMPT = "Would you like to refine the model? (y/n): "

FILE_PATH = "attributes_test.csv"
#FILE_PATH = "attributes.csv"
OUTPUT_FILE_PATH = "attributes_preprocessed.csv"
def get_user_choice(prompt, default, type_func=int, validation_func=None):
    """Prompt user for input and validate it."""
    while True:
        try:
            user_input = input(prompt).strip() or default
            value = type_func(user_input)
            if validation_func and not validation_func(value):
                raise ValueError
            return value
        except ValueError:
            print(INVALID_INPUT_MESSAGE if type_func is int else INVALID_PERCENTAGE_MESSAGE)



def main():
    # Ask user if they want to train a new model or refine an existing one
    task_choice = get_user_choice(TASK_PROMPT, '1')
    # Get user input for data treatment and percentage
    data_percentage = get_user_choice(DATA_PERCENTAGE_PROMPT, '100', float, lambda x: 1 <= x <= 100)  
    preprocess_and_feature_engineer(FILE_PATH, OUTPUT_FILE_PATH, data_percentage)
    
    # Load and prepare the dataset
    data = pd.read_csv(OUTPUT_FILE_PATH)
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMNS]

    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
       
 
    if task_choice == 1:     
        print("Features", features)
        print("Target", target)
        # Initialize and perform tuner search
        tuner_selection = get_user_choice(TUNER_PROMPT, '1')
        tuner = initialize_tuner(tuner_selection,
                                build_model, 
                                directory='tuner_results', 
                                project_name='brightness_prediction'
                                )
        tuner.search(X_train, y_train, 
                    epochs=20, 
                    validation_split=0.2, 
                    verbose=1)
        tuner.results_summary()   
        # # Evaluate and save the best model
        best_model = tuner.get_best_models(num_models=1)[0]

        # Ask user if they want to refine the best model
        refine_choice = get_user_choice(REFINE_MODEL_PROMPT, 'n', str, lambda x: x.lower() in ['y', 'n'])
        if refine_choice.lower() == 'y':
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(filepath='model_best.h5', monitor='val_loss', save_best_only=True)
            best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[early_stopping, model_checkpoint])
    # Refine an existing model
    else:
        best_model = build_best_model()
        best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200)

    
    predictions = best_model.predict(X_test)
    print_evaluation_metrics(y_test, predictions)
    file_name = "best_model_" + "_".join(TARGET_COLUMNS) + ".onnx"
    #save_model(best_model, 'best_model.h5', 'best_model_savedmodel', 'best_model_red.onnx')
    save_model(best_model, 'best_model.h5', 'best_model_savedmodel', file_name)




if __name__ == "__main__":
    main()
