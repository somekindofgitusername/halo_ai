from keras_tuner import RandomSearch, Hyperband, BayesianOptimization

def initialize_tuner(tuner_selection, build_model, directory='tuner_results', project_name='model_tuning'):
    tuner_selection = str(tuner_selection)
    print(f"Using {tuner_selection} tuner.")
    if tuner_selection == "1":
        print("RandomSearch")
        tuner = RandomSearch(
            build_model,
            objective='val_mean_absolute_error',
            max_trials=10,
            executions_per_trial=1,
            directory=directory,
            project_name=f'{project_name}_random_search',
            overwrite=True
        )
    elif tuner_selection == "2":
        print("Hyperband")
        tuner = Hyperband(
            build_model,
            #objective='val_mean_absolute_error',
            objective = 'val_mean_squared_error',
            #objective='val_loss',
            max_epochs=20,
            factor=3,
            directory=directory,
            project_name=f'{project_name}_hyperband',
            overwrite=True
        )
    else:  # BayesianOptimization is the default
        print("BayesianOptimization")
        tuner = BayesianOptimization(
            build_model,
            objective='val_mean_absolute_error',
            max_trials=40,
            executions_per_trial=2,
            directory=directory,
            project_name=f'{project_name}_bayesian',
            overwrite=True
        )
    return tuner
