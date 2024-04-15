from keras_tuner import RandomSearch, Hyperband, BayesianOptimization

def initialize_tuner(tuner_selection, build_model, directory='tuner_results', project_name='model_tuning'):
    if tuner_selection == "1":
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
        tuner = Hyperband(
            build_model,
            objective='val_mean_absolute_error',
            max_epochs=30,
            factor=3,
            directory=directory,
            project_name=f'{project_name}_hyperband',
            overwrite=True
        )
    else:  # BayesianOptimization is the default
        tuner = BayesianOptimization(
            build_model,
            objective='val_mean_absolute_error',
            max_trials=10,
            executions_per_trial=1,
            directory=directory,
            project_name=f'{project_name}_bayesian',
            overwrite=True
        )
    return tuner
