from siemens_pilots.utils.data import Subject
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
import os.path as op
import numpy as np
import pandas as pd

'''
1. 4-parameter model (everything same)
2. 8-parameter model (everything different)
3. Model A, 4-parameters (mu_wide = 10 + 2* (mu_narrow  - 10))
4. Model B, 4-parameters (mu_wide = 10 + 2* (mu_narrow  - 10), sd_wide = sd_narrow * 2)
5. Mu sd free, everything else fixed
'''

range_increase_natural_space = (40 - 10) / (25 - 10) # (2)
range_increase_log_space = (np.log(40) - np.log(10)) / (np.log(25) - np.log(10)) # (~1.5)

def get_paradigm(sub, model_label, gaussian=True, multiband=None):
    behavior = sub.get_behavioral_data(session=None)

    behavior = behavior[behavior['multiband'] == multiband]

    range_increase = range_increase_log_space if not gaussian else range_increase_natural_space

    if model_label in [1, 2, 3, 4, 5]:
        paradigm = behavior[['n', 'range']].rename(columns={'n':'x'})
        paradigm['range'] = (paradigm['range'] == 'wide')

        if not gaussian:
            paradigm['x'] = np.log(paradigm['x'])
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if model_label in[3, 4]:
        paradigm['beta'] = paradigm['range'].map({False:1, True:range_increase})
        paradigm.drop('range', axis=1, inplace=True)


    paradigm = paradigm.astype(np.float32)

    return paradigm

def get_model(paradigm, model_label, gaussian=True):

    if model_label == 1:
        regressors = {}
    elif model_label == 2:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)', 'amplitude':'0 + C(range)', 'baseline':'0 + C(range)'}
    elif model_label == 3:
        regressors = {'mu':'0 + beta'}
    elif model_label == 4:
        regressors = {'mu':'0 + beta', 'sd':'0 + beta'}
    elif model_label == 5:
        regressors = {'mu':'0 + C(range)', 'sd':'0 + C(range)'}
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")

    if model_label in [1, 2, 5]:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)
    elif model_label in [3, 4]:
        if gaussian:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':10})
        else:
            model = RegressionGaussianPRF(paradigm=paradigm, regressors=regressors, baseline_parameter_values={'mu':np.log(10)})

    return model

def get_parameter_grids(model_label, gaussian=True):
    """Returns modes, sigmas, amplitudes, and baselines based on model_label and gaussian flag."""
    
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    def make_grid(min_val, max_val, steps, log_space):
        """Helper function to create either linear or log-spaced grids."""
        if log_space:
            return np.linspace(np.log(min_val), np.log(max_val), steps, dtype=np.float32)
        return np.linspace(min_val, max_val, steps, dtype=np.float32)

    # Standard mode/sigma definitions for other models
    model_params = {
        (1, ): (5, 45, 41, 3, 30, 30),
        (2, 5): (5, 45, 13, 3, 30, 13),
        (3, 4,):   (0, 15, 41, 3, 15, 30),  # Special case for log-space
    }

    for labels, (mode_min, mode_max, mode_steps, sigma_min, sigma_max, sigma_steps) in model_params.items():
        if model_label in labels:
            modes = make_grid(mode_min, mode_max, mode_steps, not gaussian)
            sigmas = make_grid(sigma_min, sigma_max, sigma_steps, not gaussian)
            return modes, sigmas, amplitudes, baselines

    raise ValueError(f"Unknown model_label: {model_label}")

def fit_model(model, paradigm, data, model_label, max_n_iterations=1000, gaussian=True):

    print(paradigm)
    optimizer = ParameterFitter(model, data.astype(np.float32), paradigm.astype(np.float32))

    # Get parameter grids
    if model_label == 11:
        modes1, modes2, sigmas, amplitudes, baselines = get_parameter_grids(model_label, gaussian)
    else:
        modes, sigmas, amplitudes, baselines = get_parameter_grids(model_label, gaussian)

    # Define grid fitting based on model_label
    model_grid_configs = {
        6: lambda: optimizer.fit_grid(modes, modes, sigmas, amplitudes, baselines),
        8: lambda: optimizer.fit_grid(modes, modes, sigmas, sigmas, amplitudes, baselines),
        9: lambda: optimizer.fit_grid(modes, sigmas, sigmas, amplitudes, baselines),
        11: lambda: optimizer.fit_grid(modes1, modes2, sigmas, sigmas, amplitudes, baselines),
        2: lambda: optimizer.fit_grid(modes, modes, sigmas, sigmas, amplitudes, amplitudes, baselines, baselines),
        5: lambda: optimizer.fit_grid(modes, sigmas, sigmas, amplitudes, amplitudes, baselines, baselines),
        7: lambda: optimizer.fit_grid(modes, sigmas, sigmas, amplitudes, amplitudes, baselines, baselines),
    }

    print("FITTING GRID")
    # Default grid fitting
    grid_pars = model_grid_configs.get(model_label, lambda: optimizer.fit_grid(modes, sigmas, amplitudes, baselines))()

    print(grid_pars.describe())

    # Define fixed parameters
    fixed_pars = list(model.parameter_labels)
    fixed_mapping = {
        (1, 3, 4, 6, 8, 9, 10, 11): [('amplitude_unbounded', 'Intercept'), ('baseline_unbounded', 'Intercept')],
        (2, 5, 7): [
            ('amplitude_unbounded', 'C(range)[0.0]'),
            ('baseline_unbounded', 'C(range)[0.0]'),
            ('amplitude_unbounded', 'C(range)[1.0]'),
            ('baseline_unbounded', 'C(range)[1.0]'),
        ]
    }

    for keys, to_remove in fixed_mapping.items():
        if model_label in keys:
            for item in to_remove:
                fixed_pars.pop(fixed_pars.index(item))

    # Fit one (only baseline/amplitude)
    gd_pars = optimizer.fit(
        init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False,
        max_n_iterations=max_n_iterations, fixed_pars=fixed_pars, r2_atol=0.001
    )

    # Fit two
    gd_pars = optimizer.fit(
        init_pars=optimizer.estimated_parameters, learning_rate=.01, store_intermediate_parameters=False,
        max_n_iterations=max_n_iterations, r2_atol=0.00001
    )

    print(gd_pars.describe())

    return gd_pars

def get_conditionspecific_parameters(model_label, model, estimated_parameters, gaussian=True):

    range_increase = range_increase_log_space if not gaussian else range_increase_natural_space

    print("Getting parameters with range_increase", range_increase)
    
    if model_label in [1, 2, 5]:
        conditions = pd.DataFrame({'x':[0,0], 'range':[0,1]}, index=pd.Index(['narrow', 'wide'], name='range'))
    elif model_label in [3, 4]:
        conditions = pd.DataFrame({'beta':[1,range_increase]}, index=pd.Index(['narrow', 'wide'], name='range'))
    else:
        raise NotImplementedError(f"Model {model_label} is not implemented")
        
    pars = model.get_conditionspecific_parameters(conditions, estimated_parameters)

    return pars
