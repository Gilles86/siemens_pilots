import cortex
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import os.path as op
import pandas as pd
from nilearn import surface
import numpy as np

subject = 'alina'

# subjects = [Subject(subject_id=subject_id) for subject_id in get_all_subject_ids()]

# ds = {}


# df = []

# keys = []
# for sub, model_label in product(subjects, range(1, 9)):
#     try:
#         pars = sub.get_prf_parameters_surf(model_label=model_label, smoothed=True, gaussian=True, space='fsaverage')
#         df.append(pars)
#         keys.append((sub.subject_id, model_label))
#     except Exception as e:
#         print(f'Failed for {sub.subject_id} model {model_label}: {e}')

# df = pd.concat(df, keys=keys, names=['subject_id', 'model_label'])


df = []

index = []

for ds, mb, smoothed in product(['ds-siemenspilots24', 'ds-siemenspilotsfmap'], [0, 2, 4], [False, True]):

    key = 'glm_stim1.denoise'

    if smoothed:
        key += '.smoothed'

    d = []
    for hemi in ['L', 'R']:
        fn = op.join('/data', ds, 'derivatives', key, f'sub-{subject}', 'func', f'sub-alina_task-task_space-T1w_acq-mb{mb}_space-fsnative_hemi-{hemi}_desc-R2_pe.func.gii')
        d.append(surface.load_surf_data(fn)[:, np.newaxis])
    d = pd.concat((pd.DataFrame(d[0], ),
                   pd.DataFrame(d[1], )),
                   keys=['L', 'R'], names=['hemi'])

    index.append((ds, mb, smoothed))
    df.append(d)

df = pd.concat(df, keys=index, names=['ds', 'mb', 'smoothed'])

df.columns = ['R2']

vertices = {}

for (smoothed, ds, mb), d in df.groupby(['smoothed', 'ds', 'mb']):

    smoothed_str = 'smoothed' if smoothed else 'unsmoothed'

    name = f'{ds}_mb{mb}_{smoothed_str}'

    vertices[name] = d


    vertices[name] = get_alpha_vertex(d['R2'].values, (d['R2'] > 10).values, vmin=0.0, vmax=50, subject=f'siemenspilots.sub-{subject}', cmap='plasma')

cortex.webgl.show(vertices)