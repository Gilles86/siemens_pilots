import cortex
import numpy as np
import matplotlib.pyplot as plt
from siemens_pilots.utils.data import Subject
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import os.path as op
import pandas as pd
from nilearn import surface
import numpy as np

subject = 'alina'

sub = Subject(subject_id=subject, bids_folder='/data/ds-siemenspilotsfmap')

ds = {}

keys = [] 
for smoothed, multiband in product([False, True], [0, 2, 4]):
    prf_pars = sub.get_prf_parameters_surf(model_label=4, multiband=multiband, smoothed=smoothed, gaussian=True)

    
    key = f'mb{multiband}.smoothed' if smoothed else f'mb{multiband}.unsmoothed'

    r2 = prf_pars['r2']
    ds[f'{key}_r2'] = get_alpha_vertex(r2.values, (r2 > 0.06).values, vmin=0.0, vmax=0.3,
                                       subject=f'siemenspilots.sub-{subject}', cmap='plasma')


    
for smoothed, multiband in product([False, True], [0, 2, 4]):
    prf_pars = sub.get_prf_parameters_surf(model_label=4, multiband=multiband, smoothed=smoothed, gaussian=True)

    r2 = prf_pars['r2']
    mu = prf_pars['mu.narrow']

    key = f'mb{multiband}.smoothed' if smoothed else f'mb{multiband}.unsmoothed'
    ds[f'{key}_mu'] = get_alpha_vertex(mu.values, (r2 > 0.05).values, vmin=5, vmax=20,
                                       subject=f'siemenspilots.sub-{subject}', cmap='nipy_spectral')

cortex.webgl.show(ds)

vmin, vmax = 5, 25
x = np.linspace(0, 1, 101, True)

# Width is 80 x 1, so 
im = plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
        extent=[vmin, vmax, 0, 1], aspect=1.*(vmax-vmin) / 20.,
        origin='lower')
print(im.get_extent())
plt.yticks([])
plt.tight_layout()

ns = np.array([5, 10, 15, 20, 25])
ns = ns[ns <= vmax]
plt.xticks(ns)
plt.show()

