from glmsingle.glmsingle import GLM_single
import argparse
import os
import os.path as op
from nilearn import image
from siemens_pilots.utils.data import Subject, get_run_from_mb
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main(subject, mb, fit_both_pes, encoding_direction=None,
         bids_folder='/data/ds-siemenspilots', smoothed=True):

    derivatives = op.join(bids_folder, 'derivatives')
    sub = Subject(subject, bids_folder=bids_folder)

    bold = []

    if not fit_both_pes:
        raise NotImplementedError('Only fit_both_pes=True is supported')

    ims = []
    onsets = []
    keys = []

    if mb == - 1:

        if subject == 'alina':
            sessions = ['philips']
        else:
            sessions = ['philips1', 'philips2']
        
        for session in sessions:
            for run in range(1, 9):
                ims.append(sub.get_bold(session, mb=mb, run=run))
                onsets.append(sub.get_onsets(session, run))
                keys.append((session, run))
                print('Minimum onset: ', onsets[-1].onset.min())
                print('Maximum onset: ', onsets[-1].onset.max())

    else:
        for session in [1, 2, 3]:
            for repetition in [1,2]:
                run = get_run_from_mb(mb, session, repetition)
                ims.append(sub.get_bold(session, mb=mb, repetition=repetition))
                onsets.append(sub.get_onsets(session, run))
                keys.append((session, run))

    onsets = pd.concat(onsets, keys=keys, names=['session', 'run'])

    base_dir = 'glm_stim1.denoise'

    if smoothed:
        base_dir += '.smoothed'
        ims = [image.smooth_img(im, fwhm=5.0) for im in ims]


    # onsets = sub.get_onsets(session)
    onsets['trial_type'] = onsets.apply(lambda row: f'stimulus_{row["n"]}' if row['trial_type'] == 'stimulus' else f'response_{row.response}', axis=1)
    onsets['duration'] = 0.0

    base_dir = op.join(derivatives, base_dir, f'sub-{subject}', 'func')

    if not op.exists(base_dir):
        os.makedirs(base_dir)

    dms = []

    # Loop over unique (session, run) pairs in the 'onsets' DataFrame and their corresponding images in 'ims'
    for ((session, run), onset), im in zip(onsets.groupby(['session', 'run']), ims):
        
        # Load the image to extract the repetition time (TR) from the header
        tr = image.load_img(im).header['pixdim'][4]  # TR is stored in the 5th index of 'pixdim'

        print(f'WARNING, TR is {tr}')

        # Extract the number of volumes (time points) in the fMRI image
        n = image.load_img(im).shape[-1]  # Last dimension is the time dimension

        # Generate an array of 'frametimes', corresponding to the middle of each TR window
        frametimes = np.linspace(tr/2., (n - 0.5) * tr, n)

        # Round onset times to the nearest valid frame time (ensuring alignment with frametimes)
        onset['onset'] = np.clip(
            np.round(onset['onset'] / (tr / 2)) * (tr / 2),  # Round to nearest half-TR
            tr / 2,  # Ensure onsets are not lower than the first valid frame time
            (n - 0.5) * tr  # Ensure onsets do not exceed the last frame time
        )

        # Generate the first-level design matrix using the onset times
        dm = make_first_level_design_matrix(
            frametimes,  # Time points for model sampling
            onset,  # Onset times for events
            hrf_model='fir',  # Use finite impulse response (FIR) HRF model
            drift_model=None,  # No drift model applied
            drift_order=0  # No polynomial drift terms
        )

        # Drop the 'constant' column (typically used for baseline adjustment)
        dm = dm.drop('constant', axis=1).fillna(0.0)  # Replace NaN values with 0

        # Clean column names by removing '_delay_0' suffix (if present)
        dm.columns = [c.replace('_delay_0', '') for c in dm.columns]

        # Append the design matrix to the list of design matrices
        dms.append(dm)


    dms = pd.concat(dms, keys=keys, names=['session', 'run'], axis=0).fillna(0.0)
    dms /= dms.max()
    dms = np.round(dms)

    X = [d.values for _, d in dms.groupby(['session', 'run'])]

    # # create a directory for saving GLMsingle outputs

    opt = dict()

    opt['sessionindicator'] = np.array([session for (session, run) in keys])[np.newaxis, :]
    print(opt['sessionindicator'])

    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [0, 0, 0, 1]
    opt['n_pcs'] = 20

    # see https://github.com/cvnlab/GLMsingle/pull/130
    # confounds = sub.get_confounds(session=session)
    # confounds = [d.values for run, d in sub.get_confounds().groupby('run')]
    # opt['extra_regressors'] = confounds

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    data = [image.load_img(im).get_fdata() for im in ims]

    outputdir = op.join(base_dir, 'glmdenoise', f'mb-{mb}')
    os.makedirs(outputdir, exist_ok=True)

    figuredir = op.join(base_dir, 'figures', f'mb-{mb}')
    os.makedirs(figuredir, exist_ok=True)

    results_glmsingle = glmsingle_obj.fit(
        X,
        data,
        0.6,
        tr,
        outputdir=outputdir,
        figuredir=figuredir)

    betas = results_glmsingle['typed']['betasmd']
    betas = image.new_img_like(ims[0], betas)
    stim_betas = image.index_img(betas, slice(None, None, 2))
    resp_betas = image.index_img(betas, slice(1, None, 2))
    
    fn_template = op.join(base_dir, 'sub-{subject}_task-task_space-T1w_acq-mb{mb}_desc-{par}_pe.nii.gz')

    stim_betas.to_filename(fn_template.format(subject=subject, par='stim', mb=mb))
    resp_betas.to_filename(fn_template.format(subject=subject, par='response', mb=mb))

    r2 = results_glmsingle['typed']['R2']
    r2 = image.new_img_like(ims[0], r2)
    r2.to_filename(fn_template.format(subject=subject, par='R2', mb=mb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('multiband', default=None, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-siemenspilots')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.multiband, True,
         bids_folder=args.bids_folder, smoothed=args.smoothed)
