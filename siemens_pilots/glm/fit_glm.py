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
    for session in [1, 2, 3]:
        for repetition in [1,2]:
            run = get_run_from_mb(mb, session, repetition)
            ims.append(sub.get_bold(session, mb=mb, repetition=repetition))
            onsets.append(sub.get_onsets(session, run))
            keys.append((session, run))

    onsets = pd.concat(onsets, keys=keys, names=['session', 'run'])
    print(onsets)

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

    for ((session, run), onset), im in zip (onsets.groupby(['session', 'run']), ims):
        tr = image.load_img(im).header['pixdim'][4]
        n = image.load_img(im).shape[-1]
        frametimes = np.linspace(tr/2., (n - .5)*tr, n)
        print(im, tr, n)
        print(onset)
        onset['onset'] = ((onset['onset']+tr/2.) // tr) * tr

        dm = make_first_level_design_matrix(frametimes,
                                            onset,
                                            hrf_model='fir',
                                            drift_model=None,
                                            drift_order=0).drop('constant', axis=1).fillna(0.0)


        dm.columns = [c.replace('_delay_0', '') for c in dm.columns]
        dm /= dm.max()
        dm = np.round(dm)
        print(dm)
        dms.append(dm)

    dms = pd.concat(dms, keys=keys, names=['session', 'run'], axis=0).fillna(0.0)
    print(dms)

    X = [d.values for _, d in dms.groupby(['session', 'run'])]
    print(X)

    for x in X:
        print(x.shape)

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

    # see https://github.com/cvnlab/GLMsingle/pull/130
    # confounds = sub.get_confounds(session=session)
    # confounds = [d.values for run, d in sub.get_confounds().groupby('run')]
    # opt['extra_regressors'] = confounds

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    data = [image.load_img(im).get_fdata() for im in ims]

    results_glmsingle = glmsingle_obj.fit(
        X,
        data,
        0.6,
        tr,
        outputdir=base_dir)

    betas = results_glmsingle['typed']['betasmd']
    betas = image.new_img_like(ims[0], betas)
    stim_betas = image.index_img(betas, slice(None, None, 2))
    resp_betas = image.index_img(betas, slice(1, None, 2))
    
    fn_template = op.join(base_dir, 'sub-{subject}_task-task_space-T1w_acq-mb{mb}_desc-{par}_pe.nii.gz')

    stim_betas.to_filename(fn_template.format(subject=subject, session=session, par='stim'))
    resp_betas.to_filename(fn_template.format(subject=subject, session=session, par='response'))

    r2 = results_glmsingle['typed']['R2']
    r2 = image.new_img_like(ims[0], r2)
    r2.to_filename(fn_template.format(subject=subject, session=session, par='R2'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('multiband', default=None, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-siemenspilots')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.multiband, True,
         bids_folder=args.bids_folder, smoothed=args.smoothed)
