import argparse
import os.path as op
from siemens_pilots.utils.data import Subject
from nilearn import surface
import nibabel as nb
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform
import numpy as np

def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder):

        subjects_dir = op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', 'fsaverage')
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = 'fsaverage'
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r

def main(subject, multiband, bids_folder, smoothed):

    sub = Subject(subject, bids_folder=bids_folder)
    surfinfo = sub.get_surf_info()

    glm_key = 'glm_stim1.denoise'
    if smoothed:
        glm_key += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', glm_key, f'sub-{subject}', 'func')

    # sub-alina_task-task_space-T1w_acq-mb0_desc-R2_pe.nii.gz
    r2_volume = op.join(target_dir, f'sub-{subject}_task-task_space-T1w_acq-mb{multiband}_desc-R2_pe.nii.gz')

    print(f'Writing to {target_dir}')

    for hemi in ['L', 'R']:
        samples = surface.vol_to_surf(r2_volume, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
        samples = samples.astype(np.float32)
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples)])
                
        target_fn =  op.join(target_dir, f'sub-{subject}_task-task_space-T1w_acq-mb{multiband}_space-fsnative_hemi-{hemi}_desc-R2_pe.func.gii')

        nb.save(im, target_fn)

        transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('multiband', default=0, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-siemenspilotsfmap')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.multiband, bids_folder=args.bids_folder, smoothed=args.smoothed)
