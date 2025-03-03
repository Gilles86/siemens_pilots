import os.path as op
import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import image
from nilearn.input_data import NiftiMasker

mb_orders = [[0, 2, 4], [4, 2, 0], [2, 0, 4]]

def get_lr_direction(session, run):
    mb = mb_orders[session-1][(run-1) % 3]
    repetition = (run-1) // 3 + 1

    assert mb in [0, 2, 4]

    if repetition == 1:
        if mb in [0, 4]:
            return 'LR'
        else:
            return 'RL'
    else:
        if mb in [0, 4]:
            return 'RL'
        else:
            return 'LR'

def get_run_from_mb(mb, session, repetition):
    return mb_orders[session-1].index(mb) + (repetition-1) * 3 + 1
class Subject(object):


    def __init__(self, subject_id, bids_folder='/data/ds-siemenspilots'):
        if type(subject_id) == int:
            subject_id = f'{subject_id:02d}'
        
        self.subject_id = subject_id
        self.bids_folder = bids_folder

        self.derivatives_dir = op.join(bids_folder, 'derivatives')


    def get_behavioral_data(self, session=None, tasks=None, raw=False, multiband=None, add_info=True):

        if session is None:
            data = pd.concat((self.get_behavioral_data(session, tasks, raw, add_info) for session in self.get_sessions()), keys=self.get_sessions(), names=['session'])

            if tasks is None:
                data = data.reorder_levels(['subject', 'session', 'run', 'trial_nr']).sort_index()
            else:
                data = data.reorder_levels(['subject', 'session', 'task', 'run', 'trial_nr']).sort_index()

            return data

        if tasks is None:
            tasks = ['estimation_task']

        behavior_folder = op.join(self.bids_folder, 'sourcedata', 'behavior', f'sub-{self.subject_id}', f'ses-{session}', )


        df = []
        keys = []

        for task in tasks:
            if task == 'feedback':
                runs = [1, 5]
            elif task == 'estimation_task':
                runs = list(range(1, 7))

            for run in runs:
                try:
                    fn = op.join(behavior_folder, f'sub-{self.subject_id}_ses-{session}_task-{task}_run-{run}_events.tsv')
                    d = pd.read_csv(fn, sep='\t')

                    if (d['n'] > 25).any():
                        d['range'] = 'wide'
                    else:
                        d['range'] = 'narrow'

                    d['multiband'] = mb_orders[session-1][(run - 1) % 3]
                    keys.append((self.subject_id, task, run))
                    df.append(d)
                except Exception as e:
                    print(f'Problem with {task} run {run}: {e}')

        df = pd.concat(df, keys=keys, names=['subject', 'task', 'run']).set_index('event_type', append=True)
        df = df.droplevel(-2)

        if raw:
            if add_info:
                raise ValueError('add_info is not implemented for raw data')
            return df

        df = df.set_index('trial_nr', append=True)

        df = df.xs('feedback', level='event_type')

        if len(df.index.unique(level='task')) == 1:
            df = df.droplevel('task')

        df['response'] = df['response'].astype(float)
        df['n'] = df['n'].astype(float)

        if add_info:
            df['error'] = df['response'] - df['n']
            df['abs_error'] = np.abs(df['error'])
            df['squared_error'] = df['error']**2

        return df

    def get_sessions(self):
        return [1,2,3]

    def get_bold(self, session, run=None, repetition=None, mb=None):

        assert(run is not None or ((repetition is not None) and (mb is not None))), 'Either run or repetition and mb must be specified'

        if run is None:
            assert mb in [0, 2, 4], 'mb must be 0, 2 or 4'
            assert repetition in [1,2], 'repetition must be 1 or 2'
            run = get_run_from_mb(mb, session, repetition)

        if mb is None:
            mb = mb_orders[session-1][(run - 1) % 3]
            repetition = run // 3 + 1

        lr_direction = get_lr_direction(session, run)

        preproc_folder = Path(self.derivatives_dir, 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func')

        # sub-alina_ses-1_task-numestimate_acq-mb0_dir-LR_run-01_space-T1w_desc-preproc_bold.nii.gz
        fn = preproc_folder / f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb{mb}_dir-{lr_direction}_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz'

        assert(fn.exists()), f'File {fn} does not exist'

        return fn

    def get_onsets(self, session, run):

        onsets = pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_events.tsv'), index_col='trial_nr', sep='\t')

        return onsets

    def get_runs(self, session):
        return list(range(1, 7))

    def get_single_trial_estimates(self, multiband, type='stim', smoothed=False, roi=None):

        if roi is not None:
            raise NotImplementedError('ROI not implemented')

        dir = 'glm_stim1.denoise'

        if smoothed:
            dir += '.smoothed'

        dir = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func')
        # sub-alina_task-task_space-T1w_acq-mb0_desc-R2_pe.nii.gz
        fn = op.join(dir, f'sub-{self.subject_id}_task-task_space-T1w_acq-mb{multiband}_desc-{type}_pe.nii.gz')

        im = image.load_img(fn, dtype=np.float32)

        n_volumes = 240
        assert(im.shape[3] == n_volumes), f'Expected {n_volumes} volumes, got {im.shape[3]}'

        return im

    def get_brain_mask(self, session=None, epi_space=True, return_masker=True, debug_mask=False):

        if not epi_space:
            raise ValueError('Only EPI space is supported')

        session = 1 if session is None else session

        # sub-alina_ses-1_task-numestimate_acq-mb0_dir-LR_run-01_desc-brain_mask.nii.gz
        fn = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}',
                    f'ses-{session}', 'func', 
                    f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb0_dir-LR_run-01_space-T1w_desc-brain_mask.nii.gz')

        mask_img = image.load_img(fn)

        if debug_mask:
            # Convert to numpy array
            mask_data = mask_img.get_fdata()

            # Create a downsampled mask: keep 1 in 100 voxels
            mask_indices = np.argwhere(mask_data > 0)
            np.random.shuffle(mask_indices)
            subsample_size = max(1, len(mask_indices) // 100)  # Ensure at least one voxel
            subsample_indices = mask_indices[:subsample_size]

            # Create a new empty mask
            debug_mask_data = np.zeros_like(mask_data)

            # Set the selected voxels to 1
            for idx in subsample_indices:
                debug_mask_data[tuple(idx)] = 1

            # Create a new Nifti image
            mask_img = image.new_img_like(mask_img, debug_mask_data)

        if return_masker:
            return NiftiMasker(mask_img=mask_img)

        return mask_img
