import os
import os.path as op
import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import image, surface
from nilearn.input_data import NiftiMasker
from tqdm.contrib.itertools import product
from itertools import product as product_


def get_lr_direction(session, run, mb_orders):
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

def get_run_from_mb(mb, session, repetition, mb_orders):
    return mb_orders[session-1].index(mb) + (repetition-1) * 3 + 1
class Subject(object):


    def __init__(self, subject_id, bids_folder='/data/ds-siemenspilots'):
        if type(subject_id) == int:
            subject_id = f'{subject_id:02d}'
        
        self.subject_id = subject_id
        self.bids_folder = bids_folder

        self.derivatives_dir = op.join(bids_folder, 'derivatives')

    def get_mb_orders(self):

        if self.subject_id not in ['13']:
            return [[0, 2, 4], [4, 2, 0], [2, 0, 4]]
        else:
            return [[2, 4, 5], [5, 4, 2], [4, 2, 5]]


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

        mb_orders = self.get_mb_orders()

        df = []
        keys = []

        for task in tasks:
            if task == 'feedback':
                runs = [1, 5]
            elif task == 'estimation_task':
                runs = self.get_runs(session)

            for run in runs:
                try:
                    fn = op.join(behavior_folder, f'sub-{self.subject_id}_ses-{session}_task-{task}_run-{run}_events.tsv')
                    d = pd.read_csv(fn, sep='\t')

                    if (d['n'] > 25).any():
                        d['range'] = 'wide'
                    else:
                        d['range'] = 'narrow'

                    if isinstance(session, str) and session.startswith('philips'):
                        d['multiband'] = -1
                    else:
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
        if self.subject_id == 'alina':
            return [1,2,3, 'philips']
        elif self.subject_id in ['13', '41']:
            return [1,2,3, 'philips1', 'philips2']

    def get_bold(self, session, run=None, repetition=None, mb=None):

        assert(run is not None or ((repetition is not None) and (mb is not None))), 'Either run or repetition and mb must be specified'

        if run is None:
            assert mb in [0, 2, 4, 5], 'mb must be 0, 2 or 4'

            assert repetition in [1,2], 'repetition must be 1 or 2'
            run = get_run_from_mb(mb, session, repetition, self.get_mb_orders())


        mb_orders = self.get_mb_orders()

        if mb is None:
            mb = mb_orders[session-1][(run - 1) % 3]
            repetition = run // 3 + 1


        preproc_folder = Path(self.derivatives_dir, 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func')

        if session in ['philips', 'philips1', 'philips2']:
            assert mb in [-1, None]
            fn = preproc_folder / f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_space-T1w_desc-preproc_bold.nii.gz'
        else:

            if self.subject_id in ['13']:
                fn = preproc_folder / f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb{mb}_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz'
            else:
                lr_direction = get_lr_direction(session, run)
                fn = preproc_folder / f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb{mb}_dir-{lr_direction}_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz'

        assert(fn.exists()), f'File {fn} does not exist'

        return fn

    def get_onsets(self, session, run):

        onsets = pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', f'ses-{session}', 'func', f'sub-{self.subject_id}_ses-{session}_task-task_run-{run}_events.tsv'), index_col='trial_nr', sep='\t')

        return onsets

    def get_runs(self, session):
        if isinstance(session, str) and session.lower().startswith('philips'):
            return list(range(1, 9))  # 8 runs for Philips sessions
        return list(range(1, 7))  # 6 runs for other sessions

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

        if (multiband in [-1]) and (self.subject_id != 'alina'):
            n_volumes = 480
        else:
            n_volumes = 240

        assert(im.shape[3] == n_volumes), f'Expected {n_volumes} volumes, got {im.shape[3]}'

        return im

    def get_brain_mask(self, session=None, epi_space=True, return_masker=True, debug_mask=False):

        if not epi_space:
            raise ValueError('Only EPI space is supported')

        session = 1 if session is None else session

        # sub-alina_ses-1_task-numestimate_acq-mb0_dir-LR_run-01_desc-brain_mask.nii.gz
        if self.subject_id in ['13']:
            fn = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}',
                        f'ses-{session}', 'func', 
                        f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb2_run-01_space-T1w_desc-brain_mask.nii.gz')
        else:
            fn = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}',
                        f'ses-{session}', 'func', 
                        f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb0_dir-LR_run-01_space-T1w_desc-brain_mask.nii.gz')

        mask_img = fn

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

    def get_surf_info(self):
        info = {'L':{}, 'R':{}}

        for hemi in ['L', 'R']:

            fs_hemi = {'L':'lh', 'R':'rh'}[hemi]

            info[hemi]['inner'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'anat', f'sub-{self.subject_id}_hemi-{hemi}_white.surf.gii')
            info[hemi]['mid'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'anat', f'sub-{self.subject_id}_hemi-{hemi}_thickness.shape.gii')
            info[hemi]['outer'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'anat', f'sub-{self.subject_id}_hemi-{hemi}_pial.surf.gii')
            # info[hemi]['inflated'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject_id}', 'ses-1', 'anat', f'sub-{self.subject_id}_ses-1_hemi-{hemi}_inflated.surf.gii')
            info[hemi]['curvature'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer', f'sub-{self.subject_id}', 'surf', f'{fs_hemi}.curv')

            for key in info[hemi]:
                assert(op.exists(info[hemi][key])), f'{info[hemi][key]} does not exist'

        return info

    def get_prf_parameters_volume(self,
            multiband,
            smoothed=True,
            keys=None,
            roi=None,
            return_image=False,
            gaussian=True,
            model_label=4):

        dir = 'encoding_model'

        dir += f'.model{model_label}'
        
        if gaussian:
            dir += '.gaussian'
        else:
            dir += '.logspace'

        if smoothed:
            dir += '.smoothed'

        parameters = []

        assert keys is None or 'r2' not in keys, 'r2 is always included'

        if keys is None:
            keys = ['mu', 'sd', 'amplitude', 'baseline']

        masker = self.get_volume_mask(roi=roi, epi_space=True, return_masker=True)

        # sub-alina_acq-0_desc-amplitude.narrow.optim_space-T1w_pars.nii.gz
        fn_template = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func',
                              'sub-{subject_id}_acq-mb{multiband}_desc-{parameter_key}.{range_n}.optim_space-T1w_pars.nii.gz')

        for parameter_key, range_n in product_(keys, ['narrow', 'wide']):
            fn = fn_template.format(parameter_key=parameter_key, multiband=multiband, subject_id=self.subject_id, range_n=range_n)
            pars = pd.Series(masker.transform(fn).squeeze(), name=(parameter_key, range_n))
            parameters.append(pars)


        r2_fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject_id}', 'func', f'sub-{self.subject_id}_acq-mb{multiband}_desc-r2.optim_space-T1w_pars.nii.gz')

        parameters.append(pd.Series(masker.transform(r2_fn).squeeze(), name=('r2', None)))
        keys.append(['r2'])

        parameters =  pd.concat(parameters, axis=1, names=['parameter', 'range']).astype(np.float32)

        if return_image:
            return masker.inverse_transform(parameters.T)

        return parameters

    def get_volume_mask(self, roi=None, session=1, epi_space=False, return_masker=False, verbose=False):
        """Retrieve a volume mask, optionally in EPI space and as a NiftiMasker."""

        def _log(msg):
            """Helper function for verbose logging."""
            if verbose:
                print(msg)

        _log(f"Session: {session}, ROI: {roi}, epi_space: {epi_space}, return_masker: {return_masker}")

        # Load the base brain mask
        base_mask_path = op.join(
            self.bids_folder, 'derivatives', f'fmriprep/sub-{self.subject_id}/ses-{session}/func',
            # sub-alina_ses-1_task-numestimate_acq-mb0_dir-LR_run-01_space-T1w_desc-brain_mask.nii.gz
            f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb0_dir-LR_run-01_space-T1w_desc-brain_mask.nii.gz'
        )
        base_mask = image.load_img(base_mask_path, dtype='int32')  # Prevent weird nilearn warning

        # Resample to first functional run (ensuring affine match)
        first_run = self.get_bold(session=session, run=1)
        base_mask = image.resample_to_img(base_mask, first_run, interpolation='nearest', force_resample=False, copy_header=True)
        _log("Base mask loaded and resampled.")

        # Handle case when ROI is None
        if roi is None:
            if epi_space:
                mask = base_mask
            else:
                _log("ROI is None and epi_space=False -> Raising NotImplementedError")
                raise NotImplementedError

        # Handle anatomical masks
        elif roi.startswith(('NPC', 'NF', 'NTO')):  # Tuple avoids multiple `or` conditions
            _log(f"Processing ROI: {roi}")

            anat_mask_path = op.join(
                self.derivatives_dir, 'ips_masks', f'sub-{self.subject_id}', 'anat',
                f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz'
            )

            if epi_space:
                epi_mask_path = op.join(
                    self.derivatives_dir, 'ips_masks', f'sub-{self.subject_id}', 'func',
                    f'ses-{session}', f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz'
                )

                if not op.exists(epi_mask_path):
                    _log(f"EPI mask does not exist: {epi_mask_path}, creating it.")

                    # Ensure parent directory exists
                    os.makedirs(op.dirname(epi_mask_path), exist_ok=True)

                    # Resample anatomical mask to EPI space and save
                    im = image.resample_to_img(
                        image.load_img(anat_mask_path, dtype='int32'),
                        image.load_img(base_mask, dtype='int32'),
                        interpolation='nearest'
                    )
                    im.to_filename(epi_mask_path)
                    _log(f"Saved new EPI mask: {epi_mask_path}")

                mask = epi_mask_path
            else:
                mask = anat_mask_path

        else:
            _log(f"Unknown ROI: {roi} -> Raising NotImplementedError")
            raise NotImplementedError

        _log(f"Final mask path: {mask}")

        # Load the final mask as a Nifti1Image
        mask = image.load_img(mask, dtype='int32')
        _log("Loaded final mask into Nifti1Image.")

        # Return either a NiftiMasker or the raw mask
        if return_masker:
            _log("Returning NiftiMasker")
            masker = NiftiMasker(mask_img=mask, resampling_target="data")  # Ensures compatibility with input images
            masker.fit()
            return masker

        _log("Returning final mask as Nifti1Image.")
        return mask

    def get_prf_parameters_surf(self, multiband, model_label=4, smoothed=False, hemi=None, space='fsnative', gaussian=True):

        parameter_keys = ['mu.narrow', 'mu.wide',
                    'sd.narrow', 'sd.wide',
                    'amplitude.narrow', 'amplitude.wide',
                    'baseline.narrow', 'baseline.wide',
                    'r2']        

        if hemi is None:
            prf_l = self.get_prf_parameters_surf(multiband, model_label, smoothed, hemi='L', space=space)
            prf_r = self.get_prf_parameters_surf(multiband, model_label, smoothed, hemi='R', space=space)
            
            return pd.concat((prf_l, prf_r), axis=0, 
                    keys=pd.Index(['L', 'R'], name='hemi'))

        key = 'encoding_model'

        key += f'.model{model_label}'

        if gaussian:
            key += '.gaussian'
        else:
            raise NotImplementedError

        if smoothed:
            key += '.smoothed'

        parameters = []

        dir = op.join(self.bids_folder, 'derivatives', key, f'sub-{self.subject_id}', 'func')
        fn_template =  op.join(dir, 'sub-{subject_id}_acq-mb{multiband}_desc-{parameter_key}.optim_space-fsnative_hemi-{hemi}.func.gii')

        for parameter_key in parameter_keys:

            fn = fn_template.format(multiband=multiband, parameter_key=parameter_key, subject_id=self.subject_id, hemi=hemi, space=space)

            pars = pd.Series(surface.load_surf_data(fn))
            pars.index.name = 'vertex'

            parameters.append(pars)

        return pd.concat(parameters, axis=1, keys=parameter_keys, names=['parameter'])

    def get_confounds(self, session=1, run=1, confounds=None):

        def filter_confounds(confounds, n_acompcorr=10):
            confound_cols = ['dvars', 'framewise_displacement']

            a_compcorr_cols = [f"a_comp_cor_{i:02d}" for i in range(n_acompcorr)]
            confound_cols += a_compcorr_cols

            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            motion_cols += [f'{e}_derivative1' for e in motion_cols]
            confound_cols += motion_cols

            steady_state_cols = [c for c in confounds.columns if 'non_steady_state' in c]
            confound_cols += steady_state_cols

            outlier_cols = [c for c in confounds.columns if 'motion_outlier' in c]
            confound_cols += outlier_cols

            cosine_cols = [c for c in confounds.columns if 'cosine' in c]
            confound_cols += cosine_cols

            
            return confounds[confound_cols].fillna(0)

        mb_orders = self.get_mb_orders()
        mb = mb_orders[session-1][(run - 1) % 3]
        
        preproc_folder = Path(self.derivatives_dir, 'fmriprep', f'sub-{self.subject_id}', f'ses-{session}', 'func')

        if self.subject_id in ['13']:
            fn = preproc_folder / f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb{mb}_run-{run:02d}_desc-confounds_timeseries.tsv'
        else:
            lr_direction = get_lr_direction(session, run)
            fn = preproc_folder / f'sub-{self.subject_id}_ses-{session}_task-numestimate_acq-mb{mb}_dir-{lr_direction}_run-{run:02d}_desc-confounds_timeseries.tsv'

        confounds = pd.read_csv(fn, sep='\t')
        confounds = filter_confounds(confounds)

        return confounds