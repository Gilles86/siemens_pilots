import argparse
from siemens_pilots.utils.data import Subject
import pandas as pd
import os.path as op
import os


def main(subject, session, bids_folder):

    subject = Subject(subject, bids_folder=bids_folder)

    d = subject.get_behavioral_data(session=session, tasks=['estimation_task'], raw=True, add_info=False).droplevel([0, 1])
    print(d.iloc[:25])
    onset0 = d.groupby('run').apply(lambda x: x.xs('pulse', 0, 'event_type')['onset'].iloc[0])
    d['onset'] -= onset0.loc[d.index.get_level_values('run')].values
    d['trial_nr'] = (d.index.get_level_values('run') - 1) * 40 + d['trial_nr']
    d = d.set_index('trial_nr', append=True) 

    d = d[d['onset'] > 0.0]
    d = d[d.response != 'num_subtract']

    stimulus = d.xs('stimulus', 0, 'event_type')[['onset', 'n']]
    stimulus['trial_type'] = 'stimulus'
    print(stimulus)

    response = d.xs('response', 0, 'event_type')[['onset']].join(d.xs('feedback', 0, 'event_type')['response'])
    response['trial_type'] = 'response'
    response = response.groupby(['run', 'trial_nr']).first()
    print(response)


    assert(len(response) == len(stimulus))

    onsets = pd.concat((stimulus, response)).groupby('run', ).apply(lambda x: x.sort_values('onset')).reset_index('trial_nr')

    target_dir = op.join(bids_folder, f'sub-{subject.subject_id}', f'ses-{session}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run in subject.get_runs(session):
        target_fn = op.join(target_dir, f'sub-{subject.subject_id}_ses-{session}_task-task_run-{run}_events.tsv')
        print(f'Saving {target_fn}')

        onsets.loc[run].to_csv(target_fn, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('session', type=int)
    parser.add_argument('--bids_folder', default='/data/ds-siemenspilots')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder)