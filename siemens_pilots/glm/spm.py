def get_tr(mb):

    if mb == 2:
        return 1.33
    elif mb == 4:
        return 0.681
    elif mb == 5:
        return 0.549
    
def get_mask(subject):
    from siemens_pilots.utils.data import Subject
    sub = Subject(subject)
    return  sub.get_brain_mask(return_masker=False, epi_space=True)
    

def get_subject_info(subject, mb):
    from siemens_pilots.utils.data import Subject
    from siemens_pilots.utils.data import get_run_from_mb
    from nipype.interfaces.base import Bunch
    import numpy as np
    from scipy.stats import zscore

    sub = Subject(subject)
    mb_orders = sub.get_mb_orders()

    subject_info = []
    functional_runs = []

    for session in [1, 2, 3]:
        for repetition in [1, 2]:

            run = get_run_from_mb(mb, session, repetition, mb_orders)
            data = sub.get_onsets(session, run)

            functional_runs.append(str(sub.get_bold(session, mb=mb, run=run)))
            confounds = sub.get_confounds(session, run)

            print(functional_runs)

            onsets = []
            conditions = []
            durations = []
            pmod = []
            regressor_names =['stimulus', 'response']

            pmod = [Bunch(name=["presented_n"], param=[zscore(np.nan_to_num(data[data['trial_type'] == 'stimulus']['n'].values)).tolist()], poly=[1]),
                    Bunch(name=["responded_n"], param=[zscore(np.nan_to_num(data[data['trial_type'] == 'response']['response'].values)).tolist()], poly=[1]), ]

            for event_type in regressor_names:
                print(event_type)
                onsets.append(data[data.trial_type == event_type].onset.values.tolist())
                print(len(onsets))
                conditions.append(event_type)
                durations.append([1])

            subject_info.append(Bunch(
                conditions=conditions,
                onsets=onsets,
                durations=durations,
                pmod=pmod,
                regressors=confounds.values.T.tolist(),
                regressor_names=confounds.columns.values.tolist(),
            ))
    
    return subject_info, functional_runs


def get_contrasts():
    condition_names = [
        "stimulus",
        "response",
        "stimulusxpresented_n^1",
        "responsexresponded_n^1",
    ]

    con01 = ["stimulus", "T", condition_names[:1], [1 / 3.0]]
    con02 = ["response", "T", condition_names[1:2], [1 / 3.0]]
    con03 = ["presented n", "T", condition_names[2:3], [1 / 3.0]]
    con04 = ["responsed n", "T", condition_names[3:4], [1 / 3.0]]

    return [
        con01,
        con02,
        con03,
        con04,
]
