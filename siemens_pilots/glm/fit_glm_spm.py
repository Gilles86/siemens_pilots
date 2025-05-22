import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import IdentityInterface
from nipype.algorithms.modelgen import SpecifySPMModel, SpecifyModel
import argparse
from nipype import Node, Workflow
from nipype.interfaces.spm import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces import spm
from nipype.algorithms.misc import Gunzip
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
from nipype.interfaces.matlab import MatlabCommand
from pathlib import Path
import os

from spm import (
    get_subject_info,
    get_tr,
    get_mask,
    get_contrasts
)


def main(subject, mb, bids_folder='/data/ds-siemenspilots'):

    print('yo')

    # MatlabCommand.set_default_matlab_cmd('/Applications/MATLAB_R2022b.app/bin/matlab')
    # MatlabCommand.set_default_paths(str(Path(os.environ['HOME']) / 'spm12'))
    spm.SPMCommand.set_mlab_paths(matlab_cmd='/Applications/MATLAB_R2022b.app/bin/matlab')

    input_node = pe.Node(niu.IdentityInterface(fields=["subject", "mb"]),
                        name='inputnode')
    input_node.inputs.subject = subject
    input_node.inputs.mb = mb

    getsubjectinfo = pe.Node(
            Function(
                input_names=["subject", "mb"],
                output_names=["subject_info", "functional_runs"],
                function=get_subject_info,
            ),
            name="getsubjectinfo",
        )


    modelspec = pe.Node(
        SpecifySPMModel(
            concatenate_runs=False,
            input_units="secs",
            output_units="secs",
            # time_repetition=tr,
            high_pass_filter_cutoff=128,
        ),
        name="modelspec",
    )

    level1design = pe.Node(
        Level1Design(
            bases={"hrf": {"derivs": [0, 0]}},
            timing_units="secs",
            # interscan_interval=TR,
            model_serial_correlations="AR(1)",
            microtime_resolution=40,
            microtime_onset=20,
            flags={"mthresh": 0.8, "global": "None"},
            # mask_image=mask,
            # volterra_expansion_order=1,
        ),
        name="level1design",
    )

    level1estimate = pe.Node(
        EstimateModel(estimation_method={"Classical": 1}, write_residuals=False),
        name="level1estimate",
    )

    level1conest = pe.Node(EstimateContrast(), name="level1conest")

    wf = Workflow(name=f"level1_workflow_sub-{subject}_mb-{mb}", base_dir='/tmp/workflow_folders')

    smoother = pe.Node(spm.Smooth(fwhm=[6, 6, 6]), name="smoother")

    unzip_functional_files = pe.MapNode(Gunzip(), name="unzip_functional_files", iterfield=["in_file"])
    unzip_mask = pe.Node(Gunzip(), name="unzip_mask")

    level1conest = pe.Node(EstimateContrast(), name="level1conest")

    level1conest.inputs.contrasts = get_contrasts()

    datasink = pe.Node(nio.DataSink(base_directory=str(Path(bids_folder) / 'derivatives' / 'level1' / f'sub-{subject}' / f'mb-{mb}' / 'func')),
                name="datasink")

        
    wf.connect([
        (input_node, getsubjectinfo, [("subject", "subject"), ("mb", "mb")]),
        (input_node, modelspec, [(("mb", get_tr), "time_repetition")]),
        (input_node, level1design, [(("mb", get_tr), "interscan_interval")]),
        (input_node, unzip_mask, [(("subject", get_mask), "in_file")]),
        (unzip_mask, level1design, [('out_file', "mask_image")]),
        (getsubjectinfo, unzip_functional_files, [('functional_runs', "in_file")]),
        (unzip_functional_files, smoother, [('out_file', "in_files")]),
        (smoother, modelspec, [("smoothed_files", "functional_runs"),]),
        (getsubjectinfo, modelspec, [("subject_info", "subject_info")]),
        (modelspec, level1design, [("session_info", "session_info")]),
        (level1design, level1estimate, [("spm_mat_file", "spm_mat_file")]),
        ( level1estimate, level1conest, [
            ("spm_mat_file", "spm_mat_file"),
            ("beta_images", "beta_images"),
            ("residual_image", "residual_image"), ],),
        (level1conest, datasink, [("spmT_images", "contrasts"),
                                ("spm_mat_file", "contrasts.@spm_mat_file")]),
    ])


    wf.run(plugin="MultiProc", plugin_args={"n_procs": 4})
    wf.write_graph(graph2use='flat', format='png', simple_form=True)

if __name__ == "__main__":
    print('yo1')
    parser = argparse.ArgumentParser(description="Fit GLM using SPM")
    parser.add_argument( "subject", type=int, help="Subject ID",)
    parser.add_argument( "mb", type=int, help="MB factor",)
    args = parser.parse_args()

    main(args.subject, args.mb)