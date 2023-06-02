import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from nilearn.input_data import NiftiMasker
from tqdm import tqdm

from config import ProjectParams

from .utils import timer

logger = logging.getLogger(__name__)


@dataclass
class Subject:
    """
    A class to represent a subject in the project.

    Attributes
    ----------
    project_name : str
        The name of the project.
    subject_id : str
        The identifier of the subject.
    session_no : int
        The number of the session.
    mask_roi : str
        The name of the mask ROI.
    t_r : float
        The time repetition of the functional data.
    """

    project_name: str
    subject_id: str
    session_no: int
    mask_roi: str
    t_r: float

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def get_mask(self, path_masks: Path, mask_type: str) -> np.ndarray:
        """
        Load and return a brain mask from a given path or using a default mask.

        Parameters
        ----------
        path_masks : Path
            Path to the directory containing the mask file.
        mask_type : str
            The category of the mask, either 'anat' or 'func'.

        Returns
        -------
        mask_data : numpy.ndarray
            A 3D binary array where voxels inside the brain are marked as True
            and voxels outside the brain are marked as False.

        """
        logger.info(
            f"Fetching '{mask_type}' mask for subject '{self.subject_id}'..."
        )

        if mask_type == "anatomical":
            mask_file = path_masks / mask_type / f"{self.mask_roi}.nii.gz"
        elif mask_type == "functional":
            mask_file = (
                path_masks
                / mask_type
                / self.mask_roi
                / f"sub_{self.subject_id}.nii.gz"
            )
        else:
            logger.error(f"'{mask_type}' is an unknown mask_type")
            raise ValueError(f"'{mask_type}' is an unknown mask_type")

        if not mask_file.exists():
            logger.error(f"Mask file not found: {mask_file}")
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        return NiftiMasker(str(mask_file), t_r=self.t_r)  # type: ignore

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def fetch_bold_mni(
        self,
        path_fmriprep: Path,
        mask: NiftiMasker,
        run_no: int,
    ) -> pd.DataFrame:
        """
        Fetch preprocessed fMRI data in MNI152 space for a specific run.

        Parameters
        ----------
        path_fmriprep : pathlib.Path
            Path to the preprocessed fMRI data.
        mask : Union[nilearn.input_data.NiftiMasker, numpy.ndarray]
            NiftiMasker instance used to mask the data.
        run_no : int
            Run number to fetch the data for.

        Returns
        -------
        pd.DataFrame
            Preprocessed fMRI data in MNI152 space for the specified run.
        """
        path_bold = (
            path_fmriprep
            / f"sub-{self.subject_id}"
            / f"ses-{self.session_no:02}"
            / "func"
            / f"sub-{self.subject_id}_ses-{self.session_no:02}_task-{self.project_name}_run-{run_no}_bold_space-MNI152NLin2009cAsym_preproc.nii"
        )
        if not path_bold.exists():
            raise FileNotFoundError(f"File not found: {path_bold}")

        bold_mni = pd.DataFrame(mask.fit_transform(str(path_bold)))
        bold_mni["run"] = run_no

        return bold_mni

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def fetch_events(self, path_subs: Path, run_no: int) -> pd.DataFrame:
        """
        Fetch and return a Pandas DataFrame containing the events associated
        with the specified run of the current subject.

        Parameters
        ----------
        path_subs : Path
            Path to the directory containing the subject data.
        run_no : int
            The number of the run for which to fetch the events.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the events for the specified run of the
            current subject.
        """
        path_events = (
            path_subs
            / f"sub-{self.subject_id}"
            / f"ses-{self.session_no:02}"
            / "func"
            / f"sub-{self.subject_id}_ses-{self.session_no:02}_task-{self.project_name}_run-{run_no}_events.tsv"
        )
        if not path_events.exists():
            logger.error(f"Events file not found: {path_events}")
            raise FileNotFoundError(f"Events file not found: {path_events}")

        events = pd.read_table(path_events)
        events["run"] = run_no

        return events

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def fetch_confounds(
        self,
        path_fmriprep: Path,
        incl_confounds: List[str],
        run_no: int,
    ) -> pd.DataFrame:
        """
        Fetch and return a DataFrame containing the confounds associated with a
        specific run of the subject.

        Parameters
        ----------
        path_fmriprep : Path
            Path to the directory containing the preprocessed fMRI data.
        incl_confounds : Tuple[str]
            Tuple of the names of the confounds to include in the returned
            DataFrame.
        run_no : int
            The number of the run for which to fetch the confounds.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the specified confounds for the specified
            run of the subject.
        """
        if not isinstance(incl_confounds, list):
            incl_confounds = list(incl_confounds)

        path_confounds = (
            Path(path_fmriprep)
            / f"sub-{self.subject_id}"
            / f"ses-{self.session_no:02}"
            / "func"
            / f"sub-{self.subject_id}_ses-{self.session_no:02}_task-{self.project_name}_run-{run_no}_bold_confounds.tsv"
        )
        if not path_confounds.exists():
            logger.error(f"Confounds file not found: '{path_confounds}'")
            raise FileNotFoundError(
                f"Confounds file not found: '{path_confounds}'"
            )

        confounds = pd.read_table(path_confounds)
        confounds = confounds[incl_confounds].fillna(method="bfill")
        confounds["run"] = run_no

        return confounds

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def fetch_data(
        self,
        path_fmriprep: Path,
        path_subs: Path,
        path_masks: Path,
        runs: int,
        mask_type: str,
        incl_confounds: Tuple[str],
        add_id: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetches fMRI bold signal, events and confounds data for a given
        subject.

        Parameters
        ----------
        path_fmriprep : Path
            Path to fmriprep output directory.
        path_subs : Path
            Path to the directory containing the events.tsv files.
        path_masks : Path
            Path to the directory containing the mask images.
        runs : int
            Number of runs in the fMRI data.
        mask_type : str
            Type of mask - "anat" or "func".
        incl_confounds : Tuple[str]
            Tuple of confounds to include in the final dataframe.
        add_id : bool, optional
            Whether or not to add subject id to the dataframes.
            Defaults to True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple containing the bold signal, events and confounds data.
        """
        mask = self.get_mask(path_masks=path_masks, mask_type=mask_type)

        logger.info(
            "Fetching fMRI-preprocessed BOLD data of subject"
            f" '{self.subject_id}'..."
        )
        bold_mni = [
            self.fetch_bold_mni(
                path_fmriprep=path_fmriprep,
                mask=mask,
                run_no=run,
            )
            for run in tqdm(
                range(1, runs + 1),
                desc=(
                    f"Fetching data from each run of subject {self.subject_id}"
                ),
            )
        ]
        logger.info(
            f"Fetching events tables of subject '{self.subject_id}'..."
        )
        events = [
            self.fetch_events(path_subs=path_subs, run_no=run)
            for run in range(1, runs + 1)
        ]

        logger.info(
            f"Fetching confounds tables of subjects '{self.subject_id}'..."
        )
        confounds = [
            self.fetch_confounds(
                path_fmriprep=path_fmriprep,
                incl_confounds=incl_confounds,
                run_no=run,
            )
            for run in range(1, runs + 1)
        ]
        self.bold_mni = pd.concat(bold_mni)
        self.events = pd.concat(events)
        self.confounds = pd.concat(confounds)

        if add_id:
            self.bold_mni["subject"] = self.subject_id
            self.events["subject"] = self.subject_id
            self.confounds["subject"] = self.subject_id

        return self.bold_mni, self.events, self.confounds
