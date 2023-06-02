import logging
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from config import ProjectParams

from .subject import Subject
from .utils import check_dataset_existence, load_object, store_object, timer

logger = logging.getLogger(__name__)


@timer(
    enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
    logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
)
def create_dataset(
    project_params: ProjectParams, mask_roi: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches the preprocessed fMRI data and events from disk. If the data is not
    present on disk, it will create the dataset and save it to disk for future
    use.

    Parameters
    ----------
    project_params : ProjectParams
        A `ProjectParams` object that contains the necessary parameters for
        the project.
    mask_roi : str
        A string indicating the brain region of interest (ROI) to use for
        selecting voxels.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of three pandas DataFrames containing the preprocessed fMRI
        data, events, and confounds.

    Raises
    ------
    FileNotFoundError
        If any of the required files are missing from disk.
    """
    logger.info("===> Creating dataset from scratch...")
    all_bold_mni, all_events, all_confounds = [], [], []
    for subject in tqdm(
        project_params.SUBJECTS.value,
        desc=f"Preparing data for masked brain area '{mask_roi}'",
    ):
        new_subject = Subject(
            project_name=project_params.NAME.value,
            subject_id=subject,
            session_no=project_params.SESSION.value,
            mask_roi=mask_roi,
            t_r=project_params.TR_SECS.value,
        )
        bold_mni, events, confounds = new_subject.fetch_data(
            path_fmriprep=project_params.PATH_LOAD_FMRIPREP.value,
            path_subs=project_params.PATH_LOAD_BASE.value,
            path_masks=project_params.PATH_MASKS.value,
            runs=project_params.NO_RUNS.value,
            mask_type=project_params.MASK_TYPE.value,
            incl_confounds=project_params.INCL_CONFOUNDS.value,
        )
        all_bold_mni.append(bold_mni)
        all_events.append(events)
        all_confounds.append(confounds)

    return (
        pd.concat(all_bold_mni),
        pd.concat(all_events),
        pd.concat(all_confounds),
    )


@timer(
    enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
    logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
)
def get_data(
    mask_roi: str,
    store_format: str = "pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches the preprocessed fMRI data and events from disk. If the data is not
    present on disk, it will create the dataset and save it to disk for future
    use.

    Parameters
    ----------
    mask_roi : str
        A string indicating the brain region of interest (ROI) to use for
        selecting voxels.
    store_format : str, optional
        The format to use for storing the dataset (default is 'pkl').

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of three pandas DataFrames containing the preprocessed fMRI
        data, events, and confounds.

    Raises
    ------
    FileNotFoundError
        If any of the required files are missing from disk.
    """
    logger.info(
        f"===> Getting the fmriprep data for masked brain area '{mask_roi}'..."
    )
    path_data = (
        ProjectParams.PATH_INTERMEDIATE_DATA.value
        / mask_roi
        / ProjectParams.FILENAME_UNPROCESSED.value[0]
    )

    # Check if the data exists on disk
    if not check_dataset_existence(
        path_data=path_data,
        file_names=ProjectParams.FILENAME_UNPROCESSED.value[1],
        store_format=store_format,
    ):
        all_bold_mni, all_events, all_confounds = create_dataset(
            project_params=ProjectParams, mask_roi=mask_roi
        )
        for dataset, file_name in zip(
            (all_bold_mni, all_events, all_confounds),
            ProjectParams.FILENAME_UNPROCESSED.value[1],
        ):
            store_object(
                p_object=dataset,
                as_name=file_name,
                as_type=store_format,
                path=path_data,
            )

        return all_bold_mni, all_events, all_confounds
    else:
        loaded_datasets = []
        for dataset in ProjectParams.FILENAME_UNPROCESSED.value[1]:
            logger.info(f"Loading found dataset from file name '{dataset}'...")
            loaded_datasets.append(
                load_object(
                    from_name=dataset, from_type=store_format, path=path_data
                )
            )

        return tuple(loaded_datasets)
