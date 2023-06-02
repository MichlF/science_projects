import logging

from tqdm import tqdm

from config import ProjectParams
from src.logging.logger import init_logger
from src.perform_decoding import perform_decoding
from src.perform_deconvolution import perform_deconvolution
from src.perform_rdm_analysis import perform_rdm
from src.prepare_dataset import get_data

# Initialize the logger
init_logger(logging_level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:

    for roi in tqdm(ProjectParams.MASK_LIST.value):
        logger.info(
            "************ Starting analysis of project"
            f" {ProjectParams.NAME.value} for area {roi} ************\n"
        )

        bold_mni, events, confounds = get_data(mask_roi=roi)

        timecourses = perform_deconvolution(
            mask_roi=roi,
            bold_mni=bold_mni,
            events=events,
            confounds=confounds,
            generate_plots=True,
        )

        perform_decoding(
            timecourses=timecourses, mask_roi=roi, generate_plots=True
        )

        perform_rdm(timecourses=timecourses, mask_roi=roi, generate_plots=True)


if __name__ == "__main__":
    main()
