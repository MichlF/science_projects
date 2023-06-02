import logging

from config_masks import Masks

from src.logging.logger import init_logger
from src.perform_beh_paper_prep import perform_beh_paper_prep
from src.perform_brain_paper_prep import perform_brain_paper_prep

# Initialize the logger
init_logger(logging_level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting paper preparations")
    perform_beh_paper_prep()
    perform_brain_paper_prep(mask_rois=Masks.SELECTED.value)
    logger.info("Done")


if __name__ == "__main__":
    main()
