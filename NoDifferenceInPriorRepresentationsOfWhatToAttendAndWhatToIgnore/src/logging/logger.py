import json
import logging
import logging.config as logging_config
import os
from typing import Union

import yaml


def init_logger(
    config: Union[str, dict] = "src/logging/logging.yaml",
    env_key: str = "LOG_CFG",
    logging_level: Union[int, str] = "INFO",
) -> None:
    """
    Initializes a logger with settings from a configuration file.

    Parameters
    ----------
    config : str or dict, optional
        Path to a YAML or JSON file containing the logger settings, or a Python
        dictionary containing the logger configuration. By default, the
        function looks for a file named "logging.yaml" in the current
        directory.
    env_key : str, optional
        Name of the environment variable that can be used to specify the path
        to the configuration file. By default, "LOG_CFG".
    logging_level : int or str, optional
        Python logging object determining the logging level. By default,
        None, which uses the logging level specified in the configuration file.
        If a logging level is specified here, it overrides the logging level
        in the configuration file.

    Raises
    ------
    ValueError
        If the configuration file format is not supported.
    """
    if isinstance(config, str):
        path = os.getenv(env_key, config)
        with open(path, "rt") as f:
            if path.endswith(".yaml"):
                config = yaml.safe_load(f.read())
            elif path.endswith(".json"):
                config = json.load(f)
            else:
                raise ValueError("Unsupported configuration file format")
    # Older versions of Python don't support logging.config.dictConfig
    try:
        logging.config.dictConfig(config)  # type: ignore
    except Exception:
        logging_config.dictConfig(config)  # type: ignore

    # Override logging level, if requested
    if logging_level is not None:
        logging.getLogger().setLevel(logging_level)

    logging.info("Logging started...")
