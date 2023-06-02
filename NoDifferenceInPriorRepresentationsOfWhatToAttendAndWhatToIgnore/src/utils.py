import functools
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t, ttest_ind, ttest_rel

from config import PlottingParams, ProjectParams

logger = logging.getLogger(__name__)


def timer(
    enabled: bool = True,
    decimal_places: int = 4,
    logging_active: bool = False,
):
    """
    A decorator that times the execution of the decorated function and prints
    the time taken.

    Parameters
    ----------
    enabled : bool, optional
        A flag to enable/disable the timer. Default is True.
    decimal_places : int, optional
        The number of decimal places to round the time to. Default is 4.
    logging_active : bool, optional
        A flag to enable/disable logging output. Default is False.

    Returns
    -------
    Callable[..., Any]
        The decorated function that prints the execution time
        if the timer is enabled.
    """

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if enabled:
                before = perf_counter()
                value = function(*args, **kwargs)
                after = perf_counter()
                time_diff = after - before
                formatted_time = f"{time_diff:.{decimal_places}f}"
                if logging_active:
                    logger.debug(
                        f"Function {function.__name__} took"
                        f" {formatted_time} secs to run."
                    )
                else:
                    print(
                        f"Function {function.__name__} took"
                        f" {formatted_time} secs to run."
                    )
            else:
                value = function(*args, **kwargs)
            return value

        return wrapper

    return decorator


@timer(enabled=True, logging_active=True)
def store_object(p_object, as_name: str, as_type: str, path: Path):
    """
    Stores an object to disk in a specified format.

    Parameters
    ----------
    p_object : Any
        The object to be stored.
    as_name : str
        The name of the file to be saved.
    as_type : str
        The format to be used for saving (either "pkl" or "npy").
    path : Path
        The directory path to save the file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an unknown format is specified for as_type.

    Example
    -------
    This function can be used to store an object as follows:

        >>> data = [1, 2, 3, 4, 5]
        >>> store_object(data, "data", "npy", Path.cwd())

    Notes
    -----
    The supported formats are "pkl" for pickle and "npy" for numpy.
    The function will raise a ValueError if an unknown format is
    specified for `as_type`.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"{path} did not exist. Creating it...")

    if as_type == "pkl":
        if isinstance(p_object, pd.DataFrame):
            p_object.to_pickle(path / f"{as_name}.{as_type}")
        else:
            with open(path / f"{as_name}.{as_type}", "wb") as f:
                pickle.dump(p_object, f)
    elif as_type == "npy":
        np.save(path / f"{as_name}.{as_type}", p_object)
    else:
        logger.error(f"'{as_type}' is an unknown type.")
        ValueError(f"'{as_type}' is an unknown type.")
    logger.info(f"Saved object '{as_name}.{as_type}' in '{path}'")


@timer(enabled=True, logging_active=True)
def load_object(from_name: str, from_type: str, path: Path):
    """
    Loads an object from a file.

    Parameters:
    -----------
    from_name : str
        The name of the file to load the object from.
    from_type : str
        The type of the file to load the object from. Currently supported types
        are 'pkl' and 'npy'.
    path : pathlib.Path
        The path to the directory containing the file.

    Returns:
    --------
    The object loaded from the specified file.

    Usage example:
    --------------
    >>> from_name = "my_data"
    >>> from_type = "pkl"
    >>> path = Path.cwd() / "data"
    >>> loaded_data = load_object(from_name, from_type, path)
    >>> print(loaded_data)
    [1, 2, 3, 4, 5]
    """
    logger.info(f"Loading object '{from_name}.{from_type}' from '{path}'")
    if from_type == "pkl":
        try:
            dataframe = pd.read_pickle(path / f"{from_name}.pkl")
            return dataframe
        except Exception:
            with open(path / f"{from_name}.pkl", "rb") as f:
                dataclass = pickle.load(f)
            return dataclass
    elif from_type == "npy":
        return np.load(path / f"{from_name}.npy")
    else:
        logger.error(f"'{from_type}' is an unknown type.")
        ValueError(f"'{from_type}' is an unknown type.")
    logger.info(f"Loaded object '{from_name}.{from_type}' from '{path}'")


@timer(enabled=True, logging_active=True)
def check_dataset_existence(
    path_data: Path, file_names: Tuple[str, ...], store_format: str
) -> bool:
    """
    Checks if the required files exist on disk.

    Args:
        path_data: A `Path` object indicating the path to the folder
        containing the data.
        file_names: A tuple of strings indicating the names of the required
        files.

    Returns:
        A boolean indicating whether or not all the required files are present
        on disk.
    """
    if not isinstance(file_names, tuple):
        file_names = tuple([file_names])

    missing_files = [
        name
        for name in file_names
        if not (path_data / f"{name}.{store_format}").exists()
    ]
    if missing_files:
        logger.info(
            f"The following files were not found at '{path_data}':"
            f" '{missing_files}'."
        )
        return False
    else:
        logger.info(f"Found a dataset for '{file_names=}'. Skipping creation.")
        return True


def style_plot_periods(
    plotting_params: PlottingParams,
    periods: Union[pd.DataFrame, pd.Series],
    ax: plt.Axes,
    h_line_reference: Union[int, float] = 0.0,
) -> plt.Axes:
    # Add x-ticks
    ax.set_xticklabels(
        list(periods.index.get_level_values("condition").unique()),
        rotation=45,
    )
    # Add horizontal line for zero or chance level reference
    ax.axhline(
        h_line_reference,
        color=plotting_params.HLINE_COLOR.value,
        lw=plotting_params.LINEWIDTH.value,
        linestyle=plotting_params.HLINE_LINESTYLE.value,
    )
    sns.despine(ax=ax, trim=plotting_params.TRIM.value)

    return ax


def style_plot_timecourse(
    project_params: ProjectParams,
    plotting_params: PlottingParams,
    timecourses: pd.DataFrame,
    ax: plt.Axes,
    h_line_reference: float = 0.0,
) -> plt.Axes:
    """
    Apply styling to a given plot axes.

    Parameters
    ----------
    project_params : ProjectParams
        An instance of the ProjectParams class containing project-specific
        parameters.
    plotting_params : PlottingParams
        An instance of the PlottingParams class containing
        plotting-specific parameters.
    timecourses : pd.DataFrame
        A pandas DataFrame containing the timecourses data for all
        subjects.
    ax : plt.Axes
        The plot axes to apply the styling to.

    Returns
    -------
    plt.Axes
        The plot axes with the applied styling.
    """
    # Add x-ticks and labels
    time_vector = np.arange(
        timecourses.index.get_level_values("time").values[0],
        timecourses.index.get_level_values("time").values[-1],
        step=project_params.TR_SECS.value * 2,
    )
    ax.set_xticks(time_vector)
    ax.set_xticklabels(
        np.round(time_vector + plotting_params.ONSET_CUE.value, 1)
    )
    # Add horizontal line for zero reference
    ax.axhline(
        h_line_reference,
        color=plotting_params.HLINE_COLOR.value,
        lw=plotting_params.LINEWIDTH.value,
        linestyle=plotting_params.HLINE_LINESTYLE.value,
    )
    # Add vertical line for important onsets
    ax.axvline(
        plotting_params.ONSET_CUE.value
        + project_params.INTERVALS_SECS.value[0],
        color=plotting_params.VLINE_COLOR.value,
        lw=plotting_params.LINEWIDTH.value - 0.5,
        linestyle=plotting_params.VLINE_LINESTYLE.value,
        dash_capstyle=plotting_params.VLINE_CAPSTYLE.value,
        dash_joinstyle=plotting_params.VLINE_JOINSTYLE.value,
    )
    ax.axvline(
        plotting_params.ONSET_SEARCH.value
        + project_params.INTERVALS_SECS.value[0],
        color=plotting_params.VLINE_COLOR.value,
        lw=plotting_params.LINEWIDTH.value - 0.5,
        linestyle=plotting_params.VLINE_LINESTYLE.value,
        dash_capstyle=plotting_params.VLINE_CAPSTYLE.value,
        dash_joinstyle=plotting_params.VLINE_JOINSTYLE.value,
    )
    # Add shaded areas for TROIs and annotate
    for troi, troi_name, onset in zip(
        project_params.TROIS.value,
        project_params.TROIS_NAME.value,
        (
            plotting_params.ONSET_CUE.value,
            plotting_params.ONSET_SEARCH.value,
        ),
    ):
        ax.axvspan(
            troi[0],
            troi[1],
            alpha=plotting_params.SHADED_ALPHA.value,
            color=plotting_params.SHADED_COLOR.value,
        )
        longest_troi_name = max(project_params.TROIS_NAME.value, key=len)
        ax.annotate(
            text=f" {troi_name.upper():<{len(longest_troi_name)}}",
            xy=(
                onset + project_params.INTERVALS_SECS.value[0],
                ax.get_ylim()[1],
            ),
            annotation_clip=False,
            rotation=plotting_params.ANNOT_ROTATION.value,
            va="top",
            fontsize=plotting_params.ANNOT_SIZE.value,
            alpha=plotting_params.ANNOT_ALPHA.value,
            backgroundcolor=plotting_params.ANNOT_BGCOLOR.value,
        )
        sns.despine(ax=ax, trim=plotting_params.TRIM.value)

    return ax


def within_ci(
    data: np.ndarray,
    p_value: float = 0.05,
    tail: str = "two",
    morey: bool = True,
) -> np.ndarray:
    """
    within_confidence_int

    Cousineau's method (2005) for calculating within-subject confidence
    intervals. If needed, Morey's correction (2008) can be applied
    (recommended).

    Parameters
    ----------
    data : ndarray
        Data for which CIs should be calculated
    p_value : float, optional
        p-value for determining t-value (the default is .05).
    tail : string, optional
        Two-tailed ('two') or one-tailed t-value.
    morey : bool, optional
        Apply Morey correction (the default is True)

    Returns
    -------
    confidence_intervals : ndarray
        Confidence intervals for each condition
    """
    assert tail in [
        "two",
        "one",
    ], f"tail must be one of 'two' or 'one', not {tail}"
    if tail == "two":
        p_value = p_value / 2

    # Normalize by subtracting participants mean performance from
    # each observation, and then add grand mean to each observation
    individual_means = data.mean(axis=1).reshape(data.shape[0], 1)
    grand_mean = data.mean(axis=1).mean()
    data = data - individual_means + grand_mean
    # Look up t-value and calculate CIs
    t_value = abs(t.ppf([p_value], data.shape[0] - 1)[0])
    confidence_intervals = (
        data.std(axis=0, ddof=1) / np.sqrt(data.shape[0]) * t_value
    )
    # Apply Morrey (2008) correction
    if morey:
        confidence_intervals = confidence_intervals * (
            data.shape[1] / float((data.shape[1] - 1))
        )

    return confidence_intervals


def clusterbased_permutation(
    X1: np.ndarray,
    X2: Union[np.ndarray, float],
    p_val: float = 0.05,
    cl_p_val: float = 0.05,
    paired: bool = True,
    tail: str = "both",
    n_permutations: int = 1000,
    mask: Optional[np.ndarray] = None,
    conn: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Implements Maris, E., & Oostenveld, R. (2007). Nonparametric statistical
    testing of EEG- and MEG- data. Journal of Neurosience Methods, 164(1),
    177-190. http://doi.org/10.1016/J.Jneumeth.2007.03.024

    Arguments
    - - - - -

    X1 (array):
        subject X dim1 X dim2 (optional), where dim1 and dim2 can be any type
        of dimension (time, frequency, electrode, etc). Values inarray
        represent some dependent measure (e.g classification accuracy or power)
    X2 (array | float):
        either a datamatrix with same dimensions as X1, or a single value
        against which X1 will be tested
    p_val (float):
        p_value used for inclusion into the cluster
    cl_p_val (float):
        p_value for evaluation overall cluster significance
    paired (bool):
        paired t testing (True) or independent t testing (False)
    tail (str):
        apply one- or two- tailed t testing
    n_permutations (int):
        number of permutations
    mask (array):
        dim1 X dim2 array. Can be used to restrict cluster based
        test to a specific region.
    conn (array):
        outlines which dim1 points are connected to other dim1
        points. Usefull when doing a cluster based permutation test across
        electrodes

    Returns
    - - - -

    cl_p_vals (array): dim1 X dim2 with p-values < cl_p_val for significant
    clusters and 1's for all other clusters

    """
    # if no mask is provided include all datapoints in analysis
    if mask is None:
        mask = np.array(np.ones(X1.shape[1:]), dtype=bool)
        print(
            f"\nUsing all {mask.size} datapoints in cluster based permutation"
        )
    elif mask.shape != X1[0].shape:
        print("\nMask does not have the same shape as X1. Adjust mask!")
    else:
        print(
            f"\nThere are {int(mask.sum())} out of {mask.size} datapoints in"
            " your mask during cluster based permutation"
        )
    # Check whether X2 is a chance variable or a data array
    if isinstance(X2, (float, int)):
        X2 = np.tile(X2, X1.shape)

    # compute observed cluster statistics
    (
        pos_sizes,
        neg_sizes,
        pos_labels,
        neg_labels,
        sig_cl,
    ) = compute_clustersizes(
        X1=X1,
        X2=X2,
        mask=mask,
        paired=paired,
        p_val=p_val,
        conn=conn,
    )
    cl_p_vals = np.ones(sig_cl.shape)

    # Determine how often permuted clusters exceed observed cluster threshold
    c_pos_cl = np.zeros(np.max(np.unique(pos_labels)))
    c_neg_cl = np.zeros(np.max(np.unique(neg_labels)))

    # initiate random arrays
    X1_rand = np.zeros(X1.shape)
    X2_rand = np.zeros(X1.shape)

    print("Permutating...")
    for permutation in range(n_permutations):
        if verbose:
            print(
                f"{(float(permutation)/n_permutations)*100}% of permutations\n"
            )

        # create random partitions
        if paired:  # keep observations paired under permutation
            rand_idx = np.random.rand(X1.shape[0]) < 0.5
            X1_rand[rand_idx, :] = X1[rand_idx, :]
            X1_rand[~rand_idx, :] = X2[~rand_idx, :]
            X2_rand[rand_idx, :] = X2[rand_idx, :]
            X2_rand[~rand_idx, :] = X1[~rand_idx, :]
        else:  # fully randomize observations under permutation
            all_X = np.vstack((X1, X2))
            all_X = all_X[np.random.permutation(all_X.shape[0]), :]
            X1_rand = all_X[: X1.shape[0], :]
            X2_rand = all_X[X1.shape[0] :, :]

        # compute cluster statistics under random permutation
        rand_pos_sizes, rand_neg_sizes, _, _, _ = compute_clustersizes(
            X1=X1_rand,
            X2=X2_rand,
            mask=mask,
            paired=paired,
            p_val=p_val,
            conn=conn,
        )
        max_rand = np.max(np.hstack((rand_pos_sizes, rand_neg_sizes)))

        # count cluster p values
        c_pos_cl += max_rand > pos_sizes
        c_neg_cl += max_rand > neg_sizes

    # compute cluster p values
    p_pos = c_pos_cl / n_permutations
    p_neg = c_neg_cl / n_permutations

    # remove clusters that do not pass threshold

    # !Use np.where instead of a loop: In the section where the function
    # !removes clusters that do not pass the threshold, the function can use
    # !np.where instead of a loop to set the values in the cl_p_vals array.

    # Find clusters: 0 is no cluster
    if tail == "both":
        for i, cl in enumerate(np.unique(ar=pos_labels)[1:]):
            if p_pos[i] < cl_p_val / 2:
                cl_p_vals[pos_labels == cl] = p_pos[i]
            else:
                pos_labels[pos_labels == cl] = 0
        for i, cl in enumerate(np.unique(ar=neg_labels)[1:]):
            if p_neg[i] < cl_p_val / 2:
                cl_p_vals[neg_labels == cl] = p_neg[i]
            else:
                neg_labels[neg_labels == cl] = 0
    elif tail == "right":
        for i, cl in enumerate(np.unique(ar=pos_labels)[1:]):
            if p_pos[i] < cl_p_val:
                cl_p_vals[pos_labels == cl] = p_pos[i]
            else:
                pos_labels[pos_labels == cl] = 0
    elif tail == "left":
        # 0 is not a cluster
        for i, cl in enumerate(np.unique(ar=neg_labels)[1:]):
            if p_neg[i] < cl_p_val:
                cl_p_vals[neg_labels == cl] = p_neg[i]
            else:
                neg_labels[neg_labels == cl] = 0
    else:
        raise ValueError(
            f"tail must be one of 'both', 'right' or 'left', not {tail}"
        )

    return cl_p_vals


def compute_clustersizes(
    X1: np.ndarray,
    X2: np.ndarray,
    mask: np.ndarray,
    paired: bool,
    p_val: float,
    conn: Optional[np.ndarray] = None,
):
    """Compute the actual p-values for each feature based on the data and mask.

    Args:
        X1 (np.ndarray):
            The first group of data, of shape (n_samples, n_features).
        X2 (np.ndarray):
            The second group of data, of shape (n_samples, n_features).
        mask (np.ndarray):
            A boolean mask of shape (n_features,) indicating which features to
            include in the analysis.
        paired (bool):
            Whether to use a paired t-test or an independent t-test.
        p_val (float):
            The p-value threshold to use for cluster inference.

    Returns:
        A tuple of two arrays: the t-values and p-values for each feature
        based on theactual data.

    # ! NOTE:
    At the moment only supports two tailed tests
    At the moment does not support connectivity
    Function should be split up into smaller functions
    """

    # STEP 1: determine 'actual' p value
    # apply the mask to restrict the data
    X1_mask = X1[:, mask]
    X2_mask = X2[:, mask]

    p_vals = np.ones(shape=mask.shape)
    t_vals = np.zeros(shape=mask.shape)

    if paired:
        t_vals[mask], p_vals[mask] = ttest_rel(a=X1_mask, b=X2_mask)
    else:
        t_vals[mask], p_vals[mask] = ttest_ind(a=X1_mask, b=X2_mask)

    # initialize clusters and use mask to restrict relevant info
    significant_clusters = np.mean(a=X1, axis=0) - np.mean(a=X2, axis=0)
    significant_clusters[~mask] = 0
    p_vals[~mask] = 1

    # STEP 2: apply threshold and determine positive and negative clusters
    cl_mask = p_vals < p_val
    positive_clusters = np.zeros(shape=cl_mask.shape)
    negative_clusters = np.zeros(shape=cl_mask.shape)
    positive_clusters[significant_clusters > 0] = cl_mask[
        significant_clusters > 0
    ]
    negative_clusters[significant_clusters < 0] = cl_mask[
        significant_clusters < 0
    ]

    # STEP 3: label clusters
    if conn is None:
        n_positive, positive_labels = cv2.connectedComponents(np.uint8(positive_clusters))  # type: ignore
        n_negative, negative_labels = cv2.connectedComponents(np.uint8(negative_clusters))  # type: ignore
        # hack to control for onedimensional data (CHECK whether correct)
        positive_labels = np.squeeze(a=positive_labels)
        negative_labels = np.squeeze(a=negative_labels)
    else:
        raise NotImplementedError("Function does not yet support connectivity")

    # STEP 4: compute the sum of t stats in each cluster (pos and neg)
    positive_sizes, negative_sizes = np.zeros(shape=n_positive - 1), np.zeros(
        shape=n_negative - 1
    )
    for i, label in enumerate(np.unique(ar=positive_labels)[1:]):
        positive_sizes[i] = np.sum(a=t_vals[positive_labels == label])
    if sum(positive_sizes) == 0:
        positive_sizes = 0

    for i, label in enumerate(np.unique(ar=negative_labels)[1:]):
        negative_sizes[i] = abs(np.sum(a=t_vals[negative_labels == label]))
    if sum(negative_sizes) == 0:
        negative_sizes = 0

    return (
        positive_sizes,
        negative_sizes,
        positive_labels,
        negative_labels,
        p_vals,
    )


def cluster_plot(
    ax: plt.Axes,
    X1: np.ndarray,
    X2: np.ndarray,
    times: np.ndarray,
    line_y: float,
    p_val: float = 0.05,
    cl_p_val: float = 0.05,
    color: str = "black",
    ls: str = "-",
    linewidth: float = 2,
    verbose: bool = True,
):
    """
    Plots significant clusters of time series data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot the data on.
    X1 : array_like, shape (n_samples, n_features)
        The data for group 1, where n_samples is the number of samples
        and n_features is the number of features.
    X2 : array_like, shape (n_samples, n_features)
        The data for group 2, where n_samples is the number of samples
        and n_features is the number of features.
    times : array_like, shape (n_times,)
        The time points for the data.
    line_y : float
        The y-axis value for the horizontal line that marks the
        location of significant clusters.
    p_val : float, optional
        The p-value threshold for the t-test. Only clusters with
        a p-value below this threshold will be considered significant.
        Default is 0.05.
    cl_p_val : float, optional
        The cluster-level p-value threshold. Only clusters with a
        cluster-level p-value below this threshold will be plotted.
        Default is 0.05.
    color : str, optional
        The color of the plotted line. Default is "black".
    ls : str, optional
        The line style of the plotted line. Default is "-".
    linewidth : float, optional
        The width of the plotted line. Default is 2.
    verbose : bool, optional
        Whether to print information about the cluster analysis.
        Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function uses the `clusterbased_permutation` function to perform
    a cluster-based permutation test on the data and identify significant
    clusters. The significant clusters are then plotted on the specified axis.
    """
    sig_cl = clusterbased_permutation(
        X1=X1, X2=X2, p_val=p_val, cl_p_val=cl_p_val, verbose=verbose
    )
    mask = np.where(sig_cl < 1)[0]
    sig_cl = np.split(
        ary=mask, indices_or_sections=np.where(np.diff(mask) != 1)[0] + 1
    )
    for cl in sig_cl:
        ax.plot(
            times[cl],
            np.ones(cl.size) * line_y,
            color=color,
            ls=ls,
            linewidth=linewidth,
        )
