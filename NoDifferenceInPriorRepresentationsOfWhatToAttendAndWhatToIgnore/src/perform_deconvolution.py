import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nideconv
import pandas as pd
import seaborn as sns
from config import PlottingParams, ProjectParams
from tqdm import tqdm

from .utils import (
    check_dataset_existence,
    load_object,
    store_object,
    style_plot_periods,
    style_plot_timecourse,
    timer,
    within_ci,
)

# Switch over to non-interactive backend to prevent crashes when generating a
# lot of plots
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@dataclass
class Deconvolution:
    """
    A class for performing deconvolution on fMRI data.

    Parameters
    ----------
    project_params : ProjectParams
        An object containing project parameters.
    plotting_params : PlottingParams
        An object containing plotting parameters.
    bold : pd.DataFrame
        A pandas DataFrame containing fMRI bold signal data.
    events : pd.DataFrame
        A pandas DataFrame containing event information.
    confounds : pd.DataFrame
        A pandas DataFrame containing confound information.
    mask_roi : str
        A string specifying the name of the mask ROI.
    measurement : str, optional
        A string specifying the measurement, by default "t-values".
    """

    project_params: ProjectParams
    plotting_params: PlottingParams
    bold: pd.DataFrame
    events: pd.DataFrame
    confounds: pd.DataFrame
    mask_roi: str
    measurement: str = "t-values"

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def clean_events_file(self) -> None:
        """
        Cleans the events DataFrame by dropping unnecessary columns and
        renaming the trial_type_extended column to event_type.
        """
        logger.info("Cleaning events file...")

        self.events = self.events.drop(
            ["trial_type", "duration", "condition", "condition_nonTTemplate"],
            axis=1,
        )
        self.events = self.events.rename(
            columns={"trial_type_extended": "event_type"}
        )

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def create_model(self, t_r: float) -> None:
        """
        Creates a GroupResponseFitter model for fMRI data analysis.

        Parameters
        ----------
        t_r : float
            The repetition time of the fMRI data.
        """
        logger.info(f"Creating model with a TR of '{t_r}'...")

        self.model = nideconv.GroupResponseFitter(
            timeseries=self.bold,
            onsets=self.events,
            input_sample_rate=1 / t_r,
            confounds=self.confounds,
            concatenate_runs=False,
        )

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def add_events_to_model(
        self,
        interval: Tuple[float, float],
        basis_set: str = "fourier",
        n_regressors: int = 7,
        show_warnings: bool = False,
    ) -> None:
        """
        Adds events to a previously created model.

        Parameters
        ----------
        interval : Tuple[float, float]
            The interval (in seconds) on which to model the events.
        basis_set : str, optional
            The basis set to use for the model, by default "fourier".
        n_regressors : int, optional
            The number of regressors to use in the model, by default 7.
        show_warnings : bool, optional
            Whether to show warnings when adding events, by default False.
        """
        logger.info(
            f"Adding events to model with '{basis_set}' basis set and"
            f" '{n_regressors}' regressors..."
        )

        for key in tqdm(
            self.events["event_type"].unique(),
            desc=(
                f"Adding events to model with {basis_set} basis set and"
                f" {n_regressors} regressors"
            ),
        ):
            self.model.add_event(
                key,
                interval=interval,
                basis_set=basis_set,
                n_regressors=n_regressors,
                show_warnings=show_warnings,
            )

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def fit_model(
        self,
        solution_type: str = "ols",
        alphas: Tuple[int, ...] = (0.1, 1.0, 10),  # type: ignore
    ) -> None:
        """
        Fits the model to the data.

        Parameters
        ----------
        solution_type : str, optional
            The type of solver to use, by default 'ols'.
        alphas : Tuple[int, ...], optional
            The alpha values to use for the solver, by default (0.1, 1.0, 10).
        """
        logger.info(f"Fitting model with '{solution_type.upper()}'...")

        self.model.fit(type=solution_type, alphas=alphas)

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def get_timecourses_from_model(
        self, measurement: str, oversample: int = 1
    ) -> None:
        """
        Get timecourses as a pandas DataFrame from a fitted
        GroupResponseFitter object.

        Parameters
        ----------
        measurement : str
            The type of measurement to retrieve the timecourses as.
            Can be 't-values' or 'psc'.
        oversample : int, optional
            The factor by which to oversample the timecourses, by default 1.
        """
        logger.info(
            f"Getting timecourses as '{measurement}' (oversampled at"
            f" '{oversample}')..."
        )

        if measurement == "t-values":
            self.timecourses = self.model.get_t_value_timecourses(
                oversample=oversample
            )
        elif measurement == "psc":
            self.timecourses = self.model.get_timecourses(
                oversample=oversample
            )
        else:
            logger.error(
                f"'{measurement}' is an unknown keyword for timecourse"
                " measurement"
            )
            raise KeyError(
                f"'{measurement}' is an unknown keyword for timecourse"
                " measurement"
            )

    def clean_timecourses(self, remove_nans: bool = True) -> None:
        """
        Extracts factors from the given timecourses and adds them to the
        DataFrame.

        Parameters
        ----------
        factors : tuple of str
            A tuple of strings specifying the factors to extract.
        """
        logger.info("Cleaning dataset...")
        # Remove the pesky multiindex and covariate column
        self.timecourses = self.timecourses.reset_index().drop(
            columns="covariate", axis=1, level=0
        )
        # Split event type into different condition columns
        self.timecourses[
            ["condition", "category", "exemplar", "trial"]
        ] = self.timecourses["event type"].str.split("-", expand=True)
        self.timecourses = self.timecourses.drop("event type", axis=1, level=0)

        # Re-build dataframe
        self.timecourses = self.timecourses.droplevel(level=1, axis=1)
        self.timecourses = self.timecourses.groupby(
            by=[
                "subject",
                "run",
                "condition",
                "category",
                "exemplar",
                "time",
            ],
            level=None,
        ).mean()

        # Remove voxels (columns) with NaNs
        if remove_nans:
            mask = self.timecourses.isna().any(axis=0)
            self.timecourses = self.timecourses.loc[:, ~mask]

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def rename_conditions(self, rename_conditions: dict) -> None:
        self.timecourses = self.timecourses.rename(
            index=lambda x: rename_conditions.get(x, x), level="condition"
        )

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def generate_plots(
        self,
        plot_data: pd.DataFrame,
        path_save: Path,
        which_plots: str = "both",
        measurement: str = "t-values",
    ) -> None:
        """
        Generate timecourse plots based on the deconvolved data.

        Parameters
        ----------
        plot_data : pd.DataFrame
            The input data to be plotted.
        path_save : Path
            The file path where the generated plots will be saved.
        which_plots : str, optional
            Determines which type of plot to create. Default is "both", which
            creates both single subject and averaged subject plots. Valid
            values are "single", "all_subs", and "both".
        h_line_reference : float, optional
            The reference value for the horizontal line in the plot. Default is
            0.0.
        y_range_avg : tuple, optional
            The y-axis range for the averaged subject plot. Default is (0,1).
        measurement : str, optional
            The name of the measurement data to plot. Default is "t-values".

        Raises
        ------
        ValueError
            If `which_plots` is not one of "single", "all_subs", or "both".
        """
        logger.info(
            f"Generating '{which_plots}' plots from deconvolved data..."
        )

        if which_plots not in ("both", "single", "all_subs"):
            raise ValueError(
                f"'{which_plots}' is an unknown key value for which_plots"
            )

        # Plot dataset for each subject
        if which_plots in ("each subject", "both"):
            self.create_timecourse_plot_each_sub(
                plot_data=plot_data,
                path_save=path_save,
                measurement=measurement,
            )
        # Plot dataset averaged across subjects
        if which_plots in ("averaged subjects", "both"):
            self.create_timecourse_plot_avg_subs(
                plot_data=plot_data,
                path_save=path_save,
                measurement=measurement,
            )
            self.create_periods_plot_avg_subs(
                plot_data=plot_data,
                path_save=path_save,
                measurement=measurement,
            )

    def create_timecourse_plot_each_sub(
        self,
        plot_data,
        path_save: Path,
        h_line_reference: float = 0.0,
        measurement: str = "t-values",
    ) -> None:
        """
        Create a plot of the time course data for each subject.

        Parameters
        ----------
        plot_data : pd.DataFrame
            A pandas DataFrame containing the timecourses data for all
            subjects.
        path_save : Path
            The path to save the generated plot. The figure will be saved with
            a filename in the format
            `timecourse_{measurement}_sub-{subject}.pdf`. The folder structure
            for the saved figure will be created as `{path_save}/subjects` and
            the data used to generate the figure will be saved in
            `{path_save}/data/timecourse_{measurement}_each_sub.csv`.
        h_line_reference : float, optional
            The y-axis value for a horizontal reference line, by default 0.0.
        y_range : Optional[Tuple[float, float]], optional
            A tuple of the y-axis limits for the plot, by default None.
        measurement : str, optional
            The name of the measurement data to plot, by default "t-values".
        """
        # Get mean of each subject for each condition across time
        data = (
            plot_data.groupby(["subject", "condition", "time"])
            .mean()
            .mean(axis=1)
            .rename(measurement)
        )

        # Create path folders for figure and figure generation
        if path_save:
            path_save = path_save / "subjects"
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)
            # Save data for figure generation
            data.to_csv(f"{path_data}/timecourse_{measurement}_each_sub.csv")

        # Plot each condition of each subject against time in secs
        for subject in data.index.get_level_values("subject").unique():
            _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)

            for condition in data.index.get_level_values("condition").unique():
                ax.plot(
                    data.loc[subject, condition].index,
                    data.loc[subject, condition],
                    label=condition,
                )

            # Type-specific styling
            ax.set_ylabel(
                measurement,
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_ylim(bottom=min(data), top=max(data))
            ax.set_xlabel(
                "time (secs)",
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_title(
                f"Sub-{subject} deconvolved BOLD on"
                f" {self.mask_roi} ({self.project_params.MASK_TYPE.value})",
                pad=self.plotting_params.LABEL_PADDING.value * 2,
            )
            ax.legend(loc="upper right")
            # Generic styling
            ax = style_plot_timecourse(
                project_params=self.project_params,
                plotting_params=self.plotting_params,
                timecourses=data,
                ax=ax,
                h_line_reference=h_line_reference,
            )

            # Save figure
            if path_save:
                plt.savefig(
                    f"{path_save}/timecourse_{measurement}_sub-{subject}.pdf",
                    bbox_inches="tight",
                )
                plt.close()

    def create_timecourse_plot_avg_subs(
        self,
        plot_data,
        path_save: Path,
        h_line_reference: float = 0.0,
        measurement: str = "t-values",
    ) -> None:
        """
        Create a plot of the group-average timecourses for each condition
        across all subjects.

        Parameters
        ----------
        plot_data : pd.DataFrame
            A pandas DataFrame containing the timecourses data for all
            subjects.
        path_save : Path, optional
            The path to save the plot to as a PDF file, by default None.
        h_line_reference : float, optional
            The horizontal reference line to plot, by default 0.0.
        y_range : tuple of floats, optional
            The minimum and maximum y-axis values to plot, by default (0, 1).
        measurement : str, optional
            The type of measurement to plot, by default "t-values".

        Returns
        -------
        None
            The function saves a plot of the group-average timecourses and
            the corresponding data for figure generation as CSV file.

        Raises
        ------
        ValueError
            If the 'measurement' parameter is not valid.

        Notes
        -----
        The function creates a plot with the group-average timecourses for each
        condition across all subjects, using a pandas DataFrame containing the
        timecourses data. It then saves the plot as a PDF file and the data for
        figure generation as CSV file.

        Examples
        --------
        >>> plot_data = pd.DataFrame(...)
        >>> path_save = Path("figures/timecourses")
        >>> create_timecourse_plot_avg_subs(plot_data, path_save)
        """
        # Create path folders for figure and figure generation
        if path_save:
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)

        subjects = plot_data.index.get_level_values("subject").unique()
        # Get mean across all subjects for each condition
        data = plot_data.mean(axis=1).reset_index()
        data = data.pivot_table(
            values=0, index=["condition", "time"], columns="subject"
        )
        data_errorbar = data.sem(axis=1)
        data = data.mean(axis=1)

        # Plot average of each condition of all subjects against time in secs
        _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)
        for condition in data.index.get_level_values("condition").unique():
            # Data, separate line for each condition
            ax.plot(
                data.loc[condition].index,
                data.loc[condition],
                label=condition,
            )
            # Error bars as shaded areas
            ax.fill_between(
                x=data[condition].index,
                y1=data[condition] + data_errorbar[condition],
                y2=data[condition] - data_errorbar[condition],
                alpha=self.plotting_params.ERROR_ALPHA.value,
            )

        # Type-specific styling
        ax.set_ylabel(
            measurement,
            labelpad=self.plotting_params.LABEL_PADDING.value,
        )
        ax.set_ylim(
            bottom=min(data - (2 * data_errorbar)),
            top=max(data + (2 * data_errorbar)),
        )
        ax.set_xlabel(
            "time (secs)",
            labelpad=self.plotting_params.LABEL_PADDING.value,
        )
        ax.set_title(
            f"Group-average (n={ len(subjects) }) deconvolved BOLD on"
            f" {self.mask_roi} ({self.project_params.MASK_TYPE.value})",
            pad=self.plotting_params.LABEL_PADDING.value * 2,
        )
        # Add legend with data labels and error patch label
        patch = mpatches.Patch(
            color="grey", alpha=self.plotting_params.ERROR_ALPHA.value
        )
        ax.legend(
            handles=ax.get_legend_handles_labels()[0] + [patch],
            labels=ax.get_legend_handles_labels()[1] + ["SEM"],
            loc="upper right",
        )
        # Generic styling
        ax = style_plot_timecourse(
            project_params=self.project_params,
            plotting_params=self.plotting_params,
            timecourses=data,
            ax=ax,
            h_line_reference=h_line_reference,
        )

        # Save figure and data for figure generation
        if path_save:
            plt.savefig(
                f"{path_save}/timecourse_{measurement}_avg_subs.pdf",
                bbox_inches="tight",
            )
            plt.close()
            data.to_csv(f"{path_data}/timecourse_{measurement}_avg_subs.csv")

    def create_periods_plot_avg_subs(
        self,
        plot_data,
        path_save: Path,
        statistic_error: str = "sem",
        h_line_reference: float = 0.0,
        measurement: str = "t-values",
    ) -> None:
        """
        Creates a plot of group-average deconvolved BOLD data across different
        conditions in different time periods for each subject. The plot is
        saved in a specified path and the data for the plot is saved in a
        separate path.

        Args:
        plot_data (pd.DataFrame):
            DataFrame containing the data to plot
        path_save (Path):
            Path to save the plot and data
        statistic_error (str, optional):
            Statistic error type for error bars. Can be "sem" for standard
            error of the mean or "ci" for 95% confidence interval.
        Defaults to "sem".
        h_line_reference (float, optional):
            Horizontal line to plot on the plot. Defaults to 0.0.
        y_range (Tuple[float, float], optional):
            Range of y-axis. Defaults to (0, 1).
        measurement (str, optional):
            Name of the measurement to plot. Defaults to "t-values".
        """
        # Create path folders for figure and figure generation
        if path_save:
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)

        subjects = plot_data.index.get_level_values("subject").unique()
        for idx, troi in enumerate(self.project_params.TROIS.value):
            _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)
            # Data
            troi_data = pd.DataFrame(
                plot_data.loc[
                    (
                        (plot_data.index.get_level_values("time") >= troi[0])
                        & (plot_data.index.get_level_values("time") <= troi[1])
                    )
                ]
                .groupby(["subject", "condition"])
                .mean()
                .mean(axis=1),
                columns=[measurement],
            )
            # Data preparations
            troi_name = self.project_params.TROIS_NAME.value[idx]
            if statistic_error == "sem":
                troi_data_errorbar = troi_data.groupby("condition").sem()
            elif statistic_error == "ci":
                troi_data_errorbar = troi_data.pivot_table(
                    values=measurement,
                    index=["subject"],
                    columns="condition",
                )
                troi_data_errorbar = within_ci(troi_data_errorbar.values)
            else:
                raise ValueError(f"Unknown statistic error: {statistic_error}")
            avgs = troi_data.groupby(["condition"]).mean()
            # Plot
            sns.barplot(
                data=avgs.reset_index(),
                x="condition",
                y=measurement,
                ax=ax,
                yerr=troi_data_errorbar.T,
            )
            sns.swarmplot(
                data=troi_data.reset_index(),
                x="condition",
                y=measurement,
                ax=ax,
                linewidth=self.plotting_params.SWARM_LINEWIDTH.value,
                dodge=self.plotting_params.SWARM_DODGE.value,
                alpha=self.plotting_params.SWARM_ALPHA.value,
            )

            # Type-specific styling
            ax.set_ylim(
                bottom=min(troi_data["t-values"]),
                top=max(troi_data["t-values"]),
            )
            ax.set_title(
                f"Group-average (n={len(subjects)}) deconvolved BOLD in"
                f" {troi_name} period on"
                f" {self.mask_roi} ({self.project_params.MASK_TYPE.value})"
            )
            # Generic styling
            ax = style_plot_periods(
                plotting_params=self.plotting_params,
                periods=troi_data,
                ax=ax,
                h_line_reference=h_line_reference,
            )

            # Save figure and data for figure generation
            if path_save:
                plt.savefig(
                    f"{path_save}/periods_{troi_name}_{measurement}_avg_subs.pdf",
                    bbox_inches="tight",
                )
                plt.close()
                troi_data.to_csv(
                    f"{path_data}/periods_{troi_name}_{measurement}_avg_subs.csv"
                )


@timer(
    enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
    logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
)
def perform_deconvolution(
    mask_roi: str,
    bold_mni: pd.DataFrame,
    events: pd.DataFrame,
    confounds: pd.DataFrame,
    store_format="pkl",
    generate_plots: bool = True,
) -> Union[pd.DataFrame, pd.Series, Any]:
    """
    Perform deconvolution on bold_mni using the provided events and confounds
    data. Saves and loads deconvolved data if it is not already stored.
    Returns the deconvolved time courses for the masked brain area.

    Args:
    mask_roi (str):
        Name of the masked brain area.
    bold_mni (pd.DataFrame):
        The bold_mni data.
    events (pd.DataFrame):
        The events data.
    confounds (pd.DataFrame):
        The confounds data.
    store_format (str, optional):
        The file format for storing the deconvolved data. Defaults to "pkl".
    generate_plots (bool, optional):
        Whether to generate plots or not. Defaults to True.

    Returns:
    Union[pd.DataFrame, pd.Series, Any]:
        The deconvolved time courses for the masked brain area.
    """
    logger.info(
        "===> Starting deconvolution process for masked brain area"
        f" '{mask_roi}'..."
    )

    path_data = (
        ProjectParams.PATH_INTERMEDIATE_DATA.value
        / mask_roi
        / ProjectParams.FILENAME_DECONVOLVE.value[0]
    )

    if not check_dataset_existence(
        path_data=path_data,
        file_names=ProjectParams.FILENAME_DECONVOLVE.value[1],
        store_format=store_format,
    ):
        project = Deconvolution(
            project_params=ProjectParams,  # type: ignore
            plotting_params=PlottingParams,  # type: ignore
            bold=bold_mni,
            events=events,
            confounds=confounds,
            mask_roi=mask_roi,
        )
        project.clean_events_file()
        project.create_model(t_r=project.project_params.TR_SECS.value)
        project.add_events_to_model(
            interval=project.project_params.INTERVALS_SECS.value,
        )
        project.fit_model()
        project.get_timecourses_from_model(measurement=project.measurement)
        store_object(
            p_object=project,
            as_name=project.project_params.FILENAME_DECONVOLVE.value[1],
            as_type=store_format,
            path=path_data,
        )
    else:
        project = load_object(
            from_name=ProjectParams.FILENAME_DECONVOLVE.value[1],
            from_type=store_format,
            path=path_data,
        )

    project.clean_timecourses(remove_nans=True)
    project.rename_conditions(
        rename_conditions=ProjectParams.CONDITIONS_RENAME.value
    )

    if generate_plots:
        path_figure = (
            project.project_params.PATH_FIGURES.value
            / project.mask_roi
            / project.project_params.FILENAME_DECONVOLVE.value[0]
        )
        data = project.timecourses.groupby(
            level=("subject", "run", "condition", "category", "time")
        ).mean()
        project.generate_plots(
            plot_data=data,
            which_plots="both",
            path_save=path_figure,
        )

    return project.timecourses
