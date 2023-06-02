import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import PlottingParams, ProjectParams
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifier,
    RidgeClassifierCV,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.svm import SVC
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

logger = logging.getLogger(__name__)


@dataclass
class Decoding:
    project_params: ProjectParams
    plotting_params: PlottingParams
    timecourses: pd.DataFrame
    mask_roi: str
    conditions_to_drop: Union[str, List[str]] = field(default_factory=list)
    conditions_to_pair: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        The __post_init__ method is called after the initialization of an
        object's attributes is complete. This method raises an error if
        both conditions_to_drop and conditions_to_pair are defined, as only
        one of them can be used.

        Raises:
        AssertionError:
            If both conditions_to_drop and conditions_to_pair
            are defined.
        """
        if self.conditions_to_drop and self.conditions_to_pair:
            logger.error(
                "Only conditions_to_drop or conditions_to_pair can be defined."
            )
            raise AssertionError(
                "Only conditions_to_drop or conditions_to_pair can be defined."
            )

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def create_decoder(self, add_decoder: Optional[dict] = None) -> None:
        """
        Creates decoders with parameters for GridSearch and optionally adds
        a new decoder to the classifier_dict.

        Args:
            add_decoder (dict, optional): A new decoder to be added to the
            classifier_dict. Defaults to None.

        Examples:
            To create decoders and add a new decoder:

            >>> classifier = Decoder()
            >>> classifier.create_decoder(
                    add_decoder={"svm": SVC(kernel="rbf")}
                    )

        Notes:
            The classifier_dict is a dictionary containing the names and
            parameters of several classifiers used for decoding. The default
            classifiers included in the classifier_dict are:
            - "logistic": Logistic Regression with L1 penalty
            - "logistic_50": Logistic Regression with L2 penalty, C=50
            - "logistic_l2": Logistic Regression with L2 penalty
            - "logisticCV": Logistic Regression with L1 penalty and CV
            - "SVC": Support Vector Classifier with linear kernel
            - "ridge": Ridge Classifier with alpha=1
            - "ridgeCV": Ridge Classifier with Cross Validation

        """
        logger.info(
            "Creating decoders with parameters for GridSearch and adding"
            f" decoder '{add_decoder}'..."
        )
        self.classifier_dict = {
            "logistic": LogisticRegression(
                C=1.0, penalty="l1", solver="liblinear"
            ),  # "liblinear" 1 vs. rest classification only!
            "logistic_50": LogisticRegression(
                C=50.0, penalty="l2", solver="lbfgs", max_iter=5000
            ),
            "logistic_l2": LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=5000
            ),
            "logisticCV": LogisticRegressionCV(
                Cs=[
                    0.0001,
                    0.0005,
                    0.001,
                    0.01,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1,
                    2.5,
                    5,
                    7.5,
                    10,
                    100,
                    500,
                    1000,
                ],  # type: ignore
                max_iter=5000,
            ),
            "SVC": SVC(kernel="linear"),
            "ridge": RidgeClassifier(alpha=1),
            "ridgeCV": RidgeClassifierCV(
                alphas=[
                    0.0001,
                    0.0005,
                    0.001,
                    0.01,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1,
                    2.5,
                    5,
                    7.5,
                    10,
                    100,
                    500,
                    1000,
                ]
            ),
        }
        # Update the decoder object, if desired
        if add_decoder:
            self.classifier_dict.update(add_decoder)

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def prepare_dataset(
        self,
        groupby: Union[str, List[str]],
        drop_columns: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Preprocesses the dataset by grouping it according to the given
        `groupby` columns and averaging across all runs. Additionally,
        any columns specified in `drop_columns` are removed from the
        resulting dataset.

        Args:
            groupby (Union[str, List[str]]):
                The column(s) to group thedataset by.
            drop_columns (Optional[Union[str, List[str]]]):
                A list of column(s) to drop from the resulting dataset.
                Defaults to None.
        """
        logger.info(f"Preparing dataset by grouping by '{groupby}'...")
        if drop_columns is None:
            drop_columns = []
        # Average across all runs
        self.timecourses = (
            self.timecourses.groupby(groupby)
            .mean()
            .drop(labels=drop_columns, axis=1)
        )

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def drop_conditions(self, conditions: Union[str, List[str]]) -> None:
        logger.info(f"Dropping '{conditions}' from dataset...")
        if isinstance(conditions, str):
            conditions = [conditions]

        self.timecourses = self.timecourses.loc[
            ~self.timecourses.index.get_level_values("condition").isin(
                conditions
            )
        ]

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def paired_conditions(self, conditions: List[str]) -> None:
        logger.info(
            f"Selecting paired conditions '{conditions}' from dataset..."
        )
        if not isinstance(conditions, list) and len(conditions) == 2:
            logger.error(
                "conditions must be a list of length 2, not"
                f" '{type(conditions)}' of length '{len(conditions)}'"
            )
            raise TypeError(
                "conditions must be a list of length 2, not"
                f" '{type(conditions)}' of length '{len(conditions)}'"
            )

        self.timecourses = self.timecourses.loc[
            self.timecourses.index.get_level_values("condition").isin(
                conditions
            )
        ]

    @timer(
        enabled=ProjectParams.TIME_FUNCTIONS_ENABLED.value,
        logging_active=ProjectParams.TIME_FUNCTIONS_LOGGER.value,
    )
    def decode_cat_and_status(
        self,
        decoder: str = "ridgeCV",
        validation: BaseCrossValidator = LeaveOneGroupOut(),
        n_jobs: int = -1,
        *args,
        **kwargs,
    ) -> None:
        """Decode category and status of timecourses for each subject.

        Calculate decoding accuracies using specified decoder and save the
        results in instance variables. Three types of decoding are performed:
        within-category, across-category, and status decoding. The decoding
        accuracies are calculated using cross-validation on the specified
        cross-validator object.

        Args:
            decoder (str):
                Name of the decoder to use. Defaults to 'ridgeCV'.
            validation (BaseCrossValidator):
                Cross-validation object to use. Defaults to LeaveOneGroupOut().
            n_jobs (int):
                Number of CPU cores to use. -1 means use all available CPUs.
            *args:
                Additional arguments passed to the cross_val_score() method.
            **kwargs:
                Additional keyword arguments passed to the cross_val_score()
                method.

        Returns:
            None

        """
        logger.info(
            f"Calculating decoding acurracies using decoder '{decoder}'..."
        )
        tcs = self.timecourses.copy()
        subjects_idx = tcs.index.get_level_values("subject")
        conditions_idx = tcs.index.get_level_values("condition")
        times_idx = tcs.index.get_level_values("time")
        categories_idx = tcs.index.get_level_values("category")
        self.cat_chance_level = 1 / len(categories_idx.unique())
        self.status_chance_level = 1 / 2

        # Category decoding dict can be built on the fly.
        decoded_wcat_results = {}
        # Status & across cat decoding dict=list(dicts) for each condition pair
        decoded_status_results, decoded_acat_results = {}, {}
        decoded_status_results = {
            f"{c1}_{c2}": {}
            for c1, c2 in itertools.combinations(conditions_idx.unique(), 2)
        }
        decoded_acat_results = {
            f"{c1}_{c2}": {}
            for c1, c2 in itertools.combinations(conditions_idx.unique(), 2)
        }
        # Progressbar
        progressbar_subjects = tqdm(
            subjects_idx.unique(), desc="Iterating over subjects"
        )
        for subject in subjects_idx.unique():
            for tr in times_idx.unique():
                processed_pairs = set()
                for idx, condition in enumerate(conditions_idx.unique()):
                    # Within-category decoding
                    data = tcs.loc[
                        (subject, slice(None), condition, slice(None), tr), :
                    ]
                    decoded_wcat_results[
                        (subject, condition, tr)
                    ] = cross_val_score(
                        estimator=self.classifier_dict[decoder],
                        X=data,
                        y=data.index.get_level_values("category"),
                        groups=data.index.get_level_values("run"),
                        cv=validation,
                        n_jobs=n_jobs,
                        *args,
                        **kwargs,
                    )
                    # Across category and status decoding
                    other_conditions = (
                        list(conditions_idx.unique())[:idx]
                        + list(conditions_idx.unique())[idx + 1 :]
                    )
                    condition_combinations = list(
                        itertools.product([condition], other_conditions)
                    )
                    for condition_pair in condition_combinations:
                        if condition_pair[::-1] in processed_pairs:
                            continue
                        processed_pairs.add(condition_pair)
                        condition_pair_str = (
                            f"{condition_pair[0]}_{condition_pair[1]}"
                        )
                        data = tcs.loc[
                            (
                                subject,
                                slice(None),
                                conditions_idx.isin(condition_pair),
                                slice(None),
                                tr,
                            ),
                            :,
                        ]
                        # Across-category decoding: train on one
                        # condition, test another
                        both_conditions = []
                        for idx, _ in enumerate(condition_pair):
                            this_pair = list(condition_pair)
                            if idx > 0:
                                this_pair.reverse()
                            data_train = data.loc[
                                data.index.get_level_values("condition")
                                == this_pair[0]
                            ]
                            data_test = data.loc[
                                data.index.get_level_values("condition")
                                == this_pair[1]
                            ]
                            clf_fitted = self.classifier_dict[decoder].fit(
                                X=data_train,
                                y=data_train.index.get_level_values(
                                    "category"
                                ),
                            )
                            both_conditions.append(
                                clf_fitted.score(
                                    X=data_test,
                                    y=data_test.index.get_level_values(
                                        "category"
                                    ),
                                )
                            )
                        decoded_acat_results[condition_pair_str][
                            (subject, condition_pair_str, tr)
                        ] = both_conditions

                        # Status decoding
                        data = data.groupby(
                            level=("subject", "run", "condition", "time")
                        ).mean()
                        decoded_status_results[condition_pair_str][
                            (subject, condition_pair_str, tr)
                        ] = cross_val_score(
                            estimator=self.classifier_dict[decoder],
                            X=data,
                            y=data.index.get_level_values("condition"),
                            groups=data.index.get_level_values("run"),
                            cv=validation,
                            n_jobs=n_jobs,
                            *args,
                            **kwargs,
                        )

            progressbar_subjects.update()
        progressbar_subjects.close()

        # Within-category decoding: joint dataframe
        index = pd.MultiIndex.from_product(
            [
                subjects_idx.unique(),
                conditions_idx.unique(),
                times_idx.unique(),
            ]
        )
        temp = pd.DataFrame(decoded_wcat_results).T
        # Sort the keys in the index order, before setting the index
        self.decoded_wcat_results = temp.sort_index(
            level=range(len(temp.index.names))
        )
        self.decoded_wcat_results = self.decoded_wcat_results.set_index(index)

        # Across-category and status decoding:
        # joint dataframe for each condition pair
        self.decoded_status_results, self.decoded_acat_results = [], []
        test = pd.DataFrame([])
        if not len(decoded_status_results.keys()) == len(
            decoded_acat_results.keys()
        ):
            logger.warning(
                "decoded_status_results"
                f" ('{len(decoded_status_results.keys())}') and"
                " decoded_acat_results"
                f" ('{len(decoded_acat_results.keys())}') are not the same"
                " length of condition pairs"
            )
        for condition_pair in decoded_status_results.keys():
            index = pd.MultiIndex.from_product(
                [
                    subjects_idx.unique(),
                    pd.Series(
                        condition_pair, name=conditions_idx.unique().name
                    ),
                    times_idx.unique(),
                ]
            )
            acat_temp = pd.DataFrame(decoded_acat_results[condition_pair]).T
            status_temp = pd.DataFrame(
                decoded_status_results[condition_pair]
            ).T
            # No sorting necessary because there is only one condition (pair)
            self.decoded_acat_results.append(acat_temp.set_index(index))
            test = pd.concat([test, acat_temp.set_index(index)])
            self.decoded_status_results.append(status_temp.set_index(index))
        self.decoded_acat_results = pd.concat(self.decoded_acat_results)
        self.decoded_status_results = pd.concat(self.decoded_status_results)

    def generate_plots(
        self,
        plot_data,
        path_save: Path,
        which_plots: str = "both",
        h_line_reference: float = 0.0,
        y_range_avg: Tuple[float, float] = (0, 1),
    ) -> None:
        """
        Generate timecourse and periods plots for the given plot_data and save
        them to the given path_save location.

        Args:
            plot_data:
                A dictionary containing the data to be plotted.
            path_save:
                A Path object specifying the directory where the plots will be
                saved.
            which_plots:
                A string specifying which plots to generate. Options are:
                - 'both' (default): generate both the timecourse and periods
                    plots
                - 'single': generate only the timecourse plot for each subject
                - 'all_subs': generate only the timecourse plot for all
                    subjects
                - 'averaged_subjects': generate the timecourse and periods
                    plots averaged
                across all subjects
            h_line_reference:
                A float specifying the value of the horizontal reference line
                on the plots. Default is 0.0.
            y_range_avg:
                A tuple of floats specifying the y-axis range for the periods
                plots for the averaged subjects. Default is (0, 1).

        Raises:
            ValueError: If which_plots is not one of the valid options.
        """
        logger.info(f"Generating '{which_plots}' plots from decoded data...")

        if which_plots not in ("both", "single", "all_subs"):
            raise ValueError(
                f"'{which_plots}' is an unknown key value for which_plots"
            )

        # Plot dataset for each subject
        if which_plots in ("each subject", "both"):
            self.create_timecourse_plot_each_sub(
                plot_data=plot_data,
                path_save=path_save,
                h_line_reference=h_line_reference,
            )
        # Plot dataset averaged across subjects
        if which_plots in ("averaged subjects", "both"):
            self.create_timecourse_plot_avg_subs(
                plot_data=plot_data,
                path_save=path_save,
                h_line_reference=h_line_reference,
                y_range=y_range_avg,
            )
            self.create_periods_plot_avg_subs(
                plot_data=plot_data,
                path_save=path_save,
                h_line_reference=h_line_reference,
                y_range=y_range_avg,
            )

    def create_timecourse_plot_each_sub(
        self,
        plot_data,
        path_save: Path,
        h_line_reference: float = 0.0,
        y_range: Tuple[float, float] = (0, 1),
    ) -> None:
        """
        Create a timecourse plot for each subject based on the provided plot
        data.

        Args:
            plot_data (pd.DataFrame):
                Data for generating the plot.
            path_save (Path):
                Directory to save the plot.
            h_line_reference (float, optional):
                Horizontal line reference for plot.
            y_range (Tuple[float, float], optional):
                Y-axis range for plot.
        """
        data = plot_data.mean(axis=1).reset_index()  # avg across runs
        data = data.pivot_table(
            values=0, index=["subject", "time"], columns="condition"
        )

        # Create path folders for figure and figure generation
        if path_save is not None:
            path_save = path_save / "subjects"
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)
            # Save data for figure generation
            data.to_csv(f"{path_data}/timecourse_each_sub.csv")

        for subject, subject_data in data.groupby("subject"):
            _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)
            for condition in subject_data.columns:
                ax.plot(
                    subject_data.index.get_level_values("time").values,
                    subject_data[condition].values,
                    label=condition,
                )

            # Plot type-specific styling
            ax.set_ylabel(
                "classification accuracy",
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_ylim(bottom=y_range[0], top=y_range[1])
            ax.set_xlabel(
                "time (secs)",
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_title(
                f"Sub-{subject} decoding on"
                f" {self.mask_roi} ({self.project_params.MASK_TYPE.value})",
                pad=self.plotting_params.LABEL_PADDING.value * 2,
            )
            ax.legend(loc="upper right")
            # Generic styling
            ax = style_plot_timecourse(
                project_params=self.project_params,
                plotting_params=self.plotting_params,
                timecourses=subject_data,
                ax=ax,
                h_line_reference=h_line_reference,
            )

            if path_save:
                # Save figure
                plt.savefig(
                    f"{path_save}/timecourse_sub-{subject}.pdf",
                    bbox_inches="tight",
                )
                plt.close()

    def create_timecourse_plot_avg_subs(
        self,
        plot_data,
        path_save: Path,
        h_line_reference: float = 0.0,
        y_range: Tuple[float, float] = (0, 1),
    ) -> None:
        """
        Creates a plot showing the group average (n=number of subjects)
        decoding results for each condition over time.

        Args:
            plot_data (pd.DataFrame):
                decoding results to plot.
            path_save (Path):
                path to save the figure.
            h_line_reference (float):
                horizontal line reference on the plot.
            y_range (Tuple[float, float]):
                y-axis range.
        """
        subjects = plot_data.index.get_level_values("subject").unique()
        # Get mean across all subjects for each condition
        data = plot_data.mean(axis=1).reset_index()  # avg across runs
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
                data.loc[condition].index.get_level_values("time").values,
                data.loc[condition].values,
                label=condition,
            )
            # Error bars as shaded areas
            ax.fill_between(
                x=data[condition].index,
                y1=data[condition] + data_errorbar[condition],
                y2=data[condition] - data_errorbar[condition],
                alpha=self.plotting_params.ERROR_ALPHA.value,
            )

        # Plot type-specific styling
        ax.set_ylabel(
            "classification accuracy",
            labelpad=self.plotting_params.LABEL_PADDING.value,
        )
        ax.set_xlabel(
            "time (secs)",
            labelpad=self.plotting_params.LABEL_PADDING.value,
        )
        ax.set_ylim(bottom=y_range[0], top=y_range[1])
        ax.set_title(
            f"Group-average (n={ len(subjects) }) decoding on"
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

        if path_save:
            # Create path folders for figure and figure generation
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)
            # Save figure and data for figure generation
            plt.savefig(
                f"{path_save}/timecourse_avg_subs.pdf", bbox_inches="tight"
            )
            plt.close()
            data.to_csv(f"{path_data}/timecourse_avg_subs.csv")

    def create_periods_plot_avg_subs(
        self,
        plot_data,
        path_save: Path,
        statistic_error: str = "sem",
        h_line_reference: float = 0.0,
        y_range: Tuple[float, float] = (0, 1),
    ) -> None:
        """
        The create_periods_plot_avg_subs method creates a bar plot of the
        group-average decoding accuracy for each condition in each time period
        of interest (troi) for all subjects. The plot includes error bars
        showing either the standard error of the mean or the 95% confidence
        interval. Additionally, a swarm plot is included to show the individual
        data points. The plot is saved as a PDF file and the data used to
        create it is saved as a CSV file.

        Args:

        plot_data (pd.DataFrame):
            DataFrame containing the classification accuracy for each subject,
            condition, and time point.
        path_save (Path):
            The path where the plot and data files will be saved. If None,
            files will not be saved.
        statistic_error (str, optional):
            The type of error bar to use. Either "sem" for standard error of
            the mean or "ci" for 95% confidence interval. Defaults to "sem".
        h_line_reference (float, optional):
            The horizontal line that will be plotted across the entire graph.
            Defaults to 0.0.
        y_range (Tuple[float, float], optional):
            The minimum and maximum y-axis values of the plot. Defaults
            to (0, 1).

        Raises:
        ValueError:
            If statistic_error is not "sem" or "ci".
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
                columns=["classification accuracy"],
            )
            # Data preparations
            troi_name = self.project_params.TROIS_NAME.value[idx]
            if statistic_error == "sem":
                troi_data_errorbar = troi_data.groupby("condition").sem()
            elif statistic_error == "ci":
                troi_data_errorbar = troi_data.pivot_table(
                    values="classification accuracy",
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
                y="classification accuracy",
                ax=ax,
                yerr=troi_data_errorbar.T,
            )
            sns.swarmplot(
                data=troi_data.reset_index(),
                x="condition",
                y="classification accuracy",
                ax=ax,
                linewidth=self.plotting_params.SWARM_LINEWIDTH.value,
                dodge=self.plotting_params.SWARM_DODGE.value,
                alpha=self.plotting_params.SWARM_ALPHA.value,
            )

            # Plot type-specific styling
            ax.set_ylim(bottom=y_range[0], top=y_range[1])
            ax.set_title(
                f"Group-average (n={len(subjects)}) decoding in"
                f" {troi_name} period on"
                f" {self.mask_roi} ({self.project_params.MASK_TYPE.value})"
            )
            ax.set_ylabel(
                ax.get_ylabel(),
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            # Generic styling
            ax = style_plot_periods(
                plotting_params=self.plotting_params,
                periods=troi_data,
                ax=ax,
                h_line_reference=h_line_reference,
            )

            if path_save:
                path_save.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    f"{path_save}/periods_{troi_name}_avg_subs.pdf",
                    bbox_inches="tight",
                )
                plt.close()
                troi_data.to_csv(
                    f"{path_data}/periods_{troi_name}_avg_subs.csv"
                )


def perform_decoding(
    mask_roi: str,
    timecourses: pd.DataFrame,
    conditions_to_pair: List[str] = [],
    conditions_to_drop: Union[str, List[str]] = [],
    store_format="pkl",
    generate_plots: bool = True,
):
    """
    Perform decoding analysis on the given masked brain area.

    Args:
        mask_roi (str):
            Name of the masked brain area to analyze.
        timecourses (pd.DataFrame):
            DataFrame containing the timecourses foreach voxel.
        conditions_to_pair (List[str], optional):
            List of condition pairs to combine. Defaults to [].
        conditions_to_drop (Union[str, List[str]], optional):
            List of conditions to drop. Defaults to [].
        store_format (str, optional):
            Format to use for storing intermediate results. Defaults to "pkl".
        generate_plots (bool, optional):
            Whether to generate plots. Defaults to True.
    """
    logger.info(
        f"===> Starting decoding for masked brain area '{mask_roi}'..."
    )

    path_data = (
        ProjectParams.PATH_INTERMEDIATE_DATA.value
        / mask_roi
        / ProjectParams.FILENAME_DECODE.value[0]
    )

    if not check_dataset_existence(
        path_data=path_data,
        file_names=ProjectParams.FILENAME_DECODE.value[1],
        store_format=store_format,
    ):
        project = Decoding(
            project_params=ProjectParams,  # type: ignore
            plotting_params=PlottingParams,  # type: ignore
            timecourses=timecourses,
            mask_roi=mask_roi,
            conditions_to_drop=conditions_to_drop,
            conditions_to_pair=conditions_to_pair,
        )
        project.prepare_dataset(
            groupby=["subject", "run", "condition", "category", "time"],
        )
        if conditions_to_drop:
            project.drop_conditions(conditions=conditions_to_drop)
        if conditions_to_pair:
            project.paired_conditions(conditions=conditions_to_pair)
        project.create_decoder()
        project.decode_cat_and_status(
            decoder="ridgeCV", validation=LeaveOneGroupOut(), n_jobs=-1
        )
        store_object(
            p_object=project,
            as_name=project.project_params.FILENAME_DECODE.value[1],
            as_type=store_format,
            path=path_data,
        )
    else:
        project = load_object(
            from_name=ProjectParams.FILENAME_DECODE.value[1],
            from_type=store_format,
            path=path_data,
        )

    if generate_plots:
        path_figure = (
            project.project_params.PATH_FIGURES.value
            / project.mask_roi
            / project.project_params.FILENAME_DECODE.value[0]
        )
        for analysis in ["within-category", "across-category", "status"]:
            if analysis == "within-category":
                data = project.decoded_wcat_results
                h_line_reference = project.cat_chance_level
                y_range_avg = (
                    project.plotting_params.DECODE_YRANGE_AVG_WCAT.value
                )
            elif analysis == "across-category":
                data = project.decoded_acat_results
                h_line_reference = project.cat_chance_level
                y_range_avg = (
                    project.plotting_params.DECODE_YRANGE_AVG_ACAT.value
                )
            elif analysis == "status":
                data = project.decoded_status_results
                h_line_reference = project.status_chance_level
                y_range_avg = (
                    project.plotting_params.DECODE_YRANGE_AVG_STATUS.value
                )
            else:
                raise ValueError(f"Invalid analysis: {analysis}")
            project.generate_plots(
                plot_data=data,
                which_plots="both",
                path_save=path_figure / analysis,
                h_line_reference=h_line_reference,
                y_range_avg=y_range_avg,
            )
