import itertools
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import PlottingParams, ProjectParams
from scipy.stats import rankdata

from .utils import (
    check_dataset_existence,
    load_object,
    store_object,
    style_plot_periods,
)

# matplotlib.use("tkagg")
logger = logging.getLogger(__name__)


@dataclass
class RdmModel:
    project_params: ProjectParams
    plotting_params: PlottingParams
    timecourses: pd.DataFrame
    mask_roi: str
    conditions_to_drop: Union[str, List[str]] = field(default_factory=list)
    conditions_to_pair: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.conditions_to_drop and self.conditions_to_pair:
            logger.error(
                "Only conditions_to_drop or conditions_to_pair can be defined."
            )
            raise AssertionError(
                "Only conditions_to_drop or conditions_to_pair can be defined."
            )

    def prepare_dataset(
        self,
        groupby: Union[str, List[str]],
        drop_columns: Optional[Union[str, List[str]]] = None,
    ) -> None:
        logger.info(f"Preparing dataset by grouping by '{groupby}'...")
        if drop_columns is None:
            drop_columns = []
        # Average across all runs
        self.timecourses = self.timecourses.groupby(groupby).mean()

    def drop_conditions(self, conditions: Union[str, List[str]]) -> None:
        logger.info(f"Dropping '{conditions}' from dataset...")
        if isinstance(conditions, str):
            conditions = [conditions]

        self.timecourses = self.timecourses.loc[
            ~self.timecourses.index.get_level_values("condition").isin(
                conditions
            )
        ]

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

    def calulcate_rdm(
        self,
        statistic: Literal["pearson", "kendall", "spearman"] = "spearman",
    ) -> None:
        logger.info(
            f"Calculating correlation matrix using statistic '{statistic}'..."
        )
        case_1, case_2, case_3 = [], [], []
        subjects = self.timecourses.index.get_level_values("subject").unique()
        times = self.timecourses.index.get_level_values("time").unique()

        # Get TR indices
        trois = self.get_indices_from_trois(
            dataset=self.timecourses, trois=self.project_params.TROIS.value
        )
        # Case 1: Correlation across all subjects and time points
        for troi in trois:
            troi_data = self.timecourses.loc[
                (
                    (
                        self.timecourses.index.get_level_values("time")
                        >= troi[0]
                    )
                    & (
                        self.timecourses.index.get_level_values("time")
                        <= troi[1]
                    )
                )
            ]
            case_1.append(
                troi_data.groupby(level=["condition", "category", "exemplar"])
                .mean()
                .T.corr(method=statistic)
            )

        # Subjects loop
        for subject in subjects:
            # Case 2: Correlation for each subject across all time points
            current_subject = self.timecourses.loc[
                self.timecourses.index.get_level_values("subject") == subject
            ]
            # TR loop
            for tr in times:
                # Case 3: Correlation for each subject and time points
                current_subject_tr = current_subject.loc[
                    current_subject.index.get_level_values("time") == tr
                ]
                corr_mat = current_subject_tr.T.corr(method=statistic)
                case_3.append(corr_mat.droplevel(level="subject", axis=1))

            # TROIS loop
            for troi, troi_name in zip(
                trois, self.project_params.TROIS_NAME.value
            ):
                try:
                    current_subject = current_subject.droplevel(
                        level="period", axis=0
                    )
                except KeyError:
                    pass
                current_subject.loc[:, "period"] = troi_name
                current_subject = current_subject.set_index(
                    "period", append=True
                )
                troi_data = current_subject.loc[
                    (
                        (
                            current_subject.index.get_level_values("time")
                            >= troi[0]
                        )
                        & (
                            current_subject.index.get_level_values("time")
                            <= troi[1]
                        )
                    )
                ]
                corr_mat = (
                    troi_data.groupby(
                        level=[
                            "subject",
                            "condition",
                            "category",
                            "exemplar",
                            "period",
                        ]
                    )
                    .mean()
                    .T.corr(method=statistic)
                )
                case_2.append(
                    corr_mat.droplevel(level=["subject", "period"], axis=1)
                )
        # Case 1
        self.case_1 = pd.concat(
            [
                df.assign(period=i)
                for i, df in zip(self.project_params.TROIS_NAME.value, case_1)
            ]
        )
        self.case_1.index = pd.MultiIndex.from_arrays(
            [
                self.case_1.index.get_level_values(level)
                for level in range(self.case_1.index.nlevels)
            ]
            + [self.case_1.pop("period")]
        )
        # Case 2
        self.case_2 = pd.concat(case_2)
        # Case 3
        self.case_3 = pd.concat(
            [df.droplevel(level="time", axis=1) for df in case_3]
        )
        self.statistic = statistic

    def calc_overall_within_across_category_dissimilarity(
        self, trois: List[List[float]]
    ):
        pass

    def cal_within_between_condition_dissimilarity(self):
        logger.info(f"Calculating within-between similarity...")

        # Param defs
        condition_pairs = list(
            itertools.combinations(
                self.case_2.index.get_level_values("condition").unique(), 2
            )
        )
        periods = self.case_2.index.get_level_values("period").unique()
        subjects = self.case_2.index.get_level_values("subject").unique()
        categories = self.case_2.index.get_level_values("category").unique()

        within, between = {}, {}
        for troi in periods:
            for subject in subjects:
                for category in categories:
                    for condition_pair in condition_pairs:
                        cur_within = self.case_2.loc[
                            (
                                subject,
                                condition_pair[0],
                                category,
                                slice(None),
                                troi,
                            ),
                            (condition_pair[0], category, slice(None)),
                        ].values
                        cur_within = (
                            cur_within[
                                ~np.eye(cur_within.shape[0], dtype=bool)
                            ]
                            .reshape(cur_within.shape[0], -1)
                            .T
                        )
                        cur_within = np.mean(cur_within, axis=0)
                        cur_between = self.case_2.loc[
                            (
                                subject,
                                condition_pair[1],
                                category,
                                slice(None),
                                troi,
                            ),
                            (condition_pair[0], category, slice(None)),
                        ].values
                        cur_between = (
                            cur_between[
                                ~np.eye(cur_between.shape[0], dtype=bool)
                            ]
                            .reshape(cur_between.shape[0], -1)
                            .T
                        )
                        cur_between = np.mean(cur_between, axis=0)

                        condition_pair_label = " vs. ".join(
                            str(elem) for elem in condition_pair
                        )
                        within[
                            (troi, subject, condition_pair_label, category)
                        ] = cur_within
                        between[
                            (troi, subject, condition_pair_label, category)
                        ] = cur_between

        within = pd.DataFrame(within).T
        within["content"] = "within"
        within = within.set_index("content", append=True)
        within.index.names = [
            "period",
            "subject",
            "condition",
            "category",
            "content",
        ]
        within = (
            within.mean(axis=1)
            .groupby(["content", "period", "subject", "condition"])
            .mean()
        )
        between = pd.DataFrame(between).T
        between["content"] = "between"
        between = between.set_index("content", append=True)
        between.index.names = [
            "period",
            "subject",
            "condition",
            "category",
            "content",
        ]
        between = (
            between.mean(axis=1)
            .groupby(["content", "period", "subject", "condition"])
            .mean()
        )
        self.corr_within_between = pd.concat([within, between])

    def generate_plots(
        self,
        path_save: Path,
        condition_to_drop: Union[str, None] = None,
        metric: str = "ranked",
        which_plots: str = "both",
    ) -> None:
        logger.info(
            f"Generating '{which_plots}' RDM plots with metric as"
            f" '{metric}'..."
        )
        data_case_1 = self.case_1.copy()
        data_case_2 = self.case_2.copy()
        data_corr_within_between = self.corr_within_between

        # Drop condition and adjust path_save
        if condition_to_drop:
            data_case_1 = data_case_1.loc[
                data_case_1.index.get_level_values("condition")
                != condition_to_drop
            ]
            data_case_1 = data_case_1.loc[
                :,
                data_case_1.columns.get_level_values("condition")
                != condition_to_drop,
            ]  # type: ignore
            data_case_2 = data_case_2.loc[
                data_case_2.index.get_level_values("condition")
                != condition_to_drop
            ]
            data_case_2 = data_case_2.loc[
                :,
                data_case_2.columns.get_level_values("condition")
                != condition_to_drop,
            ]  # type: ignore
            data_corr_within_between = data_corr_within_between.loc[
                ~data_corr_within_between.index.get_level_values(
                    "condition"
                ).str.contains(condition_to_drop)
            ]
            path_save = path_save / f"dropped_{condition_to_drop}"

        # Plot dataset for each subject
        if which_plots in ("each subject", "both"):
            self.create_periods_plot_each_sub(
                data=data_case_2,
                metric=metric,
                path_save=path_save,
            )
        # Plot dataset averaged across subjects
        if which_plots in ("averaged subjects", "both"):
            self.create_periods_plot_avg_subs(
                data=data_case_1,
                metric=metric,
                path_save=path_save,
            )

        self.create_periods_plot_withinbetween_avg_subs(
            dataset=data_corr_within_between,
            path_save=path_save / "within_between",
        )

    def create_periods_plot_each_sub(
        self,
        data: pd.DataFrame,
        path_save: Path,
        metric: str = "unranked",
    ) -> None:
        # Create path folders for figure and figure generation
        if path_save:
            path_save = path_save / "subjects"
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)

        for troi_name in self.project_params.TROIS_NAME.value:
            # Get and save data for figure generation
            troi_data = data.loc[
                data.index.get_level_values("period") == troi_name
            ].droplevel("period")
            troi_data.to_csv(
                f"{path_data}/periods_{troi_name}_{self.statistic}_{metric}_each_sub.csv"
            )

            # Plot dataset for each subject
            for subject in troi_data.index.get_level_values(
                "subject"
            ).unique():
                data_sub = troi_data.loc[
                    troi_data.index.get_level_values("subject") == subject
                ]
                if metric == "ranked":
                    data_sub = RdmModel.rank_dataset(dataset_2d=data_sub)
                _, ax = plt.subplots(
                    figsize=self.plotting_params.FIGSIZE_RSA.value
                )
                sns.heatmap(
                    1 - data_sub.droplevel("subject"),
                    vmin=0,
                    vmax=1,
                    center=0.5,
                    linewidths=0.025,  # type: ignore
                    linecolor="black",
                    cmap="RdBu_r",
                    square=True,
                    cbar_kws={"label": "dissimilarity"},
                    annot=False,
                    xticklabels=1,
                    yticklabels=1,
                )
                ax.set_title(
                    f"Correlation matrix for '{troi_name}' period in subject"
                    f" {subject}",
                    pad=10,
                )
                ax.set_ylabel("Conditions")
                ax.set_xlabel("Conditions")

                # Save figure
                if path_save:
                    plt.savefig(
                        f"{path_save}/periods_{troi_name}_{self.statistic}_{metric}_sub-{subject}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

    def create_periods_plot_avg_subs(
        self,
        data: pd.DataFrame,
        path_save: Path,
        metric: str = "unranked",
    ) -> None:
        # Create path folders for figure and figure generation
        if path_save:
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)

        for troi_name in self.project_params.TROIS_NAME.value:
            _, ax = plt.subplots(
                figsize=self.plotting_params.FIGSIZE_RSA.value
            )
            # Get and save data for figure generation
            troi_data = data.loc[
                data.index.get_level_values("period") == troi_name
            ].droplevel("period")
            troi_data.to_csv(
                f"{path_data}/periods_{troi_name}_{self.statistic}_{metric}_avg_subs.csv"
            )

            # Plot dataset averaged across subjects
            if metric == "ranked":
                troi_data = RdmModel.rank_dataset(dataset_2d=troi_data)
            sns.heatmap(
                1 - troi_data,
                vmin=0,
                vmax=1,
                center=0.5,
                linewidths=0.025,  # type: ignore
                linecolor="black",
                cmap="RdBu_r",
                square=True,
                cbar_kws={"label": "dissimilarity"},
                annot=False,
                xticklabels=1,
                yticklabels=1,
            )
            ax.set_ylabel("Conditions")
            ax.set_xlabel("Conditions")
            ax.set_title(f"Correlation matrix for {troi_name} period", pad=10)
            # Save the figure, if a save path is provided
            if path_save is not None:
                plt.savefig(
                    f"{path_save}/periods_{troi_name}_{self.statistic}_{metric}_avg_subs.pdf",
                    bbox_inches="tight",
                )
                plt.close()

    def create_periods_plot_withinbetween_avg_subs(
        self, dataset: pd.Series, path_save: Path
    ):
        # Create path folders for figure and figure generation
        if path_save:
            path_save.mkdir(parents=True, exist_ok=True)
            path_data = list(path_save.parts)
            path_data[path_data.index("figures")] = "data"
            path_data = Path(*path_data)
            path_data.mkdir(parents=True, exist_ok=True)

        subjects_idx = dataset.index.get_level_values("subject").unique()
        periods_idx = dataset.index.get_level_values("period").unique()
        for period in periods_idx:
            _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)
            plot_data = dataset.loc[
                (dataset.index.get_level_values("period") == period)
            ]
            plot_data.name = "Correlation"
            avgs = plot_data.groupby(["content", "condition"]).mean()
            plot_data_errorbar = plot_data.groupby(
                ["content", "condition"]
            ).sem()
            plot_data_errorbar.name = "SEM"

            sns.barplot(
                data=avgs.reset_index(),
                x="condition",
                y="Correlation",
                hue="content",
                ax=ax,
            )
            sns.swarmplot(
                data=plot_data.reset_index(),
                x="condition",
                y="Correlation",
                hue="content",
                ax=ax,
                linewidth=self.plotting_params.SWARM_LINEWIDTH.value,
                dodge=self.plotting_params.SWARM_DODGE.value,
                alpha=self.plotting_params.SWARM_ALPHA.value,
            )
            # Plot type-specific styling
            ax.set_ylim(
                bottom=math.floor(min(plot_data) * 10**2) / 10**2,
                top=math.ceil(max(plot_data) * 10**2) / 10**2,
            )
            ax.set_title(
                f"Group-average (n={len(subjects_idx)}) between-within"
                f" correlations in {period} period on"
                f" {self.mask_roi} ({self.project_params.MASK_TYPE.value})"
            )
            ax.set_ylabel(
                ax.get_ylabel(),
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.legend(loc="upper right", frameon=False)
            ax = style_plot_periods(
                plotting_params=self.plotting_params,
                periods=plot_data,
                ax=ax,
                h_line_reference=0.0,
            )

            if path_save:
                plt.savefig(
                    f"{path_save}/periods_{period}_avg_subs.pdf",
                    bbox_inches="tight",
                )
                plt.close()
                plot_data.to_csv(f"{path_data}/periods_{period}_avg_subs.csv")

    def get_indices_from_trois(
        self,
        dataset: pd.DataFrame,
        trois: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> List[List[float]]:
        idx_trois = []
        for troi in trois:
            idx_trois.append(
                [
                    RdmModel.find_nearest(
                        dataset.index.get_level_values("time").unique().values,
                        value=troi[0],
                    )[1],
                    RdmModel.find_nearest(
                        dataset.index.get_level_values("time").unique().values,
                        value=troi[1],
                    )[1],
                ]
            )

        return idx_trois

    @staticmethod
    def find_nearest(
        array: np.ndarray, value: Union[float, int]
    ) -> Tuple[int, Union[int, float]]:
        array = np.asarray(array)
        idx = int((np.abs(array - value)).argmin())

        return idx, array[idx]

    @staticmethod
    def rank_dataset(
        dataset_2d: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(dataset_2d, pd.DataFrame):
            dataset_2d = dataset_2d.copy()
            dataset_ranked = rankdata(dataset_2d.values)
        elif isinstance(dataset_2d, np.ndarray):
            dataset_ranked = rankdata(dataset_2d)
        else:
            logger.error(f"Invalid type for dataset_2d: {type(dataset_2d)}")
            raise TypeError(f"Invalid type for dataset_2d: {type(dataset_2d)}")

        dataset_max, dataset_min = dataset_ranked.max(), dataset_ranked.min()
        dataset_rnorm = (dataset_ranked - dataset_min) / (
            dataset_max - dataset_min
        )
        if isinstance(dataset_2d, pd.DataFrame):
            dataset_2d.iloc[:, :] = np.reshape(
                dataset_rnorm,
                (np.shape(dataset_2d)[0], np.shape(dataset_2d)[1]),
            )
            return dataset_2d
        elif isinstance(dataset_2d, np.ndarray):
            return np.reshape(
                dataset_rnorm,
                (np.shape(dataset_2d)[0], np.shape(dataset_2d)[1]),
            )


def perform_rdm(
    mask_roi: str,
    timecourses: pd.DataFrame,
    paired_conditions: List[str] = [],
    drop_conditions: Union[str, List[str]] = [],
    store_format="pkl",
    generate_plots: bool = True,
):
    logger.info(
        "===> Starting representational similarity analysis for masked brain"
        f" area {mask_roi}..."
    )

    path_data = (
        ProjectParams.PATH_INTERMEDIATE_DATA.value
        / mask_roi
        / ProjectParams.FILENAME_RSAED.value[0]
    )

    if not check_dataset_existence(
        path_data=path_data,
        file_names=ProjectParams.FILENAME_RSAED.value[1],
        store_format=store_format,
    ):
        project = RdmModel(
            project_params=ProjectParams,  # type: ignore
            plotting_params=PlottingParams,  # type: ignore
            timecourses=timecourses,
            mask_roi=mask_roi,
            conditions_to_drop=drop_conditions,
            conditions_to_pair=paired_conditions,
        )
        project.prepare_dataset(
            groupby=["subject", "condition", "category", "exemplar", "time"],
            drop_columns="run",
        )
        if drop_conditions:
            project.drop_conditions(conditions=drop_conditions)
        if paired_conditions:
            project.paired_conditions(conditions=paired_conditions)
        project.calulcate_rdm(statistic="spearman")
        project.cal_within_between_condition_dissimilarity()
        store_object(
            p_object=project,
            as_name=project.project_params.FILENAME_RSAED.value[1],
            as_type=store_format,
            path=path_data,
        )
    else:
        project = load_object(
            from_name=ProjectParams.FILENAME_RSAED.value[1],
            from_type=store_format,
            path=path_data,
        )

    if generate_plots:
        path_figure = (
            project.project_params.PATH_FIGURES.value
            / project.mask_roi
            / project.project_params.FILENAME_RSAED.value[0]
        )
        for condition_to_drop in [None, "drop"]:
            project.generate_plots(
                path_save=path_figure,
                condition_to_drop=condition_to_drop,
                metric="ranked",
                which_plots="both",
            )
