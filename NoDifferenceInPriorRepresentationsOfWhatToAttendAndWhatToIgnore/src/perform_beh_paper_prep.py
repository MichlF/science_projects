import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import PlottingParams, ProjectParams

logger = logging.getLogger(__name__)


@dataclass
class PaperPreparation:
    project_params: ProjectParams
    plotting_params: PlottingParams
    path_data: Path
    conditions_to_drop: Union[str, List[str]] = field(default_factory=list)
    color_palette: Union[None, str] = None  # "husl"

    def beh_beh(
        self,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        y_range=(1500, 1900),
        save_for_JASP: bool = True,
        error: str = "SEM",
    ) -> None:
        data = pd.read_excel(
            self.project_params.PATH_BEH.value / "beh_data.xlsx",
            usecols=[
                "subject_nr",
                "count_blockSeq",
                "cond_template",
                "responseTime",
                "correct",
            ],
        )

        data = PaperPreparation.rename_conditions(
            data=data,
            condition_column="cond_template",
            rename_conditions=ProjectParams.CONDITIONS_RENAME.value,
        )

        data = data[data["correct"] == 1].drop("correct", axis=1)

        data = (
            data.groupby(["subject_nr", "cond_template"])
            .mean()
            .drop("count_blockSeq", axis=1)
        )

        # Error bars
        if error is not None:
            data_error = PaperPreparation.calculate_error(
                data=data,
                groupby=["cond_template"],
                error=error,
            )

        data_plot = (
            data.groupby(["cond_template"], sort=False).mean().reset_index()
        )
        data_plot = pd.concat([data_plot, data_error[error]], axis=1)

        # 2. Plot
        _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE_NARROW.value)
        ax = sns.barplot(
            data=data_plot,
            x="cond_template",
            y="responseTime",
            edgecolor="black",
            linewidth=1.5,
            palette=sns.color_palette(self.color_palette),
        )

        # Plot styling
        ax.set_ylim(bottom=y_range[0], top=y_range[1])
        ax.set_ylabel("Response Time")
        ax.set_xlabel("Cue Status")

        # 3. Plot error bars
        x_coords, _, _ = PaperPreparation.get_xy_coord_indices(
            ax=ax, data=data_plot, column_label="responseTime"
        )
        plt.errorbar(
            x=x_coords,
            y=data_plot["responseTime"],
            yerr=data_plot[error],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
        )
        sns.despine(ax=ax, offset=10, trim=False, bottom=True)

        if path_save:
            path_save = Path(
                f"{path_save}/dropped_{dropped_condition}/beh_beh"
                if dropped_condition is not None
                else f"{path_save}/beh_beh"
            )
            path_save.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{path_save}.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                f"{path_save}.svg",
                bbox_inches="tight",
            )
            plt.close()

            if save_for_JASP:
                jasp_data = data.pivot_table(
                    values="responseTime",
                    index="subject_nr",
                    columns=["cond_template"],
                )
                jasp_data.to_csv(f"{path_save}_df.csv")

    def beh_fmri(
        self,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        y_range=(1500, 1900),
        save_for_JASP: bool = True,
        error: str = "SEM",
    ) -> None:
        # 1. Prepare data
        data = pd.read_excel(
            self.project_params.PATH_BEH.value / "fMRI_data.xlsx",
            usecols=[
                "subject_nr",
                "count_blockSeq",
                "cond_template",
                "responseTime",
                "correct",
            ],
        )

        data = PaperPreparation.rename_conditions(
            data=data,
            condition_column="cond_template",
            rename_conditions=ProjectParams.CONDITIONS_RENAME.value,
        )

        data = data[data["correct"] == 1].drop("correct", axis=1)

        data = (
            data.groupby(["subject_nr", "cond_template"])
            .mean()
            .drop("count_blockSeq", axis=1)
        )

        # Error bars
        if error is not None:
            data_error = PaperPreparation.calculate_error(
                data=data,
                groupby=["cond_template"],
                error=error,
            )

        data_plot = (
            data.groupby(["cond_template"], sort=False).mean().reset_index()
        )
        data_plot = pd.concat([data_plot, data_error[error]], axis=1)

        # 2. Plot
        _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE_NARROW.value)
        ax = sns.barplot(
            data=data_plot,
            x="cond_template",
            y="responseTime",
            edgecolor="black",
            linewidth=1.5,
            palette=sns.color_palette(self.color_palette),
        )

        # Plot styling
        ax.set_ylim(bottom=y_range[0], top=y_range[1])
        ax.set_ylabel("Response Time")
        ax.set_xlabel("Cue Status")

        # 3. Plot error bars
        x_coords, _, _ = PaperPreparation.get_xy_coord_indices(
            ax=ax, data=data_plot, column_label="responseTime"
        )
        plt.errorbar(
            x=x_coords,
            y=data_plot["responseTime"],
            yerr=data_plot[error],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
        )
        sns.despine(ax=ax, offset=10, trim=False, bottom=True)

        if path_save:
            path_save = Path(
                f"{path_save}/dropped_{dropped_condition}/beh_fMRI"
                if dropped_condition is not None
                else f"{path_save}/beh_fMRI"
            )
            path_save.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{path_save}.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                f"{path_save}.svg",
                bbox_inches="tight",
            )
            plt.close()

            if save_for_JASP:
                jasp_data = data.pivot_table(
                    values="responseTime",
                    index="subject_nr",
                    columns=["cond_template"],
                )
                jasp_data.to_csv(f"{path_save}_df.csv")

    @staticmethod
    def rename_conditions(
        data: pd.DataFrame, condition_column: str, rename_conditions: dict
    ) -> pd.DataFrame:
        data[condition_column] = data[condition_column].replace(
            rename_conditions
        )
        return data

    @staticmethod
    def calculate_error(
        data: pd.DataFrame, groupby: List[str], error="SEM"
    ) -> pd.DataFrame:
        if error == "SEM":
            data_error = (
                data.groupby(groupby, sort=False)
                .sem()
                .rename({data.columns[0]: error}, axis=1)
                .reset_index()
            )
        else:
            logger.error(f"'{error}' is an unknown 'error' type.")
            raise ValueError(f"'{error}' is an unknown 'error' type.")

        return data_error

    @staticmethod
    def get_xy_coord_indices(
        ax: plt.Axes,
        data: pd.DataFrame,
        column_label: str = "Classification Accuracy",
    ) -> Tuple[List, List, List]:
        # For error bars, get x,y coordinates
        x_coords, y_coords = [], []
        for patch in ax.patches:
            x, y = patch.get_xy()
            height = patch.get_height()
            x_coords.append(x + 0.5 * patch.get_width())
            y_coords.append(y + height)

        # Look up the indices
        index_list = []
        for f in y_coords:
            mask = data[column_label] == f
            index = mask.idxmax()
            index_list.append(index)

        return x_coords, y_coords, index_list


def perform_beh_paper_prep() -> None:
    logger.info("===> Starting paper preparations for behavioral data")

    path_data = ProjectParams.PATH_BEH.value

    project = PaperPreparation(
        project_params=ProjectParams,  # type: ignore
        plotting_params=PlottingParams,  # type: ignore
        path_data=path_data,
    )
    project.beh_beh(
        path_save=ProjectParams.PATH_PAPER.value / "behavior",
    )

    project.beh_fmri(
        path_save=ProjectParams.PATH_PAPER.value / "behavior",
    )
