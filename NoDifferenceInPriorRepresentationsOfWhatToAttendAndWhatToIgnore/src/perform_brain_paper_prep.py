import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import PlottingParams, ProjectParams
from config_masks import Masks
from scipy.stats import ttest_1samp

from .utils import style_plot_timecourse

matplotlib.use("tkagg")
logger = logging.getLogger(__name__)


@dataclass
class PaperPreparation:
    project_params: ProjectParams
    plotting_params: PlottingParams
    mask_rois: List[str]
    path_data: Path
    conditions_to_drop: Union[str, List[str]] = field(default_factory=list)
    color_palette: Union[None, str] = None  # "husl"

    def join_datasets(
        self,
        mask_rois: List[str],
        which_data: str,
        period: Union[str, None] = None,
        aggregation: str = "avg",
        dropped_condition: Union[str, None] = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame([])

        for mask_name in mask_rois:
            path_data = self.path_data / mask_name

            if which_data in [
                "across-category",
                "status",
                "within-category",
            ]:
                if which_data == "across-category":
                    if aggregation == "timecourse":
                        df_masked = pd.read_csv(
                            f"{path_data}/decoded/{which_data}/subjects/timecourse_each_sub.csv"
                        )
                        index_columns = ["subject", "time"]
                        df_masked = pd.melt(
                            df_masked,
                            id_vars=index_columns,
                            value_vars=df_masked.columns.drop(index_columns),
                            var_name="condition",
                            value_name="Classification Accuracy",
                        )
                        df_masked = df_masked.set_index(
                            ["subject", "time", "condition"]
                        )
                    else:
                        df_masked = pd.read_csv(
                            f"{path_data}/decoded/{which_data}/periods_{period}_avg_subs.csv",
                            index_col=[0, 1],
                        )
                else:
                    if aggregation == "timecourse":
                        df_masked = pd.read_csv(
                            f"{path_data}/{self.project_params.FILENAME_DECODE.value[0]}/{which_data}/subjects/timecourse_each_sub.csv"
                        )
                        index_columns = ["subject", "time"]
                        df_masked = pd.melt(
                            df_masked,
                            id_vars=index_columns,
                            value_vars=df_masked.columns.drop(index_columns),
                            var_name="condition",
                            value_name="Classification Accuracy",
                        )
                        df_masked = df_masked.set_index(
                            ["subject", "time", "condition"]
                        )
                    else:
                        df_masked = pd.read_csv(
                            f"{path_data}/{self.project_params.FILENAME_DECODE.value[0]}/{which_data}/periods_{period}_avg_subs.csv",
                            index_col=[0, 1],
                        )

                new_index = (
                    df_masked.index.get_level_values("condition")
                    .unique()
                    .str.replace("_", " vs. ")
                )
                df_masked.index = df_masked.index.set_levels(
                    new_index, level="condition"
                )

                if dropped_condition is not None:
                    df_masked = df_masked.loc[
                        ~(
                            df_masked.index.get_level_values(
                                "condition"
                            ).str.contains(dropped_condition)
                        )
                    ]

            elif (
                which_data == self.project_params.FILENAME_DECONVOLVE.value[0]
            ):
                if aggregation == "timecourse":
                    df_masked = pd.read_csv(
                        f"{path_data}/{which_data}/subjects/timecourse_t-values_each_sub.csv",
                        index_col=[0, 1, 2],
                    )
                else:
                    df_masked = pd.read_csv(
                        f"{path_data}/{which_data}/periods_{period}_t-values_avg_subs.csv",
                        index_col=[0, 1],
                    )

                if dropped_condition is not None:
                    df_masked = df_masked.loc[
                        ~(
                            df_masked.index.get_level_values(
                                "condition"
                            ).str.contains(dropped_condition)
                        )
                    ]

            elif which_data == self.project_params.FILENAME_RSAED.value[0]:
                if dropped_condition is not None:
                    df_masked = pd.read_csv(
                        f"{path_data}/{which_data}/dropped_{dropped_condition}/periods_{period}_spearman_ranked_avg_subs.csv",
                        header=[0, 1, 2],
                        index_col=[0, 1, 2],
                    )
                else:
                    df_masked = pd.read_csv(
                        f"{path_data}/{which_data}/periods_{period}_spearman_ranked_avg_subs.csv",
                        header=[0, 1, 2],
                        index_col=[0, 1, 2],
                    )

            elif which_data == "within_between":
                if dropped_condition is not None:
                    df_masked = pd.read_csv(
                        f"{path_data}/{self.project_params.FILENAME_RSAED.value[0]}/{which_data}/dropped_{dropped_condition}/periods_{period}_avg_subs.csv",
                        index_col=[0, 1, 2, 3],
                    )
                else:
                    df_masked = pd.read_csv(
                        f"{path_data}/{self.project_params.FILENAME_RSAED.value[0]}/{which_data}/periods_{period}_avg_subs.csv",
                        index_col=[0, 1, 2, 3],
                    )

            else:
                logger.error(f"'{which_data}' is an unknown 'which_data' type")
                raise ValueError(
                    f"'{which_data}' is an unknown 'which_data' type"
                )

            df_masked["Area"] = Masks.RENAME.value[mask_name]
            df = pd.concat([df, df_masked])

        df = df.set_index("Area", append=True)

        return df

    def deconvolved_timecourse(
        self,
        mask_rois: List[str],
        which_data: str,
        chance_level: float,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        error: str = "SEM",
        save_for_JASP: bool = True,
    ) -> None:
        # Prepare save path
        if path_save:
            path_save = Path(
                f"{path_save}/dropped_{dropped_condition}/timecourse_{which_data}"
                if dropped_condition is not None
                else f"{path_save}/timecourse_{which_data}"
            )
            path_save.parent.mkdir(parents=True, exist_ok=True)

        # 1. Prepare data
        data = self.join_datasets(
            mask_rois=mask_rois,
            which_data=which_data,
            aggregation="timecourse",
            dropped_condition=dropped_condition,
        )

        for area in data.index.get_level_values("Area").unique():
            plot_data = data.loc[
                data.index.get_level_values("Area") == area
            ].pivot_table(
                values=data.columns[0],
                index=["condition", "time"],
                columns="subject",
            )

            if error is not None:
                data_errorbar = plot_data.sem(axis=1)

            plot_data = plot_data.mean(axis=1)

            # Plot average of each condition of all subjects against time in s
            _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)
            for condition in data.index.get_level_values("condition").unique():
                # Data, separate line for each condition
                ax.plot(
                    plot_data.loc[condition]
                    .index.get_level_values("time")
                    .values,
                    plot_data.loc[condition].values,
                    label=condition,
                )
                # Error bars as shaded areas
                ax.fill_between(
                    x=plot_data[condition].index,
                    y1=plot_data[condition] + data_errorbar[condition],
                    y2=plot_data[condition] - data_errorbar[condition],
                    alpha=self.plotting_params.ERROR_ALPHA.value,
                )

            # Plot type-specific styling
            ax.set_ylabel(
                data.columns[0],
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_xlabel(
                "Time (secs)",
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_ylim(
                bottom=math.floor(data.min() * 2) / 2,
                top=math.ceil(data.max() * 2) / 2,
            )
            ax.legend(
                loc="upper right",
            )

            # Generic styling
            ax = style_plot_timecourse(
                project_params=self.project_params,
                plotting_params=self.plotting_params,
                timecourses=plot_data,
                ax=ax,
                h_line_reference=chance_level,
            )

            if path_save:
                plt.savefig(
                    f"{path_save}_{area}.pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{path_save}_{area}.svg",
                    bbox_inches="tight",
                )
                plt.close()

        if save_for_JASP:
            jasp_data = data.pivot_table(
                values=data.columns[0],
                index="subject",
                columns=["Area", "condition"],
            )
            jasp_data.columns = jasp_data.columns.map("_".join)
            jasp_data.to_csv(f"{path_save}_df.csv")

    def deconvolved_avg(
        self,
        mask_rois: List[str],
        which_data: str,
        period: str,
        chance_level: float,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        error: str = "SEM",
        save_for_JASP: bool = True,
    ) -> None:
        # 1. Prepare data
        data = self.join_datasets(
            mask_rois=mask_rois,
            which_data=which_data,
            period=period,
            dropped_condition=dropped_condition,
        )
        data = data.sort_values(["subject", "condition"])

        if error is not None:
            data_error = PaperPreparation.calculate_error(
                data=data, groupby=["condition", "Area"], error=error
            )

        p_values = PaperPreparation.get_significance_vs_baseline(
            data=data,
            groupby=["Area", "condition"],
            values="t-values",
            baseline=chance_level,
            rename="p-values",
        )

        # Combine t-values, errors and p-values
        data_plot = (
            data.groupby(["Area", "condition"], sort=False)
            .mean()
            .reset_index()
        )
        data_plot = pd.concat(
            [data_plot, data_error[error], p_values["p-values"]], axis=1
        )

        # 2. Plot data
        _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE_WIDE.value)
        ax = sns.barplot(
            data=data_plot,
            x="Area",
            y="t-values",
            hue="condition",
            edgecolor="black",
            linewidth=1.5,
            palette=sns.color_palette(self.color_palette),
        )
        # Plot styling
        # ax.set_title(f"{which_data.title()} BOLD activity in {period} period")
        ax.set_ylim(
            bottom=math.ceil(
                (data_plot["t-values"].min() - data_plot[error].max())
                * 10**2
            )
            / 10**2,
            top=math.ceil(
                (data_plot["t-values"].max() + data_plot[error].max())
                * 10**2
            )
            / 10**2,
        )
        ax.legend(loc="upper right", frameon=False)
        ax.axhline(
            chance_level,
            color=PlottingParams.HLINE_COLOR.value,
            lw=PlottingParams.LINEWIDTH.value,
            linestyle=PlottingParams.HLINE_LINESTYLE.value,
        )

        # 3. Plot significance markers
        PaperPreparation.plot_significance(
            ax=ax, significance=data_plot["p-values"], mark_ns=True
        )

        # 4. Plot error bars
        x_coords, _, _ = PaperPreparation.get_xy_coord_indices(
            ax=ax, data=data_plot, column_label="t-values"
        )
        plt.errorbar(
            x=x_coords,
            y=data_plot["t-values"],
            yerr=data_plot[error],
            fmt="none",
            ecolor="black",
            elinewidth=1,
        )
        sns.despine(ax=ax, offset=10, trim=False, bottom=True)

        if path_save:
            path_save = Path(
                f"{path_save}/dropped_{dropped_condition}/periods_{period}_{which_data}"
                if dropped_condition is not None
                else f"{path_save}/periods_{period}_{which_data}"
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
                    values="t-values",
                    index="subject",
                    columns=["Area", "condition"],
                )
                jasp_data.columns = jasp_data.columns.map("_".join)
                jasp_data.to_csv(f"{path_save}_df.csv")

    def decoding_avg(
        self,
        mask_rois: List[str],
        which_data: str,
        period: str,
        chance_level: float,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        error: str = "SEM",
        y_range: Tuple[float, float] = (0.25, 0.5),
        save_for_JASP: bool = True,
    ) -> None:
        # 1. Prepare data
        data = self.join_datasets(
            mask_rois=mask_rois,
            which_data=which_data,
            period=period,
            dropped_condition=dropped_condition,
        )
        data = data.sort_values(["subject", "condition"])

        if error is not None:
            data_error = PaperPreparation.calculate_error(
                data=data, groupby=["condition", "Area"], error=error
            )

        p_values = PaperPreparation.get_significance_vs_baseline(
            data=data,
            groupby=["Area", "condition"],
            values="classification accuracy",
            baseline=chance_level,
            rename="p-values",
        )

        data_plot = (
            data.groupby(["Area", "condition"], sort=False)
            .mean()
            .reset_index()
        )
        data_plot = data_plot.rename(columns=lambda x: x.title())
        data_plot = pd.concat(
            [data_plot, data_error[error], p_values["p-values"]], axis=1
        )

        # 2. Plot data
        _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE_WIDE.value)
        ax = sns.barplot(
            data=data_plot,
            x="Area",
            y="Classification Accuracy",
            hue="Condition",
            edgecolor="black",
            linewidth=1.5,
            palette=sns.color_palette(self.color_palette),
        )

        # Plot styling
        # ax.set_title(f"Decoding {which_data} in {period} period")
        ax.set_ylim(bottom=y_range[0], top=y_range[1])
        ax.legend(loc="upper right", frameon=False)
        ax.axhline(
            chance_level,
            color=PlottingParams.HLINE_COLOR.value,
            lw=PlottingParams.LINEWIDTH.value,
            linestyle=PlottingParams.HLINE_LINESTYLE.value,
        )

        # 3. Plot significance markers
        PaperPreparation.plot_significance(
            ax=ax, significance=data_plot["p-values"], mark_ns=True
        )

        # 4. Plot error bars
        x_coords, _, _ = PaperPreparation.get_xy_coord_indices(
            ax=ax, data=data_plot
        )
        plt.errorbar(
            x=x_coords,
            y=data_plot["Classification Accuracy"],
            yerr=data_plot[error],
            fmt="none",
            ecolor="black",
            elinewidth=1,
        )
        sns.despine(ax=ax, offset=10, trim=False, bottom=True)

        if path_save:
            path_save = Path(
                f"{path_save}/dropped_{dropped_condition}/periods_{period}_{which_data}"
                if dropped_condition is not None
                else f"{path_save}/periods_{period}_{which_data}"
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
                    values="classification accuracy",
                    index="subject",
                    columns=["Area", "condition"],
                )
                jasp_data.columns = jasp_data.columns.map("_".join)
                jasp_data.to_csv(f"{path_save}_df.csv")

    def decoding_timecourse(
        self,
        mask_rois: List[str],
        which_data: str,
        chance_level: float,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        error: str = "SEM",
        y_range: Tuple[float, float] = (0, 1),
        save_for_JASP: bool = True,
    ) -> None:
        # Prepare save path
        if path_save:
            path_save = Path(
                f"{path_save}/dropped_{dropped_condition}/timecourse_{which_data}"
                if dropped_condition is not None
                else f"{path_save}/timecourse_{which_data}"
            )
            path_save.parent.mkdir(parents=True, exist_ok=True)

        # 1. Prepare data
        data = self.join_datasets(
            mask_rois=mask_rois,
            which_data=which_data,
            aggregation="timecourse",
            dropped_condition=dropped_condition,
        )

        for area in data.index.get_level_values("Area").unique():
            plot_data = data.loc[
                data.index.get_level_values("Area") == area
            ].pivot_table(
                values=data.columns[0],
                index=["condition", "time"],
                columns="subject",
            )

            if error is not None:
                data_errorbar = plot_data.sem(axis=1)

            plot_data = plot_data.mean(axis=1)

            # Plot average of each condition of all subjects against time in s
            _, ax = plt.subplots(figsize=self.plotting_params.FIGSIZE.value)
            for condition in data.index.get_level_values("condition").unique():
                # Data, separate line for each condition
                ax.plot(
                    plot_data.loc[condition]
                    .index.get_level_values("time")
                    .values,
                    plot_data.loc[condition].values,
                    label=condition,
                )
                # Error bars as shaded areas
                ax.fill_between(
                    x=plot_data[condition].index,
                    y1=plot_data[condition] + data_errorbar[condition],
                    y2=plot_data[condition] - data_errorbar[condition],
                    alpha=self.plotting_params.ERROR_ALPHA.value,
                )

            # Plot type-specific styling
            ax.set_ylabel(
                data.columns[0],
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_xlabel(
                "Time (secs)",
                labelpad=self.plotting_params.LABEL_PADDING.value,
            )
            ax.set_ylim(bottom=y_range[0], top=y_range[1])
            ax.legend(
                loc="upper right",
            )

            # Generic styling
            ax = style_plot_timecourse(
                project_params=self.project_params,
                plotting_params=self.plotting_params,
                timecourses=plot_data,
                ax=ax,
                h_line_reference=chance_level,
            )

            if path_save:
                plt.savefig(
                    f"{path_save}_{area}.pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{path_save}_{area}.svg",
                    bbox_inches="tight",
                )
                plt.close()

        if save_for_JASP:
            jasp_data = data.pivot_table(
                values=data.columns[0],
                index="subject",
                columns=["Area", "condition"],
            )
            jasp_data.columns = jasp_data.columns.map("_".join)
            jasp_data.to_csv(f"{path_save}_df.csv")

    def decoding_cat_vs_status(
        self,
        mask_rois: List[str],
        period: str,
        compare_analyses: List[str],
        compare_conditions: List[str],
        chance_level: List[float],
        path_save: Path,
        error: str = "SEM",
        y_range=(-0.02, 0.12),
        save_for_JASP: bool = True,
    ) -> None:
        # 1. Prepare data
        data_a = self.join_datasets(
            mask_rois=mask_rois,
            which_data=compare_analyses[0],
            period=period,
        )
        data_a = data_a.sort_values(["subject", "condition"])
        data_a = data_a.loc[
            (
                data_a.index.get_level_values("condition").isin(
                    compare_conditions
                )
            )
        ]
        data_b = self.join_datasets(
            mask_rois=mask_rois,
            which_data=compare_analyses[1],
            period=period,
        )
        data_b = data_b.sort_values(["subject", "condition"])
        data_b = data_b.loc[
            data_b.index.get_level_values("condition")
            == f"{sorted(compare_conditions)[0]} vs."
            f" {sorted(compare_conditions)[1]}"
        ].droplevel(level="condition")
        # For within the comparison is an avg between two separate conditions
        # For status the comparison already exists as condition
        data_a = data_a.groupby(["subject", "Area"], sort=False).mean()
        compare_analyses = [
            "category" if rename_string == "within-category" else rename_string
            for rename_string in compare_analyses
        ]
        data_a["Analysis"] = compare_analyses[0]
        data_b["Analysis"] = compare_analyses[1]
        # Calculate error bar
        if error is not None:
            data_a_error = PaperPreparation.calculate_error(
                data=data_a, groupby=["Area", "Analysis"], error=error
            )
            data_b_error = PaperPreparation.calculate_error(
                data=data_b, groupby=["Area", "Analysis"], error=error
            )
        # Average, substract chance level and add error for plotting
        dataset = pd.DataFrame([])
        temp_a = (
            data_a.groupby(["Area", "Analysis"], sort=False)
            .mean()
            .rename(columns=lambda x: x.title())
            - chance_level[0]
        )
        temp_a = pd.merge(temp_a, data_a_error, on="Area")
        dataset = pd.concat([dataset, temp_a])
        temp_b = (
            data_b.groupby(["Area", "Analysis"], sort=False)
            .mean()
            .rename(columns=lambda x: x.title())
            - chance_level[1]
        )
        temp_b = pd.merge(temp_b, data_b_error, on="Area")
        dataset = pd.concat([dataset, temp_b])

        # 2. Plot
        _, ax = plt.subplots(figsize=(7, 7))
        ax = sns.barplot(
            data=dataset.reset_index(),
            x="Area",
            y="Classification Accuracy",
            hue="Analysis",
            edgecolor="black",
            linewidth=1.5,
            palette=sns.color_palette(self.color_palette),
        )
        # Plot styling
        ax.set_ylim(bottom=y_range[0], top=y_range[1])
        ax.legend(loc="upper right", frameon=False)
        ax.axhline(
            0.0,
            color=PlottingParams.HLINE_COLOR.value,
            lw=PlottingParams.LINEWIDTH.value,
            linestyle=PlottingParams.HLINE_LINESTYLE.value,
        )

        # 3. Plot error bars
        x_coords, _, _ = PaperPreparation.get_xy_coord_indices(
            ax=ax, data=dataset
        )
        plt.errorbar(
            x=x_coords,
            y=dataset["Classification Accuracy"],
            yerr=dataset[error],
            fmt="none",
            ecolor="black",
            elinewidth=1,
        )
        sns.despine(ax=ax, offset=10, trim=False, bottom=True)

        # 4. Save plot
        if path_save:
            path_save.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{path_save}/periods_{period}_{compare_analyses[0]}_vs_{compare_analyses[1]}_{compare_conditions[0]}_vs_{compare_conditions[1]}.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                f"{path_save}/periods_{period}_{compare_analyses[0]}_vs_{compare_analyses[1]}_{compare_conditions[0]}_vs_{compare_conditions[1]}.svg",
                bbox_inches="tight",
            )
            plt.close()

            if save_for_JASP:
                data_a["classification accuracy"] = (
                    data_a["classification accuracy"] - chance_level[0]
                )
                data_b["classification accuracy"] = (
                    data_b["classification accuracy"] - chance_level[1]
                )
                jasp_data = pd.concat([data_a, data_b])
                order = list(jasp_data.index.get_level_values("Area").unique())
                jasp_data = jasp_data.pivot_table(
                    values="classification accuracy",
                    index="subject",
                    columns=["Area", "Analysis"],
                )
                jasp_data = jasp_data.loc[:, (pd.IndexSlice[order])]
                jasp_data.columns = jasp_data.columns.map("_".join)
                jasp_data.to_csv(
                    f"{path_save}/periods_{period}_{compare_analyses[0]}_vs_{compare_analyses[1]}_{compare_conditions[0]}_vs_{compare_conditions[1]}_df.csv"
                )

    def corr_behfmri_vs_posvneg_decoding(
        self,
        mask_rois: List[str],
        period: str,
        compare_analyses: List[str],
        compare_conditions: List[str],
        path_save: Path,
        save_for_JASP: bool = True,
    ) -> None:
        # Correlation between fMRI behavior and mask_roi activity for
        # negative vs. positive
        data_a = pd.read_excel(
            self.project_params.PATH_BEH.value / "fMRI_data.xlsx",
            usecols=[
                "subject_nr",
                "count_blockSeq",
                "cond_template",
                "responseTime",
                "correct",
            ],
        )
        data_a = data_a[data_a["correct"] == 1].drop("correct", axis=1)
        data_a = (
            data_a.groupby(["subject_nr", "cond_template"])
            .mean()
            .drop("count_blockSeq", axis=1)
        )
        beh_cond_difference = data_a.loc[
            data_a.index.get_level_values("cond_template") == "negative"
        ].droplevel("cond_template") - data_a.loc[
            data_a.index.get_level_values("cond_template") == "positive"
        ].droplevel(
            "cond_template"
        )

        data_b = self.join_datasets(
            mask_rois=mask_rois,
            which_data=compare_analyses[1],
            period=period,
        )

        for area in data_b.index.get_level_values("Area").unique():
            data_b_area = data_b.loc[
                (
                    data_b.index.get_level_values("condition")
                    == f"{sorted(compare_conditions)[0]} vs."
                    f" {sorted(compare_conditions)[1]}"
                )
                & (data_b.index.get_level_values("Area") == area)
            ].droplevel(level=["condition", "Area"])

            # Correlation
            corr_df = pd.concat(
                [beh_cond_difference.squeeze(), data_b_area.squeeze()],
                axis=1,
                keys=["Response Time (Δ)", f"{area} Decoding Accuracy (Δ)"],
            )
            sns.jointplot(
                x=corr_df[f"{area} Decoding Accuracy (Δ)"],
                y=corr_df["Response Time (Δ)"],
                kind="reg",
            )

            # Save plot
            if path_save:
                path_save.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    f"{path_save}/periods_{period}_{area}_{compare_analyses[0]}_vs_{compare_analyses[1]}_{compare_conditions[0]}_vs_{compare_conditions[1]}.pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{path_save}/periods_{period}_{area}_{compare_analyses[0]}_vs_{compare_analyses[1]}_{compare_conditions[0]}_vs_{compare_conditions[1]}.svg",
                    bbox_inches="tight",
                )
                plt.close()

                if save_for_JASP:
                    corr_df.to_csv(
                        f"{path_save}/periods_{period}_{area}_{compare_analyses[0]}_vs_{compare_analyses[1]}_{compare_conditions[0]}_vs_{compare_conditions[1]}_df.csv"
                    )

    def rsa_withinbetween(
        self,
        mask_rois: List[str],
        which_data: str,
        period: str,
        path_save: Path,
        dropped_condition: Union[str, None] = None,
        error: str = "SEM",
        save_for_JASP: bool = True,
    ) -> None:
        # 1. Prepare data
        data = self.join_datasets(
            mask_rois=mask_rois,
            which_data=which_data,
            period=period,
            dropped_condition=dropped_condition,
        )

        if error is not None:
            data_error = PaperPreparation.calculate_error(
                data=data,
                groupby=["condition", "content", "Area"],
                error=error,
            )

        data_grouped = (
            data.groupby(["Area", "content", "condition"], sort=False)
            .mean()
            .reset_index()
        )
        data_grouped = pd.concat([data_grouped, data_error[error]], axis=1)
        bottom = (
            math.floor(
                (min(data_grouped["Correlation"]) - max(data_grouped[error]))
                * 10**2
            )
            / 10**2
        )
        top = (
            math.ceil(
                (max(data_grouped["Correlation"]) + max(data_grouped[error]))
                * 10**2
            )
            / 10**2
        )

        # 2. Plot data by condition
        for condition in data_grouped["condition"].unique():
            data_plot = data_grouped.loc[
                data_grouped["condition"] == condition
            ]
            _, ax = plt.subplots(
                figsize=self.plotting_params.FIGSIZE_WIDE.value
            )
            ax = sns.barplot(
                data=data_plot,
                x="Area",
                y="Correlation",
                hue="content",
                edgecolor="black",
                linewidth=1.5,
                palette=sns.color_palette(self.color_palette),
                # ci=None,
            )

            # Plot styling
            ax.set_ylim(bottom=bottom, top=top)
            ax.legend(loc="upper right", frameon=False)
            ax.axhline(
                0.0,
                color=PlottingParams.HLINE_COLOR.value,
                lw=PlottingParams.LINEWIDTH.value,
                linestyle=PlottingParams.HLINE_LINESTYLE.value,
            )

            # 3. Plot error bars
            x_coords, _, index_list = PaperPreparation.get_xy_coord_indices(
                ax=ax, data=data_plot, column_label="Correlation"
            )
            plt.errorbar(
                x=x_coords,
                y=data_plot["Correlation"][index_list],
                yerr=data_plot[error][index_list],
                fmt="none",
                ecolor="black",
                elinewidth=1,
            )
            sns.despine(ax=ax, offset=10, trim=False, bottom=True)

            if path_save:
                path_csave = Path(
                    f"{path_save}/dropped_{dropped_condition}/{condition.replace(' vs. ', '_')}_periods_{period}_{which_data}"
                    if dropped_condition is not None
                    else (
                        f"{path_save}/{condition.replace(' vs. ', '_')}_periods_{period}_{which_data}"
                    )
                )
                path_csave.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    f"{path_csave}.pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{path_csave}.svg",
                    bbox_inches="tight",
                )
                plt.close()

        if save_for_JASP:
            path_csave = Path(
                f"{path_save}/dropped_{dropped_condition}/periods_{period}_{which_data}"
                if dropped_condition is not None
                else f"{path_save}/periods_{period}_{which_data}"
            )
            jasp_data = data.pivot_table(
                values="Correlation",
                index="subject",
                columns=["content", "Area", "condition"],
            )
            jasp_data.columns = jasp_data.columns.map("_".join)
            jasp_data.to_csv(f"{path_csave}_df.csv")

    @staticmethod
    def rename_conditions(
        data: pd.DataFrame, condition_column: str, rename_conditions: dict
    ) -> pd.DataFrame:
        data[condition_column] = data[condition_column].replace(
            rename_conditions
        )
        return data

    @staticmethod
    def get_significance_vs_baseline(
        data: pd.DataFrame,
        groupby: List[str],
        values: str,
        baseline: float,
        rename: str = "p-values",
    ) -> pd.DataFrame:
        results = data.groupby(groupby, sort=False).apply(
            lambda x: ttest_1samp(x[values], baseline)
        )
        p_values = results.apply(lambda x: x.pvalue)
        p_values = p_values.reset_index().rename({0: rename}, axis=1)

        return p_values

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
    def plot_significance(
        ax: plt.Axes,
        significance: Union[pd.Series, List[float]],
        offset_percentage: float = 0.08,
        mark_ns: bool = False,
    ) -> None:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        height_offset = offset_percentage * y_range
        for idx_bar, bar in enumerate(ax.containers):
            for idx_rect, rect in enumerate(bar.patches):
                p_val = significance[(len(bar.patches) * idx_bar) + idx_rect]
                if p_val < 0.001:
                    marker = "***"
                elif p_val < 0.01:
                    marker = "**"
                elif p_val < 0.05:
                    marker = "*"
                elif p_val < 0.1:
                    marker = "†"
                elif mark_ns:
                    marker = "ns"
                else:
                    marker = ""
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    ax.get_ylim()[0] + height_offset,
                    marker,
                    ha="center",
                    va="top",
                    fontsize=10,
                )

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


def perform_brain_paper_prep(mask_rois: List[str]) -> None:
    logger.info(
        "===> Starting paper preparations for masked brain areas"
        f" '{mask_rois}'..."
    )

    path_data = ProjectParams.PATH_INTERMEDIATE_DATA.value

    project = PaperPreparation(
        project_params=ProjectParams,  # type: ignore
        plotting_params=PlottingParams,  # type: ignore
        mask_rois=mask_rois,
        path_data=path_data,
    )

    project.deconvolved_timecourse(
        mask_rois=project.mask_rois,
        which_data="deconvolved",
        chance_level=0.0,
        path_save=ProjectParams.PATH_PAPER.value / "deconvolved",
    )

    project.decoding_timecourse(
        mask_rois=project.mask_rois,
        which_data="within-category",
        chance_level=1 / 3,
        y_range=(0.25, 0.65),
        path_save=ProjectParams.PATH_PAPER.value / "within-category",
    )

    project.decoding_timecourse(
        mask_rois=project.mask_rois,
        which_data="across-category",
        chance_level=1 / 3,
        y_range=(0.25, 0.65),
        dropped_condition="drop",
        path_save=ProjectParams.PATH_PAPER.value / "across-category",
    )

    project.decoding_timecourse(
        mask_rois=project.mask_rois,
        which_data="status",
        chance_level=1 / 2,
        y_range=(0.35, 0.7),
        dropped_condition="drop",
        path_save=ProjectParams.PATH_PAPER.value / "status",
    )

    for period in ["cue", "search"]:
        project.rsa_withinbetween(
            mask_rois=project.mask_rois,
            which_data="within_between",
            period=period,
            path_save=ProjectParams.PATH_PAPER.value / "rsa_within_between",
        )

        project.deconvolved_avg(
            mask_rois=project.mask_rois,
            which_data="deconvolved",
            period=period,
            chance_level=0.0,
            path_save=ProjectParams.PATH_PAPER.value / "deconvolved",
        )

        project.decoding_avg(
            mask_rois=project.mask_rois,
            which_data="within-category",
            period=period,
            chance_level=1 / 3,
            y_range=(0.25, 0.5),
            path_save=ProjectParams.PATH_PAPER.value / "within-category",
        )
        project.decoding_avg(
            mask_rois=project.mask_rois,
            which_data="across-category",
            period=period,
            chance_level=1 / 3,
            y_range=(0.25, 0.5),
            path_save=ProjectParams.PATH_PAPER.value / "across-category",
        )
        project.decoding_avg(
            mask_rois=project.mask_rois,
            which_data="across-category",
            period=period,
            chance_level=1 / 3,
            y_range=(0.25, 0.5),
            dropped_condition="drop",
            path_save=ProjectParams.PATH_PAPER.value / "across-category",
        )
        project.decoding_avg(
            mask_rois=project.mask_rois,
            which_data="status",
            period=period,
            chance_level=1 / 2,
            y_range=(0.45, 0.8),
            path_save=ProjectParams.PATH_PAPER.value / "status",
        )
        project.decoding_avg(
            mask_rois=project.mask_rois,
            which_data="status",
            period=period,
            chance_level=1 / 2,
            y_range=(0.45, 0.65),
            dropped_condition="drop",
            path_save=ProjectParams.PATH_PAPER.value / "status",
        )

        project.decoding_cat_vs_status(
            mask_rois=project.mask_rois,
            period=period,
            compare_analyses=["within-category", "status"],
            compare_conditions=["target", "non-target"],
            chance_level=[0.33, 0.5],
            path_save=ProjectParams.PATH_PAPER.value / "cat_vs_status",
        )

        project.corr_behfmri_vs_posvneg_decoding(
            mask_rois=project.mask_rois,
            period=period,
            compare_analyses=["behavior", "status"],
            compare_conditions=["target", "non-target"],
            path_save=ProjectParams.PATH_PAPER.value / "correlations",
        )
