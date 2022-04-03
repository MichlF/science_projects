# Import modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import pandas as pd
import numpy as np
import nideconv
import itertools
from dataclasses import dataclass, field
from mne.stats import permutation_cluster_1samp_test
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifier,
    RidgeClassifierCV,
)
from scipy.stats import (
    shapiro,
    wilcoxon,
    ttest_rel,
    ttest_1samp,
    sem,
    spearmanr,
    rankdata,
)
from nilearn import input_data, plotting
from nilearn.image import load_img
from pathlib import Path
from os.path import normpath

# to fix the runtime error on Mac (i.e. Python not installed as a framework)
# mpl.use('TkAgg')
# to fix weird matplotlib errors try (Note: will prevent some plots to be shown in VS-Code!):
mpl.use("Agg")

# To import selfmade modules
sys.path.insert(1, str(Path().absolute().parent.absolute()))
from misc.statistics import confidenceInt

"""
Script to analyze fMRI data

© Michel Failing, 2022
"""


@dataclass
class DataPreparation:

    path_fmriprep: str
    path_masks: str
    path_save: str
    path_subs: str
    project_name: str
    nr_runs: int
    t_r: float
    trial_length: list = field(default_factory=list)
    analysis_type: str = "category"
    get_mask: str = "anat"
    mask_name: str = "LOC-msdl"
    decoder: str = "ridgeCV"

    # Sanity checks
    def sanity_checks(self):
        assert t_r > 0 and t_r < 10, f"t_r of {t_r} is either too small or too large."
        assert get_mask in ["anat", "func"], f"Invalid get_mask type {get_mask}."
        assert analysis_type in [
            "category",
            "status",
        ], f"Invalid analysis_type {analysis_type}."

    def mask_details(self, show_figure=True, save_figure=False):
        print("Showing mask...")
        # Load mask
        if self.get_mask == "func":
            mask = self.path_masks + f"/{self.mask_name}/sub_01.nii.gz"
            fig_title = (
                f"Functionally-defined mask of {self.mask_name} (example from sub-1)"
            )
        elif self.get_mask == "anat":
            mask = self.path_masks + f"/{self.mask_name}.nii.gz"
            fig_title = f"Anatomical mask of {self.mask_name}"
        # Check it and show it for inspection
        sample_data = load_img(
            self.path_fmriprep
            + f"/sub-01/ses-01/func/sub-01_ses-01_task-{self.project_name}_run-1_bold_space-MNI152NLin2009cAsym_preproc.nii"
        )
        mask_data = load_img(mask)
        print(
            f"NOTE: This {get_mask} mask has {load_img(mask).shape} shape. The MNI data of sub-1 has {sample_data.shape}"
        )
        assert (
            mask_data.shape == sample_data.shape[:3]
        ), "Mask and actual data are not the same size."
        display = plotting.plot_glass_brain(mask, title=fig_title)
        # Actually show it
        if show_figure:
            print("If mask does not show, check your mpl.use() settings!")
            plotting.show()
        # Save it
        if save_figure:
            display.savefig(self.path_masks + f"/{self.mask_name}_figure.pdf")

        display.close()

    def data_get(self, subjects, include):

        if get_mask == "anat":  # anatomical mask is the same for each subject
            mask = self.path_masks + f"/{self.mask_name}.nii.gz"

        print("Getting data...")
        data_allsubs, design_allsubs, confounds_allsubs = [], [], []
        # Fetch and bring data in the correct format
        for subj in subjects:

            if get_mask == "func":  # functional mask is specific to each subject
                mask = self.path_masks + f"/{self.mask_name}/sub_{subj}.nii.gz"

            masker = input_data.NiftiMasker(mask, t_r=self.t_r)
            # According van Driel et al., it is best to do a single GLM including the high pass filtering in order to avoid spurious decoding.
            # We can do that by using the cosine confound outputs from fmriprep
            # (see http://www.brainvoyager.com/bvqx/doc/UsersGuide/Preprocessing/TemporalHighPassFiltering.html
            # to see some documentation about what they do).
            for run in range(1, runs + 1):
                print(
                    f"Preparing data run {run} of subject {subj} (project {self.project_name})."
                )
                da = masker.fit_transform(
                    self.path_fmriprep
                    + f"/sub-{subj}/ses-01/func/sub-{subj}_ses-01_task-{self.project_name}_run-{run}_bold_space-MNI152NLin2009cAsym_preproc.nii"
                )
                de = pd.read_table(
                    self.path_subs
                    + f"/sub-{subj}/ses-01/func/sub-{subj}_ses-01_task-{self.project_name}_run-{run}_events.tsv"
                )
                co = pd.read_table(
                    self.path_fmriprep
                    + f"/sub-{subj}/ses-01/func/sub-{subj}_ses-01_task-{self.project_name}_run-{run}_bold_confounds.tsv"
                )
                co = co[include].fillna(method="bfill")
                # Create data frames
                da, de, co = pd.DataFrame(da), pd.DataFrame(de), pd.DataFrame(co)

                # Add info
                da["run"], de["run"], co["run"] = run, run, run
                da["subject"], de["subject"], co["subject"] = (
                    int(subj),
                    int(subj),
                    int(subj),
                )

                # Store for all subs
                data_allsubs.append(da)
                design_allsubs.append(de)
                confounds_allsubs.append(co)
        # Save in class
        self.data_allsubs = pd.concat(data_allsubs)
        self.design_allsubs = pd.concat(design_allsubs)
        self.confounds_allsubs = pd.concat(confounds_allsubs)

        return self.data_allsubs, self.design_allsubs, self.confounds_allsubs

    def data_store(self, action, data_content, save_name, object2save="default"):

        # Sanity checks
        assert action in ["save", "load"], "Invalid action for storing/loading object."
        assert data_content in [
            "data",
            "figures",
            "statistics",
        ], "Invalid data_content for storing/loading object."

        # Build base path and interval
        s_path = self.path_save + f"/{data_content}/{self.get_mask}/"
        interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])

        # Finalize path and create folder structure if it doesn't exist yet
        if data_content == "data":
            f_path = s_path + f"{interval}/{self.mask_name}/"
        elif data_content == "figures":
            f_path = s_path + f"{self.analysis_type}/{interval}/{self.mask_name}/data/"
        elif data_content == "statistics":
            f_path = s_path + f"{self.analysis_type}/{interval}/{self.mask_name}/"
        Path(f_path).mkdir(parents=True, exist_ok=True)

        # Save/Load
        if action == "save":
            print(f"Saving {f_path}")
            try:  # if pandas dataframe
                object2save.to_pickle(f_path + f"{save_name}.pkl")
            except Exception as e:  # if numpy array
                print(e, "\nSave as pickle failed, trying .npy...")
                np.save(f_path + f"{save_name}.npy", object2save)
        elif action == "load":
            print(f"Loading {f_path}")
            try:  # if pandas dataframe
                return pd.read_pickle(f_path + f"{save_name}.pkl")
            except Exception as e:  # if numpy array
                print(e, "\nLoading as pickle failed, trying .npy...")
                return np.load(f_path + f"{save_name}.npy")


@dataclass
class Deconvolution(DataPreparation):
    def prepare_deconvolution(self):
        print("-----------------------\n")
        print("Starting deconvolution analysis...")
        # Before deconvolving, clean the design dataframe
        self.design_allsubs.drop(
            ["trial_type", "duration", "condition", "condition_nonTTemplate"],
            axis=1,
            inplace=True,
        )
        self.design_allsubs.rename(
            columns={"trial_type_extended": "event_type"}, inplace=True
        )

    def calc_deconvolution(
        self,
        basis_set="fourier",
        n_regressors=7,
        concatenate_runs=False,
        show_warnings=False,
        solution_type="ols",
        alphas=[1, 0],
        bayesian=False,
    ):
        print("Creating the model and fitting responses...")
        # Create model, add regressors and fit (a whole bunch of OLS fitting!)
        self.g_model = nideconv.GroupResponseFitter(
            self.data_allsubs,
            self.design_allsubs,
            1 / self.t_r,
            concatenate_runs=concatenate_runs,
            confounds=self.confounds_allsubs,
        )
        for key in self.design_allsubs["event_type"].unique():
            self.g_model.add_event(
                key,
                interval=self.trial_length,
                basis_set=basis_set,
                n_regressors=n_regressors,
                show_warnings=show_warnings,
            )
        # Fit the model
        # Note: ridge regression is not supported for multiple features (here voxels), yet
        self.g_model.fit(type=solution_type, alphas=alphas)
        # Hierarchical Bayesian Model (also not supported for multiple features, yet)
        if bayesian:
            self.bayesian_model = (
                nideconv.HierarchicalBayesianModel.from_groupresponsefitter(
                    self.g_model
                )
            )
            self.bayesian_model.build_model()

    def calc_timecourse(
        self,
        measurement="t-values",
        oversampling=1,
        rois=[],
        which_plots="both",
        figure_size=(8, 6),
        style="default",
        y_lims=[],
        padding=10,
        save_data=True,
        save_figure=False,
    ):
        print("Extracting time courses...")
        if measurement == "t-values":  # t-values
            self.tcs = self.g_model.get_t_value_timecourses(oversample=oversampling)
        elif measurement == "psc":  # percent signal change
            self.tcs = self.g_model.get_timecourses(oversample=oversampling)
        else:
            raise KeyError("Unknown keyword for timecourse measurement")

        # Split event type coding into individual columns in the df
        self.tcs["condition"] = (
            pd.Series(self.tcs.index.get_level_values("event type"))
            .apply(lambda x: x.split("-")[0])
            .values
        )
        self.tcs["category"] = (
            pd.Series(self.tcs.index.get_level_values("event type"))
            .apply(lambda x: x.split("-")[1])
            .values
        )
        self.tcs["exemplar"] = (
            pd.Series(self.tcs.index.get_level_values("event type"))
            .apply(lambda x: x.split("-")[2])
            .values
        )
        self.tcs["trial"] = (
            pd.Series(self.tcs.index.get_level_values("event type"))
            .apply(lambda x: x.split("-")[3])
            .values
        )

        ### We might wanna exclude the entire plotting into a separate function
        if save_data or save_figure:
            tcs_data = self.tcs.groupby(["subject", "condition", "time"]).mean().mean(1)
            tcs_data = tcs_data.reset_index().rename(columns={0: measurement})
            tcs_data["time"] = tcs_data["time"].round(1)
            interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
            Path(
                self.path_save
                + f"/figures/{self.get_mask}/BOLD/{interval}/{self.mask_name}/"
            ).mkdir(parents=True, exist_ok=True)

        if save_data:
            tcs_data.to_csv(
                self.path_save
                + f"/figures/{self.get_mask}/BOLD/{interval}/{self.mask_name}/"
                + f"data_all_{measurement}.csv",
                index=False,
            )

            for idx, period in enumerate(rois):
                temp_data = tcs_data.loc[
                    (tcs_data["time"] >= period[0]) & (tcs_data["time"] <= period[1])
                ]
                temp_data = temp_data.groupby(["subject", "condition"]).mean()
                temp_data = temp_data.drop("time", axis=1).reset_index()
                temp_data = temp_data.pivot(
                    index="subject", columns="condition", values=measurement
                )

                # Save
                if idx == 0:
                    temp_data.to_csv(
                        self.path_save
                        + f"/figures/{self.get_mask}/BOLD/{interval}/{self.mask_name}/"
                        + f"data_cue_{measurement}.csv"
                    )
                elif idx == 1:
                    temp_data.to_csv(
                        self.path_save
                        + f"/figures/{self.get_mask}/BOLD/{interval}/{self.mask_name}/"
                        + f"data_search_{measurement}.csv"
                    )
            print("Saved time course data...")

        if save_figure:
            # Fixed params
            plt.style.use(style)
            onset_cue = 1.6  # in secs relative to trial begin
            onset_search = 9.8
            linewidth = 2

            # Plot individual time series
            if which_plots in ["both", "single"]:
                for subj in tcs_data["subject"].unique():
                    fig, ax = plt.subplots(figsize=figure_size)
                    tcs_data_subj = tcs_data.loc[tcs_data["subject"] == subj]
                    for condition in tcs_data["condition"].unique():
                        ax.plot(
                            np.arange(
                                round(tcs_data["time"].unique()[0], 1),
                                round(tcs_data["time"].unique()[-1], 1),
                                step=self.t_r,
                            ),
                            tcs_data_subj.loc[tcs_data_subj["condition"] == condition][
                                measurement
                            ],
                            label=condition,
                        )
                    # Styling
                    ax.legend(loc="upper right")
                    ax.axhline(0.0, color="black", lw=2, linestyle=":")
                    ax.axvline(
                        onset_cue + self.trial_length[0],
                        color="gray",
                        lw=linewidth - 0.5,
                        linestyle="--",
                        dash_capstyle="round",
                        dash_joinstyle="round",
                    )
                    ax.axvline(
                        onset_search + self.trial_length[0],
                        color="gray",
                        lw=linewidth - 0.5,
                        linestyle="--",
                        dash_capstyle="round",
                        dash_joinstyle="round",
                    )
                    # ax.set_ylim(y_lims[0], y_lims[1])
                    ax.set_ylabel(measurement, labelpad=padding)
                    ax.set_xlabel("time (secs)", labelpad=padding)
                    ax.set_xticks(
                        np.arange(
                            round(tcs_data["time"].unique()[0], 1),
                            round(tcs_data["time"].unique()[-1], 1),
                            step=self.t_r * 2,
                        )
                    )
                    ax.set_xticklabels(
                        [
                            str(round(item, 1))
                            for item in np.arange(
                                round(tcs_data["time"].unique()[0], 1) + onset_cue,
                                round(tcs_data["time"].unique()[-1], 1) + onset_cue,
                                step=self.t_r * 2,
                            )
                        ]
                    )
                    # ax.set_yticks(np.arange(y_lims[0], y_lims[1]+.1, step=.2))
                    ax.set_title(
                        f"Sub-{subj} using {measurement} on {self.mask_name} ({self.get_mask})",
                        pad=padding * 2,
                    )
                    # Add shaded areas for ROIs
                    if type(rois[0]) is not list:
                        rois = [rois]
                    for roi in rois:
                        ax.axvspan(roi[0], roi[1], alpha=0.3, color="gray")
                    sns.despine(trim=True)
                    # Save figure
                    plt.savefig(
                        self.path_save
                        + f"/figures/{self.get_mask}/BOLD/{interval}/{self.mask_name}/"
                        + f"fig_sub-{subj}_{measurement}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

            if which_plots in ["both", "all"]:
                colors = ["blue", "orange", "green", "red"]
                fig, ax = plt.subplots(figsize=figure_size)
                tcs_data_mean = (
                    tcs_data.groupby(["condition", "time"]).mean().reset_index()
                )
                for cIdx, condition in enumerate(tcs_data_mean["condition"].unique()):
                    # Data
                    ax.plot(
                        np.arange(
                            round(tcs_data_mean["time"].unique()[0], 1),
                            round(tcs_data_mean["time"].unique()[-1], 1),
                            step=self.t_r,
                        ),
                        tcs_data_mean.loc[tcs_data_mean["condition"] == condition][
                            measurement
                        ],
                        label=condition,
                    )
                    # # Error bars
                    # ax.fill_between(
                    #     np.arange(
                    #         round(tcs_data["time"].unique()[0], 1),
                    #         round(tcs_data["time"].unique()[-1], 1),
                    #         step=self.t_r * 2),
                    #     np.mean(
                    #         tcs_data.loc[tcs_data["condition"] == condition],
                    #         axis=1) + self.errorBars[:, cIdx],
                    #     np.mean(
                    #         tcs_data.loc[tcs_data["condition"] == condition],
                    #         axis=1) - self.errorBars[:, cIdx],
                    #     alpha=.15)
                    # # Clusterpermutation test
                    # real_data = [
                    #     tcs_data.loc[(tcs_data["condition"] == condition) & tcs_data["subject"] == subj]
                    #     for subj in tcs_data["subject"].unique()
                    # ]
                    # diff_for_permu = np.vstack(real_data) - np.full((np.shape(np.vstack(real_data))), 0)
                    # permu = permutation_cluster_1samp_test(X=diff_for_permu, n_permutations=5000, tail=0, out_type="mask", verbose="WARNING")
                    # if permu[1]:
                    #     sign_cl = np.array(list(range(len(permu[0]))))*0
                    #     if len(permu[1]) < 2:
                    #         for cluster_idx, cluster in enumerate(permu[1]):
                    #             if permu[2][cluster_idx] <= .05:
                    #                 sign_cl[cluster] = 1
                    #     elif len(permu[1]) > 1:
                    #         for cluster_idx, cluster in enumerate(permu[1]):
                    #             if permu[2][cluster_idx] <= .05:
                    #                 sign_cl[cluster] = 1
                # Styling
                ax.legend(loc="upper right")
                ax.axhline(0.0, color="black", lw=2, linestyle=":")
                ax.axvline(
                    onset_cue + self.trial_length[0],
                    color="gray",
                    lw=linewidth - 0.5,
                    linestyle="--",
                    dash_capstyle="round",
                    dash_joinstyle="round",
                )
                ax.axvline(
                    onset_search + self.trial_length[0],
                    color="gray",
                    lw=linewidth - 0.5,
                    linestyle="--",
                    dash_capstyle="round",
                    dash_joinstyle="round",
                )
                ax.set_ylabel(measurement, labelpad=padding)
                ax.set_xlabel("time (secs)", labelpad=padding)
                ax.set_xticks(
                    np.arange(
                        round(tcs_data_mean["time"].unique()[0], 1),
                        round(tcs_data_mean["time"].unique()[-1], 1),
                        step=self.t_r * 2,
                    )
                )
                ax.set_xticklabels(
                    [
                        str(round(item, 1))
                        for item in np.arange(
                            round(tcs_data_mean["time"].unique()[0], 1) + onset_cue,
                            round(tcs_data_mean["time"].unique()[-1], 1) + onset_cue,
                            step=self.t_r * 2,
                        )
                    ]
                )
                if y_lims:
                    ax.set_ylim(y_lims[0], y_lims[1])
                    ax.set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, step=0.2))
                ax.set_title(
                    f'Group-average (n={ len(tcs_data["subject"].unique()) }) using {measurement} on {self.mask_name} ({self.get_mask})',
                    pad=padding * 2,
                )
                # Add shaded areas for ROIs
                if type(rois[0]) is not list:
                    rois = [rois]
                for roi in rois:
                    ax.axvspan(roi[0], roi[1], alpha=0.3, color="gray")
                ax.annotate(
                    "cue",
                    xy=((onset_cue + self.trial_length[0]) + 0.05, ax.get_ylim()[1]),
                    annotation_clip=False,
                )
                ax.annotate(
                    "search",
                    xy=((onset_search + self.trial_length[0]) + 0.25, ax.get_ylim()[1]),
                    annotation_clip=False,
                )
                sns.despine(trim=True)
                # Save figure
                plt.savefig(
                    self.path_save
                    + f"/figures/{self.get_mask}/BOLD/{interval}/{self.mask_name}/"
                    + f"fig_all_{measurement}.pdf",
                    bbox_inches="tight",
                )
                plt.close()
                print("Saved time course graphs...")

        return self.tcs


@dataclass
class Decoding(Deconvolution):
    def create_decoder(self, add_decoder=False):
        # Create decoders with parameters for gridsearch
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
                ],
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
            "naiveBayes": GaussianNB(),
        }
        # Update the decoder object if desired
        if add_decoder:
            self.classifier_dict.update(add_decoder)

    def prepare_data(self, paired_conditions=None):
        print("Preparing data for decoding...")
        self.conditions = np.sort(self.tcs.condition.unique())

        if self.analysis_type == "category":
            self.chance_level = 1 / len(self.tcs.category.unique())
            # Average across examplars and trials
            self.tc_d = (
                self.tcs.reset_index(level=["subject", "run"])
                .groupby(["subject", "run", "condition", "category", "time"])
                .mean()
            )

        elif self.analysis_type == "status":
            # Average across examplars and trials
            self.tc_d = (
                self.tcs.reset_index(level=["subject", "run"])
                .groupby(["subject", "run", "condition", "time"])
                .mean()
            )

        # Make subselection if paired testing
        if paired_conditions:
            assert (
                len(paired_conditions) == 2
            ), "paired_conditions is not a list of two conditions"

            self.paired_conditions = paired_conditions
            remain_conditions = self.conditions[
                np.argwhere(
                    (self.conditions == self.paired_conditions[0])
                    | (self.conditions == self.paired_conditions[1])
                )
            ]
            status_remain_conditions = np.concatenate(remain_conditions, axis=0)
            self.tc_d = self.tc_d.loc[
                (
                    self.tc_d.index.get_level_values("condition")
                    == status_remain_conditions[0]
                )
                | (
                    self.tc_d.index.get_level_values("condition")
                    == status_remain_conditions[1]
                )
            ]
            self.conditions = status_remain_conditions
            if self.analysis_type == "status":
                self.chance_level = 1 / len(self.conditions)

        # Restructure data for decoding
        self.all_subjects = {}
        if analysis_type == "category":
            for subject in self.tc_d.index.get_level_values("subject").unique():
                all_data_pTR = {key: list() for key in self.conditions}
                for tr_time in self.tc_d.index.get_level_values("time").unique():
                    for condition in self.conditions:
                        singleTR = self.tc_d.loc[
                            (self.tc_d.index.get_level_values("subject") == subject)
                            & (self.tc_d.index.get_level_values("time") == tr_time)
                            & (
                                self.tc_d.index.get_level_values("condition")
                                == condition
                            )
                        ]
                        pd.options.mode.chained_assignment = None  # Warnings off
                        # drop columns with empty voxels
                        singleTR.dropna(axis="columns", inplace=True)
                        all_data_pTR[condition].append(singleTR)
                        # Conditions key of all_data_pTR is too long (4 conditions * 27 TRs instead of 27 TRs)
                pd.options.mode.chained_assignment = "warn"  # Warnings on

                # Update data dictionary
                self.all_subjects[f"sub-{subject}"] = all_data_pTR

            # Get the target labels
            # Note: labels are already sorted as in the original dataframe,
            # so we can get the values from there and we only need to define it once
            self.target_labels = (
                all_data_pTR[condition][0].index.get_level_values("category").values
            )
            self.run_labels = (
                all_data_pTR[condition][0].index.get_level_values("run").values
            )

        elif analysis_type == "status":
            for subject in self.tc_d.index.get_level_values("subject").unique():
                all_data_pTR = []
                for tr_time in self.tc_d.index.get_level_values("time").unique():
                    singleTR = self.tc_d.loc[
                        (self.tc_d.index.get_level_values("subject") == subject)
                        & (self.tc_d.index.get_level_values("time") == tr_time)
                    ]
                    pd.options.mode.chained_assignment = None  # Warnings off
                    # drop columns with empty voxels
                    singleTR.dropna(axis="columns", inplace=True)
                    all_data_pTR.append(singleTR)
                    # Conditions key of all_data_pTR is too long (4 conditions * 27 TRs instead of 27 TRs)
                pd.options.mode.chained_assignment = "warn"  # Warnings on

                # Update data dictionary
                self.all_subjects[f"sub-{subject}"] = all_data_pTR

            # Get the target labels
            # Note: labels are already sorted as in the original dataframe,
            # so we can get the values from there and we only need to define it once
            self.target_labels = (
                all_data_pTR[0].index.get_level_values("condition").values
            )
            self.run_labels = all_data_pTR[0].index.get_level_values("run").values

        return self.conditions, self.chance_level

    def calc_decoding_within(
        self, decoder="ridgeCV", validation=LeaveOneGroupOut(), n_jobs=-1, **kwargs
    ):

        print(
            f"WITHIN DECODING: Fitting, obtaining prediction scores and cross validating (using {n_jobs} n_jobs)..."
        )
        # Preparations
        self.decoder = decoder
        self.no_TRs = len(self.tc_d.index.get_level_values("time").unique())

        if self.analysis_type == "category":
            self.all_accuracies = {
                condition + "_all": np.zeros([self.no_TRs, 0])
                for condition in self.conditions
            }

            # Fit, predict and validate
            print("Decoding subject: ", end="")
            for subject in self.tc_d.index.get_level_values("subject").unique():
                print(f"{subject}, ", end="")
                single_accuracy = {key: list() for key in self.conditions}

                for condition in self.conditions:
                    all_cv_scores, all_cv_scores_means = [], []
                    for idx in range(self.no_TRs):
                        # Get accuracy scores (could also use sklearn's cross_validate here)
                        cv_score = cross_val_score(
                            self.classifier_dict[self.decoder],
                            self.all_subjects[f"sub-{subject}"][condition][idx],
                            y=self.target_labels,
                            groups=self.run_labels,
                            cv=validation,
                            n_jobs=n_jobs,
                            **kwargs,
                        )
                        # Store individual TR
                        all_cv_scores.append(cv_score)
                        all_cv_scores_means.append(cv_score.mean())
                    single_accuracy[condition] = all_cv_scores_means
                    # Store individual condition
                    self.all_accuracies[f"{condition}_all"] = np.concatenate(
                        (
                            self.all_accuracies[f"{condition}_all"],
                            np.reshape(all_cv_scores_means, (-1, 1)),
                        ),
                        axis=1,
                    )
                # Store individual subject
                self.all_accuracies[f"sub-{subject}"] = single_accuracy
            print("done.", end=""), print("")

        elif self.analysis_type == "status":
            self.all_accuracies = {}
            self.all_accuracies_subs = {"all_subs": np.zeros([self.no_TRs, 0])}

            # Fit, predict and validate
            print("Decoding subject: ", end="")
            for subject in self.tc_d.index.get_level_values("subject").unique():
                print(f"{subject}, ", end="")

                all_cv_scores, all_cv_scores_means = [], []
                for idx in range(self.no_TRs):
                    cv_score = cross_val_score(
                        self.classifier_dict[self.decoder],
                        self.all_subjects[f"sub-{subject}"][idx],
                        y=self.target_labels,
                        groups=self.run_labels,
                        cv=validation,
                        n_jobs=n_jobs,
                        **kwargs,
                    )
                    # Store individual TR
                    all_cv_scores.append(cv_score)
                    all_cv_scores_means.append(cv_score.mean())
                # Store individual condition
                self.all_accuracies_subs["all_subs"] = np.concatenate(
                    (
                        self.all_accuracies_subs["all_subs"],
                        np.reshape(all_cv_scores_means, (-1, 1)),
                    ),
                    axis=1,
                )
                # Store individual subject
                self.all_accuracies[f"sub-{subject}"] = all_cv_scores_means
            print("done.", end=""), print("")

    def calc_decoding_across(self, decoder="ridgeCV"):

        print(
            "ACROSS DECODING: Fitting, obtaining prediction scores and cross validating... (Note: very slow!)"
        )
        # Preparations
        self.decoder = decoder
        self.no_TRs = len(self.tc_d.index.get_level_values("time").unique())
        self.all_accuracies = {}
        self.all_accuracies_subs = {"all_subs": np.zeros([self.no_TRs, 0])}
        condition_orders = [self.conditions, self.conditions[::-1]]

        # Fit, predict and validate
        print("Decoding subject: ", end="")
        for subject in self.tc_d.index.get_level_values("subject").unique():
            print(f"{subject}, ", end="")
            all_cv_scores, all_cv_scores_means = [], []
            for idx in range(self.no_TRs):
                scores = []
                for _, condition_order in enumerate(condition_orders):
                    for _run in np.unique(self.run_labels):
                        test_data = self.all_subjects[f"sub-{subject}"][
                            condition_order[1]
                        ][idx]
                        train_data = self.all_subjects[f"sub-{subject}"][
                            condition_order[0]
                        ][idx]
                        decoded_object = self.classifier_dict[self.decoder].fit(
                            train_data, self.target_labels
                        )
                        scores.append(
                            decoded_object.score(test_data, self.target_labels)
                        )
                all_cv_scores.append(scores)
                all_cv_scores_means.append(np.mean(scores))
            # Store individual condition
            self.all_accuracies_subs["all_subs"] = np.concatenate(
                (
                    self.all_accuracies_subs["all_subs"],
                    np.reshape(all_cv_scores_means, (-1, 1)),
                ),
                axis=1,
            )
            # Store individual subject
            self.all_accuracies[f"sub-{subject}"] = all_cv_scores_means
        print("done.", end=""), print("")

    def calc_errorbars(self, method="SEM"):

        print("Calculating error bars...")
        if method == "SEM":
            if (
                analysis_type == "category"
                and hasattr(self, "paired_conditions") == False
            ):
                # Calculate actual SEMs
                self.errorBars = np.array([])
                for t_rIdx in range(self.no_TRs):
                    means = np.zeros(
                        [len(self.tc_d.index.get_level_values("subject").unique()), 0]
                    )
                    for condition in self.conditions:
                        means = np.concatenate(
                            (
                                means,
                                np.reshape(
                                    self.all_accuracies[f"{condition}_all"][t_rIdx, :],
                                    (-1, 1),
                                ),
                            ),
                            axis=1,
                        )
                    temp_errorBars = sem(means[:, :], axis=0, ddof=1)
                    # Add TR to list of all TRs
                    self.errorBars = (
                        np.vstack([self.errorBars, temp_errorBars])
                        if self.errorBars.size
                        else temp_errorBars
                    )
            elif analysis_type == "status" or hasattr(self, "paired_conditions"):
                temp = []
                for subj in self.all_accuracies.items():
                    temp.append(subj[1:])
                temp = np.transpose(np.squeeze(temp))

                # Calculate actual SEMs
                self.errorBars = np.array([])
                for t_rIdx in range(self.no_TRs):
                    temp_errorBars = sem(temp[t_rIdx, :], axis=0, ddof=1)
                    # Add TR to list of all TRs
                    self.errorBars = (
                        np.vstack([self.errorBars, temp_errorBars])
                        if self.errorBars.size
                        else temp_errorBars
                    )
        elif method == "CI":
            if (
                analysis_type == "category"
                and hasattr(self, "paired_conditions") == False
            ):
                # Calculate actual CIs
                self.errorBars = np.array([])
                for t_rIdx in range(self.no_TRs):
                    means = np.zeros(
                        [len(self.tc_d.index.get_level_values("subject").unique()), 0]
                    )
                    for condition in self.conditions:
                        means = np.concatenate(
                            (
                                means,
                                np.reshape(
                                    self.all_accuracies[f"{condition}_all"][t_rIdx, :],
                                    (-1, 1),
                                ),
                            ),
                            axis=1,
                        )
                    temp_errorBars = confidenceInt.wconfidence_int(means)
                    # Add TR to list of all TRs
                    self.errorBars = (
                        np.vstack([self.errorBars, temp_errorBars])
                        if self.errorBars.size
                        else temp_errorBars
                    )
            elif analysis_type == "status" or hasattr(self, "paired_conditions"):
                raise Exception(
                    "Calculation of confidence intervals is currently only supported for within decoding in condition 'category'"
                )

    def visual_timeseries(
        self,
        which_plots="both",
        rois=[],
        figure_size=(8, 6),
        style="default",
        y_lims=[],
        padding=10,
        save_figure=False,
        save_data=True,
    ):

        print("Visualizing time series...")
        # Fixed params
        plt.style.use(style)
        onset_cue = 1.6  # in secs relative to trial begin
        onset_search = 9.8
        linewidth = 2
        if not y_lims:
            y_lims = [
                0,
                1,
                round((self.chance_level - 0.2) * 2, 1) / 2,
                round((self.chance_level + 0.4) * 2, 1) / 2,
            ]

        # Plot individual time series
        if which_plots in ["both", "single"]:
            for subj in self.tc_d.index.get_level_values("subject").unique():
                fig, ax = plt.subplots(figsize=figure_size)
                if (
                    analysis_type == "category"
                    and hasattr(self, "paired_conditions") == False
                ):
                    for condition in self.conditions:
                        ax.plot(
                            np.linspace(
                                self.tc_d.index.get_level_values("time")[0],
                                self.tc_d.index.get_level_values("time")[-1],
                                self.no_TRs,
                            ),
                            self.all_accuracies[f"sub-{subj}"][condition],
                            label=condition,
                        )
                elif analysis_type == "status" or hasattr(self, "paired_conditions"):
                    status_label = self.conditions[0] + "-" + self.conditions[1]
                    ax.plot(
                        np.linspace(
                            self.tc_d.index.get_level_values("time")[0],
                            self.tc_d.index.get_level_values("time")[-1],
                            self.no_TRs,
                        ),
                        self.all_accuracies[f"sub-{subj}"],
                        label=status_label,
                    )
                # Styling
                ax.legend(loc="upper right")
                ax.axhline(
                    self.chance_level, color="black", lw=linewidth, linestyle=":"
                )
                ax.axvline(
                    onset_cue + self.trial_length[0],
                    color="gray",
                    lw=linewidth - 0.5,
                    linestyle="--",
                    dash_capstyle="round",
                    dash_joinstyle="round",
                )
                ax.axvline(
                    onset_search + self.trial_length[0],
                    color="gray",
                    lw=linewidth - 0.5,
                    linestyle="--",
                    dash_capstyle="round",
                    dash_joinstyle="round",
                )
                ax.set_ylim(y_lims[0], y_lims[1])
                ax.set_ylabel("Classification accuracy", labelpad=padding)
                ax.set_xlabel("time (secs)", labelpad=padding)
                ax.set_xticks(
                    np.arange(
                        round(self.tc_d.index.get_level_values("time")[0], 1),
                        round(self.tc_d.index.get_level_values("time")[-1], 1),
                        step=self.t_r * 2,
                    )
                )
                ax.set_xticklabels(
                    [
                        str(round(item, 1))
                        for item in np.arange(
                            round(self.tc_d.index.get_level_values("time")[0], 1)
                            + onset_cue,
                            round(self.tc_d.index.get_level_values("time")[-1], 1)
                            + onset_cue,
                            step=self.t_r * 2,
                        )
                    ]
                )
                ax.set_yticks(np.arange(y_lims[0], y_lims[1] + 0.1, step=0.1))
                ax.set_title(
                    f"Sub-{subj} with {self.decoder} on {self.mask_name} ({self.get_mask})",
                    pad=padding,
                )
                # Add shaded areas for ROIs
                if type(rois[0]) is not list:
                    rois = [rois]
                for roi in rois:
                    ax.axvspan(roi[0], roi[1], alpha=0.3, color="gray")
                sns.despine(trim=True)
                # Save
                if save_figure or save_data:
                    interval = (
                        str(self.trial_length[0]) + "_" + str(self.trial_length[1])
                    )
                    path_save = (
                        self.path_save
                        + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
                    )
                    if self.analysis_type == "status" or hasattr(
                        self, "paired_conditions"
                    ):
                        path_save += f"{self.conditions[0]}_vs_{self.conditions[1]}/"
                    Path(path_save + "/data/").mkdir(parents=True, exist_ok=True)
                    if save_data:
                        pass  # probably dont wanna save individual data here
                    if save_figure:
                        plt.savefig(
                            path_save + f"{self.decoder}_sub-{subj}_acrossTime.pdf",
                            bbox_inches="tight",
                        )
                plt.close()

        # Plot averaged time series
        if which_plots in ["both", "all"]:
            colors = ["blue", "orange", "green", "red"]
            fig, ax = plt.subplots(figsize=figure_size)
            if (
                self.analysis_type == "category"
                and hasattr(self, "paired_conditions") == False
            ):
                for cIdx, condition in enumerate(self.conditions):
                    # Data
                    ax.plot(
                        np.linspace(
                            self.tc_d.index.get_level_values("time")[0],
                            self.tc_d.index.get_level_values("time")[-1],
                            self.no_TRs,
                        ),
                        np.mean(self.all_accuracies[f"{condition}_all"][:, :], axis=1),
                        label=condition,
                    )
                    # Error bars
                    ax.fill_between(
                        np.linspace(
                            self.tc_d.index.get_level_values("time")[0],
                            self.tc_d.index.get_level_values("time")[-1],
                            self.no_TRs,
                        ),
                        np.mean(self.all_accuracies[f"{condition}_all"][:, :], axis=1)
                        + self.errorBars[:, cIdx],
                        np.mean(self.all_accuracies[f"{condition}_all"][:, :], axis=1)
                        - self.errorBars[:, cIdx],
                        alpha=0.15,
                    )  # Note errorBars is index sliced, while self.all_accuracies is keyword sliced
                    # Clusterpermutation test
                    real_data = [
                        self.all_accuracies[f"sub-{subj}"][condition]
                        for subj in self.tc_d.index.get_level_values("subject").unique()
                    ]
                    diff_for_permu = np.vstack(real_data) - np.full(
                        (np.shape(np.vstack(real_data))), self.chance_level
                    )
                    permu = permutation_cluster_1samp_test(
                        X=diff_for_permu,
                        n_permutations=5000,
                        tail=0,
                        out_type="mask",
                        verbose="WARNING",
                    )
                    if permu[1]:
                        sign_cl = np.array(list(range(len(permu[0])))) * 0
                        if len(permu[1]) < 2:
                            for cluster_idx, cluster in enumerate(permu[1]):
                                if permu[2][cluster_idx] <= 0.05:
                                    sign_cl[cluster] = 1
                        elif len(permu[1]) > 1:
                            for cluster_idx, cluster in enumerate(permu[1]):
                                if permu[2][cluster_idx] <= 0.05:
                                    sign_cl[cluster] = 1
                    # Save data
                    if save_data:
                        if "to_save" in locals():
                            temp_save = pd.DataFrame(np.vstack(real_data))
                            temp_save["subject_nr"] = list(
                                self.tc_d.index.get_level_values("subject").unique()
                            )
                            temp_save["condition"] = condition
                            to_save = pd.concat([to_save, temp_save], axis=0)
                        else:
                            to_save = pd.DataFrame(np.vstack(real_data))
                            to_save["subject_nr"] = list(
                                self.tc_d.index.get_level_values("subject").unique()
                            )
                            to_save["condition"] = condition
                    # Permutation plotting
                    try:
                        ax.scatter(
                            np.sort(self.tc_d.index.get_level_values("time").unique()),
                            sign_cl * (self.chance_level - ((cIdx + 1) * 0.025)),
                            color=colors[cIdx],
                            marker="_",
                            linewidth=linewidth,
                        )
                    except Exception as e:
                        print(e)
                        print("Clusterpermutation gave no cluster to plot!")

            elif self.analysis_type == "status" or hasattr(self, "paired_conditions"):
                temp = []
                for subj in self.all_accuracies.items():
                    temp.append(subj[1:])
                temp = np.transpose(np.squeeze(temp))
                status_label = (
                    self.paired_conditions[0] + "-" + self.paired_conditions[1]
                )
                # Data
                ax.plot(
                    np.linspace(
                        self.tc_d.index.get_level_values("time")[0],
                        self.tc_d.index.get_level_values("time")[-1],
                        self.no_TRs,
                    ),
                    np.mean(temp, axis=1),
                    label=status_label,
                )
                # Error bars
                ax.fill_between(
                    np.linspace(
                        self.tc_d.index.get_level_values("time")[0],
                        self.tc_d.index.get_level_values("time")[-1],
                        self.no_TRs,
                    ),
                    np.mean(temp, axis=1) + np.squeeze(self.errorBars),
                    np.mean(temp, axis=1) - np.squeeze(self.errorBars),
                    alpha=0.15,
                )  # Note errorBars is index sliced, while self.all_accuracies is keyword sliced
                # Clusterpermutation test
                real_data = [
                    self.all_accuracies[f"sub-{subj}"]
                    for subj in self.tc_d.index.get_level_values("subject").unique()
                ]
                diff_for_permu = np.vstack(real_data) - np.full(
                    (np.shape(np.vstack(real_data))), self.chance_level
                )
                permu = permutation_cluster_1samp_test(
                    X=diff_for_permu,
                    n_permutations=5000,
                    tail=0,
                    out_type="mask",
                    verbose="WARNING",
                )
                sign_cl = np.array(list(range(len(permu[0])))) * 0
                for cluster_idx, cluster in enumerate(permu[1]):
                    if permu[2][cluster_idx] <= 0.05:
                        sign_cl[cluster] = 1
                # Save data
                if save_data:
                    to_save = pd.DataFrame(np.vstack(real_data))
                    to_save["subject_nr"] = list(
                        self.tc_d.index.get_level_values("subject").unique()
                    )
                    to_save["condition"] = status_label
                # Permutation plotting
                try:
                    ax.scatter(
                        np.sort(self.tc_d.index.get_level_values("time").unique()),
                        sign_cl * (self.chance_level - 0.025),
                        color=colors[0],
                        marker="_",
                        linewidth=linewidth,
                    )
                except Exception as e:
                    print(e)
                    print("Clusterpermutation gave no cluster to plot!")

            # Styling
            ax.legend(loc="upper right")
            ax.axhline(self.chance_level, color="black", lw=linewidth, linestyle=":")
            ax.axvline(
                onset_cue + self.trial_length[0],
                color="gray",
                lw=linewidth - 0.5,
                linestyle="--",
                dash_capstyle="round",
                dash_joinstyle="round",
            )
            ax.axvline(
                onset_search + self.trial_length[0],
                color="gray",
                lw=linewidth - 0.5,
                linestyle="--",
                dash_capstyle="round",
                dash_joinstyle="round",
            )
            ax.set_ylim(y_lims[2], y_lims[3])
            ax.set_ylabel("Classification accuracy", labelpad=padding)
            ax.set_yticks(np.arange(y_lims[2], y_lims[3] + 0.1, step=0.1))
            ax.set_xticks(
                np.arange(
                    round(self.tc_d.index.get_level_values("time")[0], 1),
                    round(self.tc_d.index.get_level_values("time")[-1], 1),
                    step=self.t_r * 2,
                )
            )
            ax.set_xticklabels(
                [
                    str(round(item, 1))
                    for item in np.arange(
                        round(self.tc_d.index.get_level_values("time")[0], 1)
                        + onset_cue,
                        round(self.tc_d.index.get_level_values("time")[-1], 1)
                        + onset_cue,
                        step=self.t_r * 2,
                    )
                ]
            )
            ax.set_xlabel("time (secs)", labelpad=padding)
            ax.set_title(
                f"Group-average (n={ len(self.tc_d.index.get_level_values('subject').unique()) }) with {self.decoder} classifier on {self.mask_name} ({self.get_mask})",
                pad=padding * 2,
            )
            # Add shaded areas for ROIs
            if type(rois[0]) is not list:
                rois = [rois]
            for roi in rois:
                ax.axvspan(roi[0], roi[1], alpha=0.3, color="gray")
            ax.annotate(
                "cue",
                xy=((onset_cue + self.trial_length[0]) + 0.05, ax.get_ylim()[1]),
                annotation_clip=False,
            )
            ax.annotate(
                "search",
                xy=((onset_search + self.trial_length[0]) + 0.25, ax.get_ylim()[1]),
                annotation_clip=False,
            )
            sns.despine(trim=True)
            plt.tight_layout()
            # Save data and figure
            if save_data or save_figure:
                # Create path
                interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
                path_save = (
                    self.path_save
                    + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
                )
                if self.analysis_type == "status" or hasattr(self, "paired_conditions"):
                    path_save += f"{self.conditions[0]}_vs_{self.conditions[1]}/"
                Path(path_save + "/data/").mkdir(parents=True, exist_ok=True)
                # Prepare & save data
                if save_data:
                    to_save["mask"] = self.get_mask
                    to_save["area"] = self.mask_name
                    to_save.to_csv(
                        path_save + f"data/{self.decoder}_mean_acrossTime.csv",
                        index=False,
                    )
                # Save figure
                if save_figure:
                    plt.savefig(
                        path_save + f"{self.decoder}_mean_acrossTime.pdf",
                        bbox_inches="tight",
                    )
            plt.close()

    def visual_periods(
        self,
        rois=[],
        titles=[],
        figure_size=(8, 6),
        save_figure=False,
        save_data=True,
        y_lims=[],
        padding=10,
    ):

        # Some params
        periods = [[], []]
        periods_name = ["cue", "search"]
        cis = []
        linewidth = 2
        if not y_lims:
            y_lims = [self.chance_level - 0.2, self.chance_level + 0.4]
        fig, ax = plt.subplots(1, len(rois), figsize=figure_size)

        if type(rois[0]) is not list:
            rois = [rois]
        for roi_idx, roi in enumerate(rois):
            roi_out = []
            # Find the index corresponding to the time value of roi
            for roi_in in roi:
                roi_out.append(
                    self.find_nearest(self.tc_d.index.get_level_values("time"), roi_in)[
                        0
                    ]
                )

            periods[roi_idx].append(
                pd.DataFrame(
                    data=np.arange(
                        len(self.tc_d.index.get_level_values("subject").unique())
                    ),
                    columns=["subject"],
                )
            )

            if (
                self.analysis_type == "category"
                and hasattr(self, "paired_conditions") == False
            ):
                x_labels = self.conditions
                # Get data from ROI
                for condition in self.conditions:
                    periods[roi_idx].append(
                        pd.DataFrame(
                            data=np.mean(
                                self.all_accuracies[f"{condition}_all"][
                                    roi_out[0] : roi_out[1], :
                                ],
                                axis=0,
                            ),
                            columns=[condition],
                        )
                    )
                periods[roi_idx] = pd.concat(periods[roi_idx], axis=1)
                periods[roi_idx].set_index("subject", inplace=True)

                # Calculate CIs
                cis.append(confidenceInt.wconfidence_int(periods[roi_idx].values))
                print("INFO: Within-CIs (Morey corrected) are used.")

            elif self.analysis_type == "status" or hasattr(self, "paired_conditions"):
                x_labels = [self.conditions[0] + "-" + self.conditions[1]]
                # Get data from ROI
                periods[roi_idx].append(
                    pd.DataFrame(
                        data=np.mean(
                            self.all_accuracies_subs["all_subs"][
                                roi_out[0] : roi_out[1], :
                            ],
                            axis=0,
                        ),
                        columns=x_labels,
                    )
                )
                periods[roi_idx] = pd.concat(periods[roi_idx], axis=1)
                periods[roi_idx].set_index("subject", inplace=True)

                # Calculate CIs
                cis.append(sem(periods[roi_idx].values))
                print(
                    "INFO: Status decoding has only a single condition. SEM instead of CIs are used."
                )

            # Melt for seaborn plotting
            periods[roi_idx] = pd.melt(
                periods[roi_idx],
                value_name="classification accuracy",
                var_name="condition",
            )
            periods[roi_idx]["period"] = periods_name[roi_idx]

            # Plotting
            sns.barplot(
                x="condition",
                y="classification accuracy",
                data=periods[roi_idx],
                ax=ax[roi_idx],
                yerr=cis[roi_idx],
                ci=None,
            )
            # sns.violinplot(x='condition', y='classification accuracy', data=periods[roi_idx], ax=ax[roi_idx])
            sns.swarmplot(
                x="condition",
                y="classification accuracy",
                data=periods[roi_idx],
                edgecolor="black",
                linewidth=0.3,
                dodge=True,
                ax=ax[roi_idx],
                alpha=0.75,
            )
            ax[roi_idx].set_ylim(y_lims[0], y_lims[1])
            ax[roi_idx].set_xticklabels(x_labels, rotation=45)
            ax[roi_idx].axhline(
                self.chance_level, color="black", lw=linewidth, linestyle=":"
            )
            ax[roi_idx].set_xlabel("")
            ax[roi_idx].set_title(titles[roi_idx])
            if roi_idx == 0:
                ax[roi_idx].set_ylabel("Classification accuracy", labelpad=padding)
            else:
                ax[roi_idx].set_ylabel("")
            sns.despine(trim=False)
        plt.tight_layout()
        # Save data and figure
        if save_data or save_figure:
            # Create path
            interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
            path_save = (
                self.path_save
                + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
            )
            if self.analysis_type == "status" or hasattr(self, "paired_conditions"):
                path_save += f"{self.conditions[0]}_vs_{self.conditions[1]}/"
            Path(path_save + "/data/").mkdir(parents=True, exist_ok=True)
            # Prepare & save data
            if save_data:
                to_save = pd.concat([periods[0], periods[1]])
                if (
                    self.analysis_type == "category"
                    and hasattr(self, "paired_conditions") == False
                ):
                    to_save["subject_nr"] = (
                        list(self.tc_d.index.get_level_values("subject").unique())
                        * len(self.conditions)
                        * len(rois)
                    )
                elif self.analysis_type == "status" or hasattr(
                    self, "paired_conditions"
                ):
                    to_save["subject_nr"] = list(
                        self.tc_d.index.get_level_values("subject").unique()
                    ) * len(rois)
                to_save["mask"] = self.get_mask
                to_save["area"] = self.mask_name
                to_save.to_csv(
                    path_save + f"data/{self.decoder}_mean_ROIs.csv", index=False
                )
            # Save figure
            if save_figure:
                plt.savefig(
                    path_save + f"{self.decoder}_mean_ROIs.pdf", bbox_inches="tight"
                )
        plt.close()

        return periods

    def t_testing(self, data_stats, conditionA, conditionB=None):

        assert (
            type(conditionB) == str
            or type(conditionB) == float
            or type(conditionB) == int
        ), f"ConditionB {conditionB} in t_testing is not the right format."

        if type(conditionB) == str:  # conditionA vs conditionB
            _, condA_norm = shapiro(
                np.array(data_stats.loc[data_stats["condition"] == conditionA])[:, 1]
            )
            _, condB_norm = shapiro(
                np.array(data_stats.loc[data_stats["condition"] == conditionB])[:, 1]
            )
            t_stats, p_val = ttest_rel(
                np.array(data_stats.loc[data_stats["condition"] == conditionA])[:, 1],
                np.array(data_stats.loc[data_stats["condition"] == conditionB])[:, 1],
            )
            test_name = f"{conditionA} vs. {conditionB}"
            t_stats_wil, p_val_wil = wilcoxon(
                np.array(data_stats.loc[data_stats["condition"] == conditionA])[:, 1],
                np.array(data_stats.loc[data_stats["condition"] == conditionB])[:, 1],
            )
        elif (
            type(conditionB) == int or type(conditionB) == float
        ):  # conditionA vs Baseline
            _, condA_norm = shapiro(
                np.array(data_stats.loc[data_stats["condition"] == conditionA])[:, 1]
            )
            _, condB_norm = shapiro(
                [conditionB]
                * len(
                    np.array(data_stats.loc[data_stats["condition"] == conditionA])[
                        :, 1
                    ]
                )
            )
            t_stats, p_val = ttest_1samp(
                np.array(data_stats.loc[data_stats["condition"] == conditionA])[:, 1],
                conditionB,
            )
            t_stats_wil, p_val_wil = wilcoxon(
                np.array(data_stats.loc[data_stats["condition"] == conditionA])[:, 1],
                [conditionB]
                * len(
                    np.array(data_stats.loc[data_stats["condition"] == conditionA])[
                        :, 1
                    ]
                ),
            )
            test_name = f"{conditionA} vs. {baseline}"

        return test_name, condA_norm, condB_norm, t_stats, p_val, t_stats_wil, p_val_wil

    def find_nearest(self, array, value):

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return idx, array[idx]

    def calc_statistics(self, data, save_data=True):

        print("Calculating decoding statistics...")
        to_save = pd.DataFrame()
        for roi_idx, roi_name in enumerate(rois_name):
            print(f"Statistical testing for {roi_name}...")
            if (
                self.analysis_type == "category"
                and hasattr(self, "paired_conditions") == False
            ):
                # Conditions vs. baseline
                for _, cond in enumerate(data[roi_idx].condition.unique()):
                    test_stats = self.t_testing(
                        data_stats=data[roi_idx], conditionA=cond, conditionB=baseline
                    )
                    print(test_stats)
                    to_save = to_save.append(
                        {
                            "period": roi_name,
                            "test name": test_stats[0],
                            "normality_A": test_stats[1],
                            "normality_B": test_stats[2],
                            "t_stats": test_stats[3],
                            "p_val": test_stats[4],
                            "t_stats_wil": test_stats[5],
                            "p_val_wil": test_stats[6],
                        },
                        ignore_index=True,
                    )
                # Conditions vs. each other
                for subset in itertools.combinations(conditions, 2):
                    test_stats = self.t_testing(
                        data_stats=data[roi_idx],
                        conditionA=subset[0],
                        conditionB=subset[1],
                    )
                    print(test_stats)
                    to_save = to_save.append(
                        {
                            "period": roi_name,
                            "test name": test_stats[0],
                            "normality_A": test_stats[1],
                            "normality_B": test_stats[2],
                            "t_stats": test_stats[3],
                            "p_val": test_stats[4],
                            "t_stats_wil": test_stats[5],
                            "p_val_wil": test_stats[6],
                        },
                        ignore_index=True,
                    )
            elif self.analysis_type == "status" or hasattr(self, "paired_conditions"):
                # Conditions vs. baseline
                for _, cond in enumerate(data[roi_idx].condition.unique()):
                    test_stats = self.t_testing(
                        data_stats=data[roi_idx], conditionA=cond, conditionB=baseline
                    )
                    print(test_stats)
                    to_save = to_save.append(
                        {
                            "period": roi_name,
                            "test name": test_stats[0],
                            "normality_A": test_stats[1],
                            "normality_B": test_stats[2],
                            "t_stats": test_stats[3],
                            "p_val": test_stats[4],
                            "t_stats_wil": test_stats[5],
                            "p_val_wil": test_stats[6],
                        },
                        ignore_index=True,
                    )
        # Save
        if save_data:
            interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
            path_stats = (
                self.path_save
                + f"/statistics/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}"
            )
            if self.analysis_type == "status" or hasattr(self, "paired_conditions"):
                path_stats += f"/{self.conditions[0]}_vs_{self.conditions[1]}/"
            Path(path_stats).mkdir(parents=True, exist_ok=True)
            to_save.to_csv(path_stats + f"/{self.decoder}_decoding.csv", index=False)


@dataclass
class RDM(Decoding):
    def prepare_RDM(self, paired_conditions=None, drop_condition=False):
        print("Starting representational dissimililarity matrix (RDM) analysis...")
        self.remain_conditions = np.sort(self.tcs.condition.unique())

        # Average over identical trials (i.e. trials column)
        self.data_RDM = (
            self.tcs.reset_index(level=["subject"])
            .groupby(["subject", "condition", "category", "exemplar", "time"])
            .mean()
        )  # groupby is slow with large matrices

        # Drop condiions: currently only supports dropping a single condition
        if drop_condition:
            try:
                self.drop_condition = drop_condition
                self.remain_conditions = list(self.remain_conditions)
                self.remain_conditions.remove(drop_condition)
                self.remain_conditions = np.array(self.remain_conditions)
                self.remain_conditions = self.remain_conditions.astype("object")
                # Remove unnecessary conditions
                self.data_RDM = self.data_RDM.loc[
                    (
                        self.data_RDM.index.get_level_values("condition")
                        == self.remain_conditions[0]
                    )
                    | (
                        self.data_RDM.index.get_level_values("condition")
                        == self.remain_conditions[1]
                    )
                    | (
                        self.data_RDM.index.get_level_values("condition")
                        == self.remain_conditions[2]
                    )
                ]
            except Exception as e:
                print("Removal of conditions from RDM failed.\n", e)

        if paired_conditions:
            assert (
                len(paired_conditions) == 2
            ), "paired_conditions is not a list of two conditions"
            assert (
                drop_condition == False
            ), "paired_conditions and drop_condition are defined. This might cause issues"
            remain_conditions = self.remain_conditions[
                np.argwhere(
                    (self.remain_conditions == paired_conditions[0])
                    | (self.remain_conditions == paired_conditions[1])
                )
            ]
            status_remain_conditions = np.concatenate(remain_conditions, axis=0)
            self.data_RDM = self.data_RDM[
                (
                    self.data_RDM.index.get_level_values("condition")
                    == status_remain_conditions[0]
                )
                | (
                    self.data_RDM.index.get_level_values("condition")
                    == status_remain_conditions[1]
                )
            ]
            self.remain_conditions = status_remain_conditions
            self.paired_conditions = paired_conditions

        # Get axis labels
        # Note: When resetting the pandas index in data_RDM, all former indices become alphabetically/numerically sorted, so sort them here as well.
        self.axis_labels = []
        for condition in self.data_RDM.index.get_level_values("condition").unique():
            for category in self.data_RDM.index.get_level_values("category").unique():
                for exemplar in self.data_RDM.index.get_level_values(
                    "exemplar"
                ).unique():
                    self.axis_labels.append(condition + "-" + category + "-" + exemplar)

    def calc_RDM(self):

        print("Calculating RDM...")
        # Calculate RDM for each TR of each subject separately
        # Then, we average it across subjects to obtain TR specific RDMs
        self.RDM_TR = np.zeros(
            [
                len(self.axis_labels),
                len(self.axis_labels),
                len(self.data_RDM.index.get_level_values("time").unique()),
            ]
        )
        # shut off the warnings that will come with usage of dropna
        pd.options.mode.chained_assignment = None
        self.all_RDM = np.zeros(
            [
                len(self.axis_labels),
                len(self.axis_labels),
                len(self.data_RDM.index.get_level_values("time").unique()),
                len(self.data_RDM.index.get_level_values("subject").unique()),
            ]
        )

        # Main loop
        for TR_idx, TR_real in enumerate(
            self.data_RDM.index.get_level_values("time").unique()
        ):
            average = np.zeros(
                [
                    len(self.axis_labels),
                    len(self.axis_labels),
                    len(self.data_RDM.index.get_level_values("subject").unique()),
                ]
            )

            # Sub loop
            for subjIdx, subject in enumerate(
                self.data_RDM.index.get_level_values("subject").unique()
            ):
                self.tc_RDM_temp = self.data_RDM.loc[
                    (self.data_RDM.index.get_level_values("subject") == subject)
                    & (self.data_RDM.index.get_level_values("time") == TR_real)
                ]  # shape when all conditions (48 x 2962 = (condition x category x examplar)x(voxel's t-value))
                # Calculate spearman rho matrix for specific run
                # drop columns with non-existent voxel
                self.tc_RDM_temp.dropna(axis="columns", inplace=True)
                rho, _p_vals = spearmanr(self.tc_RDM_temp, axis=1)  # shape ()
                # Invert the matrix and store it
                average[:, :, subjIdx] = 1 - rho
                self.all_RDM[:, :, TR_idx, subjIdx] = 1 - rho
            # Average across all subjects
            self.RDM_TRavg = np.mean(average, axis=2)
            # Rank and normalize
            rho_rank = rankdata(self.RDM_TRavg)
            rhoMax, rhoMin = rho_rank.max(), rho_rank.min()
            rho_rnorm = (rho_rank - rhoMin) / (rhoMax - rhoMin)
            # Rankdata returns a 1D array, so reshape back into a rho shape
            self.RDM_TR[:, :, TR_idx] = np.reshape(
                rho_rnorm, (np.shape(self.RDM_TRavg)[0], np.shape(self.RDM_TRavg)[1])
            )
        pd.options.mode.chained_assignment = "warn"  # turn warnings back on

        return self.RDM_TR

    def visual_RDM_timeseries(self, save_data=True, save_figure=False):

        print("Visualizing RDM timeseries...")
        # Start plotting
        for TR_idx, TR_real in enumerate(
            self.data_RDM.index.get_level_values("time").unique()
        ):
            fig, ax = plt.subplots(figsize=(9, 9))
            sns.heatmap(
                self.RDM_TR[:, :, TR_idx],
                vmin=0,
                vmax=1,
                center=0.5,
                linewidths=0.025,
                linecolor="black",
                cmap="RdBu_r",
                square=True,
                cbar_kws={"label": "dissimilarity"},
                yticklabels=self.axis_labels,
                xticklabels=self.axis_labels,
                annot=False,
            )
            ax.set_title(f"correlation matrix at {round(TR_real,1)} secs", pad=10)
            # plt.gcf().subplots_adjust(left=0.22) # adjust space for ylabel
            plt.tight_layout()
            if save_figure or save_data:
                interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
                path_save = (
                    self.path_save
                    + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}"
                )
                if hasattr(self, "paired_conditions"):
                    path_save += (
                        f"/{self.remain_conditions[0]}_vs_{self.remain_conditions[1]}/"
                    )
                path_save += "/RDM-MDS/"
                if len(self.remain_conditions) == 3:
                    path_save += f"dropped_{self.drop_condition}/"
                Path(path_save + "/data/").mkdir(parents=True, exist_ok=True)
                if save_data:
                    pass
                if save_figure:
                    plt.savefig(
                        path_save + f"{self.decoder}_RDM_TR{round(TR_real,1)}.pdf"
                    )
            plt.close()

    def visual_RDM_periods(self, rois=[], labels=[], save_data=True, save_figure=False):

        print("Visualizing RDM periods...")
        self.corr_matrix_psubj = np.zeros(
            [
                len(self.axis_labels),
                len(self.axis_labels),
                len(self.data_RDM.index.get_level_values("subject").unique()),
                len(rois),
            ]
        )
        self.corr_matrix = np.zeros(
            [len(self.axis_labels), len(self.axis_labels), len(rois)]
        )
        if type(rois[0]) is not list:
            rois = [rois]
        for roi_idx, roi in enumerate(rois):
            roi_out = []
            # Find the index corresponding to the time value of t_r roi
            for roi_in in roi:
                roi_out.append(
                    self.find_nearest(self.tcs.index.get_level_values("time"), roi_in)[
                        0
                    ]
                )
            # Average across ROI
            matrix = np.mean(self.all_RDM[:, :, roi_out[0] : roi_out[1]], axis=2)
            for subj_idx, subj in enumerate(
                self.tcs.index.get_level_values("subject").unique()
            ):
                rho_rank = rankdata(matrix[:, :, subj_idx])
                rho_max, rho_min = rho_rank.max(), rho_rank.min()
                rho_rnorm = (rho_rank - rho_min) / (rho_max - rho_min)
                self.corr_matrix_psubj[:, :, subj_idx, roi_idx] = np.reshape(
                    rho_rnorm,
                    (np.shape(self.RDM_TRavg)[0], np.shape(self.RDM_TRavg)[1]),
                )

                # Start plotting average across ROI
                fig, ax = plt.subplots(figsize=(9, 9))
                sns.heatmap(
                    self.corr_matrix_psubj[:, :, subj_idx, roi_idx],
                    vmin=0,
                    vmax=1,
                    center=0.5,
                    linewidths=0.025,
                    linecolor="black",
                    cmap="RdBu_r",
                    square=True,
                    cbar_kws={"label": "dissimilarity"},
                    yticklabels=self.axis_labels,
                    xticklabels=self.axis_labels,
                    annot=False,
                )
                ax.set_title(f"correlation matrix for {labels[roi_idx]} period", pad=10)
                # plt.gcf().subplots_adjust(left=0.22) # adjust space on for ylabel
                plt.tight_layout()
                if save_figure or save_data:
                    interval = (
                        str(self.trial_length[0]) + "_" + str(self.trial_length[1])
                    )
                    path_save = (
                        self.path_save
                        + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
                    )
                    if hasattr(self, "paired_conditions"):
                        path_save += f"/{self.remain_conditions[0]}_vs_{self.remain_conditions[1]}/"
                    path_save += "/RDM-MDS/"
                    if len(self.remain_conditions) == 3:
                        path_save += f"dropped_{self.drop_condition}/"
                    path_save += "/subjects/"
                    Path(path_save + "/data/").mkdir(parents=True, exist_ok=True)
                    if save_data:
                        to_save = pd.DataFrame(
                            self.corr_matrix_psubj[:, :, subj_idx, roi_idx]
                        )
                        to_save["period"] = labels[roi_idx]
                        to_save.to_csv(
                            path_save
                            + f"data/{self.decoder}_RDM_mean_{labels[roi_idx]}_sub-{subj}.csv",
                            index=False,
                        )
                    if save_figure:
                        plt.savefig(
                            path_save
                            + f"{self.decoder}_RDM_mean_{labels[roi_idx]}_sub-{subj}.pdf"
                        )
                plt.close()

            # Average across ROI and subjects
            matrix = np.mean(
                np.mean(self.all_RDM[:, :, roi_out[0] : roi_out[1], :], axis=2), axis=2
            )
            rho_rank = rankdata(matrix)
            rho_max, rho_min = rho_rank.max(), rho_rank.min()
            rho_rnorm = (rho_rank - rho_min) / (rho_max - rho_min)
            self.corr_matrix[:, :, roi_idx] = np.reshape(
                rho_rnorm, (np.shape(self.RDM_TRavg)[0], np.shape(self.RDM_TRavg)[1])
            )

            # Start plotting average across ROI and subjects
            fig, ax = plt.subplots(figsize=(9, 9))
            sns.heatmap(
                self.corr_matrix[:, :, roi_idx],
                vmin=0,
                vmax=1,
                center=0.5,
                linewidths=0.025,
                linecolor="black",
                cmap="RdBu_r",
                square=True,
                cbar_kws={"label": "dissimilarity"},
                yticklabels=self.axis_labels,
                xticklabels=self.axis_labels,
                annot=False,
            )
            ax.set_title(f"correlation matrix for {labels[roi_idx]} period", pad=10)
            # plt.gcf().subplots_adjust(left=0.22) # adjust space on for ylabel
            plt.tight_layout()
            if save_figure or save_data:
                interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
                path_save = (
                    self.path_save
                    + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
                )
                if hasattr(self, "paired_conditions"):
                    path_save += (
                        f"/{self.remain_conditions[0]}_vs_{self.remain_conditions[1]}/"
                    )
                path_save += "/RDM-MDS/"
                if len(self.remain_conditions) == 3:
                    path_save += f"/dropped_{self.drop_condition}/"
                Path(path_save + "/data/").mkdir(parents=True, exist_ok=True)
                if save_data:
                    to_save = pd.DataFrame(self.corr_matrix[:, :, roi_idx])
                    to_save["period"] = labels[roi_idx]
                    to_save.to_csv(
                        path_save
                        + f"data/{self.decoder}_RDM_mean_{labels[roi_idx]}.csv",
                        index=False,
                    )
                if save_figure:
                    plt.savefig(
                        path_save + f"{self.decoder}_RDM_mean_{labels[roi_idx]}.pdf"
                    )
            plt.close()

        return self.corr_matrix, self.corr_matrix_psubj

    def calc_RDM_stats(
        self, corr_matrix, shift=-1, rois=[[2.6, 8.2], [11.0, 16.6]], save_data=True
    ):

        print("Calculating RDM statistics...")
        # Preparations
        conditions = list(sorted(self.remain_conditions))
        one_block = 3 * 4  # categories x exemplars
        comparisons, comparisons_idx = [], 0
        dis_same_cat = np.zeros(
            (
                len(list(itertools.combinations(conditions, 2))),
                len(self.tcs.index.get_level_values("subject").unique()),
                len(rois),
            )
        )
        dis_diff_cat = np.zeros(
            (
                len(list(itertools.combinations(conditions, 2))),
                len(self.tcs.index.get_level_values("subject").unique()),
                len(rois),
            )
        )

        # Prepare data for stats
        for subset in itertools.combinations(conditions, 2):
            comparisons.append(subset)
            conditionA_idx = conditions.index(subset[0])
            conditionB_idx = conditions.index(subset[1])

            for roi_idx, _ in enumerate(rois):
                for subj_idx, subj in enumerate(
                    self.tcs.index.get_level_values("subject").unique()
                ):
                    comparison_mat = corr_matrix[
                        conditionB_idx * one_block : conditionB_idx * one_block
                        + one_block,
                        conditionA_idx * one_block : conditionA_idx * one_block
                        + one_block,
                        subj_idx,
                        roi_idx,
                    ]
                    dis_same_cat[comparisons_idx, subj_idx, roi_idx] = np.mean(
                        [
                            comparison_mat[0:4, 0:4],
                            comparison_mat[4:8, 4:8],
                            comparison_mat[8:12, 8:12],
                        ]
                    )
                    dis_diff_cat[comparisons_idx, subj_idx, roi_idx] = np.mean(
                        [
                            comparison_mat[4:8, 0:4],
                            comparison_mat[8:12, 0:4],
                            comparison_mat[8:12, 4:8],
                            comparison_mat[0:4, 4:8],
                            comparison_mat[0:4, 8:12],
                            comparison_mat[4:8, 8:12],
                        ]
                    )
                # Do actual stats
                all_dis_same_cat = dis_same_cat[comparisons_idx, :, roi_idx]
                temp_err_same = sem(all_dis_same_cat, axis=0, ddof=1)
                all_dis_diff_cat = dis_diff_cat[comparisons_idx, :, roi_idx]
                temp_err_diff = sem(all_dis_diff_cat, axis=0, ddof=1)
                temp_t, temp_p = ttest_rel(all_dis_same_cat, all_dis_diff_cat)
                temp_t_wil, temp_p_wil = wilcoxon(all_dis_same_cat, all_dis_diff_cat)

                # Store data
                period_str = "cue" if roi_idx == 0 else "search"
                if "RDM_stats" in locals():
                    RDM_stats = pd.concat(
                        [
                            RDM_stats,
                            pd.DataFrame(
                                {
                                    "condition": f"{subset[0]} vs. {subset[1]}",
                                    "period": period_str,
                                    "area": mask_name,
                                    "mean_same": [np.mean(all_dis_same_cat)],
                                    "mean_diff": [np.mean(all_dis_diff_cat)],
                                    "same_norm": shapiro(all_dis_same_cat)[1],
                                    "diff_norm": shapiro(all_dis_diff_cat)[1],
                                    "t_val": [temp_t],
                                    "p_val": [temp_p],
                                    "t_val_wilcoxon": [temp_t_wil],
                                    "p_val_wilcoxon": [temp_p_wil],
                                    "sem_same": [temp_err_same],
                                    "sem_diff": [temp_err_diff],
                                }
                            ),
                        ]
                    )
                    for sub_idx, sub in enumerate(
                        self.tcs.index.get_level_values("subject").unique()
                    ):
                        RDM_stats_psubj = pd.concat(
                            [
                                RDM_stats_psubj,
                                pd.DataFrame(
                                    {
                                        "condition": f"{subset[0]} vs. {subset[1]}",
                                        "period": period_str,
                                        "area": self.mask_name,
                                        "mean dissimilarity same": all_dis_same_cat[
                                            sub_idx
                                        ],
                                        "mean dissimilarity diff": all_dis_diff_cat[
                                            sub_idx
                                        ],
                                        "subject_nr": [sub],
                                        "sem_same": [temp_err_same],
                                        "sem_diff": [temp_err_diff],
                                    }
                                ),
                            ]
                        )
                else:
                    RDM_stats = pd.DataFrame(
                        {
                            "condition": f"{subset[0]} vs. {subset[1]}",
                            "period": "cue",
                            "area": mask_name,
                            "mean_same": [np.mean(all_dis_same_cat)],
                            "mean_diff": [np.mean(all_dis_diff_cat)],
                            "same_norm": shapiro(all_dis_same_cat)[1],
                            "diff_norm": shapiro(all_dis_diff_cat)[1],
                            "t_val": [temp_t],
                            "p_val": [temp_p],
                            "t_val_wilcoxon": [temp_t_wil],
                            "p_val_wilcoxon": [temp_p_wil],
                            "sem_same": [temp_err_same],
                            "sem_diff": [temp_err_diff],
                        }
                    )
                    for sub_idx, sub in enumerate(
                        self.tcs.index.get_level_values("subject").unique()
                    ):
                        if sub_idx == 0:
                            RDM_stats_psubj = pd.DataFrame(
                                {
                                    "condition": f"{subset[0]} vs. {subset[1]}",
                                    "period": period_str,
                                    "area": self.mask_name,
                                    "mean dissimilarity same": all_dis_same_cat[
                                        sub_idx
                                    ],
                                    "mean dissimilarity diff": all_dis_diff_cat[
                                        sub_idx
                                    ],
                                    "subject_nr": [sub],
                                    "sem_same": [temp_err_same],
                                    "sem_diff": [temp_err_diff],
                                }
                            )
                        else:
                            RDM_stats_psubj = pd.concat(
                                [
                                    RDM_stats_psubj,
                                    pd.DataFrame(
                                        {
                                            "condition": f"{subset[0]} vs. {subset[1]}",
                                            "period": period_str,
                                            "area": self.mask_name,
                                            "mean dissimilarity same": all_dis_same_cat[
                                                sub_idx
                                            ],
                                            "mean dissimilarity diff": all_dis_diff_cat[
                                                sub_idx
                                            ],
                                            "subject_nr": [sub],
                                            "sem_same": [temp_err_same],
                                            "sem_diff": [temp_err_diff],
                                        }
                                    ),
                                ]
                            )

            comparisons_idx += 1

        # Save stats
        if save_data:
            # Individual data
            interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
            path_save = (
                self.path_save
                + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
            )
            if hasattr(self, "paired_conditions"):
                path_save += (
                    f"/{self.remain_conditions[0]}_vs_{self.remain_conditions[1]}/"
                )
            path_save += "/RDM-MDS/"
            # Path(path_save).mkdir(parents=True, exist_ok=True)
            if len(self.remain_conditions) == 3:
                RDM_stats_psubj.to_csv(
                    path_save
                    + f"dropped_{self.drop_condition}/data/{self.decoder}_RDM_subj.csv",
                    index=False,
                )
            else:
                RDM_stats_psubj.to_csv(
                    path_save + f"data/{self.decoder}_RDM_subj.csv", index=False
                )
            # Stats
            interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
            path_stats = (
                self.path_save
                + f"/statistics/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
            )
            if hasattr(self, "paired_conditions"):
                path_stats += (
                    f"/{self.remain_conditions[0]}_vs_{self.remain_conditions[1]}/"
                )
            Path(path_stats).mkdir(parents=True, exist_ok=True)
            if len(self.remain_conditions) == 3:
                RDM_stats.to_csv(
                    path_stats
                    + f"{self.decoder}_RDM_dropped_{self.drop_condition}.csv",
                    index=False,
                )
            else:
                RDM_stats.to_csv(path_stats + f"{self.decoder}_RDM.csv", index=False)

        return RDM_stats, comparisons, dis_same_cat, dis_diff_cat


@dataclass
class MDS(RDM):
    def prepare_MDS(self):
        print("\n\nStarting multidimensional scaling analysis...")
        # No need for normalization here because 0-1 ranked correlations are used.

    def visual_MDS(self, save_figure=False):
        # Some preps to build style and legend
        rois = ["cue", "search"]
        exemplars = [1, 2, 3, 4]
        categories = ["cows", "dressers", "skates"]
        if len(self.remain_conditions) == 2:
            alphas = [1.0, 0.5]
        elif len(self.remain_conditions) == 3:
            alphas = [1.0, 0.625, 0.25]
        elif len(self.remain_conditions) == 4:
            alphas = [1.0, 0.75, 0.5, 0.25]
        colors = ["red", "blue", "green"]
        marker_style = ["o", "s", "X", "d"]
        line_styles_all = ["dotted", "dashed", "solid", "dashdot"]
        line_styles = line_styles_all[: len(self.remain_conditions)]

        # Create style dictionary
        MDS_styles = {
            "condition_name": [],
            "alphas": [],
            "colors": [],
            "marker_styles": [],
            "line_styles": [],
        }
        MDS_styles["condition_name"] = self.axis_labels
        [
            MDS_styles["alphas"].extend((len(exemplars) * len(categories)) * [alpha])
            for alpha in alphas
        ]
        for cond_idx, _ in enumerate(self.remain_conditions):
            [MDS_styles["colors"].extend(len(exemplars) * [color]) for color in colors]
        MDS_styles["marker_styles"] = (
            marker_style * len(categories) * len(self.remain_conditions)
        )
        [
            MDS_styles["line_styles"].extend(
                (len(exemplars) * len(categories)) * [style]
            )
            for style in line_styles
        ]

        embedding = manifold.MDS(
            n_components=2,
            metric=True,
            max_iter=1000,
            n_jobs=-1,
            dissimilarity="precomputed",
        )

        for roi_idx, roi in enumerate(rois):
            # For custom legend
            custom_lines, custom_text = [], []
            lw = 2
            marker_size = 10
            MDS_plot = embedding.fit_transform(self.corr_matrix[:, :, roi_idx])

            fig, ax = plt.subplots(figsize=(9, 9))
            for cond_idx, condition in enumerate(self.remain_conditions):
                for cat_idx, category in enumerate(categories):
                    # Add the first index to connect all dots
                    x_to_plot = np.append(
                        MDS_plot[
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars),
                            0,
                        ],
                        MDS_plot[
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars),
                            0,
                        ][0],
                    )
                    y_to_plot = np.append(
                        MDS_plot[
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars),
                            1,
                        ],
                        MDS_plot[
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars),
                            1,
                        ][0],
                    )
                    for exem_idx in range(len(exemplars) + 1):
                        if exem_idx < len(exemplars):
                            current_marker = MDS_styles["marker_styles"][
                                (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                ) : (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                )
                                + len(exemplars)
                            ][exem_idx]
                        else:
                            current_marker = MDS_styles["marker_styles"][
                                (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                ) : (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                )
                                + len(exemplars)
                            ][0]
                        ax.plot(
                            x_to_plot[exem_idx],
                            y_to_plot[exem_idx],
                            alpha=MDS_styles["alphas"][
                                (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                ) : (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                )
                                + len(exemplars)
                            ][0],
                            color=MDS_styles["colors"][
                                (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                ) : (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                )
                                + len(exemplars)
                            ][0],
                            marker=current_marker,
                            linestyle=MDS_styles["line_styles"][
                                (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                ) : (
                                    cond_idx * (len(exemplars) * len(categories))
                                    + (cat_idx * len(exemplars))
                                )
                                + len(exemplars)
                            ][0],
                            label="x",
                        )
                    ax.plot(
                        x_to_plot,
                        y_to_plot,
                        alpha=MDS_styles["alphas"][
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars)
                        ][0],
                        color=MDS_styles["colors"][
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars)
                        ][0],
                        linestyle=MDS_styles["line_styles"][
                            (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            ) : (
                                cond_idx * (len(exemplars) * len(categories))
                                + (cat_idx * len(exemplars))
                            )
                            + len(exemplars)
                        ][0],
                    )
                    # Create custom legends
                    custom_lines.append(
                        plt.Line2D(
                            [0],
                            [0],
                            alpha=alphas[cond_idx],
                            linestyle=line_styles[cond_idx],
                            color=colors[cat_idx],
                            lw=lw,
                        )
                    )
                    custom_text.append(
                        self.remain_conditions[cond_idx] + " " + categories[cat_idx]
                    )
            ax.legend(custom_lines, custom_text)
            plt.axis("off")
            plt.tight_layout()
            if save_figure:
                interval = str(self.trial_length[0]) + "_" + str(self.trial_length[1])
                path_save = (
                    self.path_save
                    + f"/figures/{self.get_mask}/{self.analysis_type}/{interval}/{self.mask_name}/"
                )
                if hasattr(self, "paired_conditions"):
                    path_save += (
                        f"/{self.remain_conditions[0]}_vs_{self.remain_conditions[1]}/"
                    )
                path_save += "/RDM-MDS/"
                if len(self.remain_conditions) == 3:
                    path_save += f"/dropped_{self.drop_condition}/"
                Path(path_save).mkdir(parents=True, exist_ok=True)
                plt.savefig(path_save + f"MDS_{roi}.pdf")
            # plt.show()


# Basic project parameter
project_name = "NRoST"  # str: project name
base_path = "I:/fMRI/NROST_analysis/"  # str: base path
subjects = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
]  # list of strs: subject numbers
include = [
    "X",
    "Y",
    "Z",
    "RotX",
    "RotY",
    "RotZ",
    "FramewiseDisplacement",
    "aCompCor00",
    "aCompCor01",
    "aCompCor02",
    "aCompCor03",
    "aCompCor04",
    "aCompCor05",
    "Cosine00",
    "Cosine01",
    "Cosine02",
    "Cosine03",
    "Cosine04",
    "Cosine05",
]  # list of strs: BIDS confound factors to include in the model
runs = 8  # int: number of runs
t_r = 0.7  # float: TR in secs
interval = [-1.4, 17.5]  # list of floats: interval of interest in secs relative to cue onset (zero event)
analysis_list = ["category", "status"]  # str: "category" or "status" analysis
get_mask = "anat"  # str: 'func' or 'anat'
mask_list = [
    "LOC",
    "IPS",
    "FEF",
]  # list of strs: underlying anatomical region of mask
decoder = "ridgeCV"  # str: type of decoder
rois_name = ["cue", "search"]  # list of strs: label temporal ROIs
rois = [
    [2.8, 8.4],
    [11.2, 16.8],
]  # list of strs: temporal ROIs in secs relative to cue onset (zero event)
analysis_conditions = [
    "all",
    ["drop", "negative"],
    ["drop", "neutral"],
    ["drop", "positive"],
    ["negative", "neutral"],
    ["negative", "positive"],
    ["neutral", "positive"],
]  # list of strs/str-pairs: pairwise combinations of conditions to be analyzed
rdm_drop = ["drop"]  # keep none last for correct statistics
path_fmriprep = normpath(
    f"{base_path}/derivatives/fmriprep"
)  # str: path to fmriprep output
path_masks = normpath(f"{base_path}/masks/{get_mask}")  # str: path to masks
path_save = normpath(f"{base_path}/results")  # str: path to save results
path_subs = normpath(base_path)  # str: path to subject folders

for mask_name in mask_list:
    for analysis_type_idx, analysis_type in enumerate(analysis_list):
        # * Initialize instance
        myProject = MDS(
            path_fmriprep,
            path_masks,
            path_save,
            path_subs,
            project_name,
            runs,
            t_r,
            interval,
            analysis_type,
            get_mask,
            mask_name,
            decoder,
        )

        if analysis_type_idx == 0:
            # * Get data
            print("\n\n", 50*"=",f"STARTING ANALYSIS of {project_name} project with {mask_name} {get_mask} mask\n")
            myProject.sanity_checks()
            myProject.mask_details(show_figure=False, save_figure=True)
            alldata = myProject.data_get(subjects, include)
            # Save
            myProject.data_store("save", object2save=alldata[0], save_name="data_allsubs", data_content="data")
            myProject.data_store("save", object2save=alldata[1], save_name="design_allsubs", data_content="data")
            myProject.data_store("save", object2save=alldata[2], save_name="confounds_allsubs", data_content="data")

            # * Deconvolution
            print("\n\n", 50 * "=", "DECONVOLUTION\n")
            # Reload
            myProject.data_allsubs = myProject.data_store(
                "load", save_name="data_allsubs", data_content="data"
            )
            myProject.design_allsubs = myProject.data_store(
                "load", save_name="design_allsubs", data_content="data"
            )
            myProject.confounds_allsubs = myProject.data_store(
                "load", save_name="confounds_allsubs", data_content="data"
            )
            # Start
            myProject.prepare_deconvolution()
            myProject.calc_deconvolution(
                basis_set="fourier",
                n_regressors=7,
                solution_type="ols",
                alphas=[0.1, 1.0, 10],
                bayesian=False,
            )
            tcs = myProject.calc_timecourse(
                oversampling=1,
                measurement="psc",
                rois=rois,
                save_data=True,
                save_figure=True,
            )
            # Save
            myProject.data_store("save", object2save=tcs, save_name="deconvolved_timecourse", data_content="data")

        # * Decoding
        # Reload
        myProject.tcs = myProject.data_store("load", save_name="deconvolved_timecourse", data_content="data")
        # Start
        for pair_idx, paired_condition in enumerate(analysis_conditions):
            myProject.create_decoder(add_decoder={"logistic_25": LogisticRegression(
                C=25., penalty='l2', solver='lbfgs', max_iter=2500)})
            if paired_condition == "all":
                if analysis_type == "category":
                    print("\n\n", 25*"*",f"{analysis_type} DECODING: using {paired_condition} condition(s) for {mask_name} {get_mask} mask", 25*"*", "\n\n")
                    conditions, baseline = myProject.prepare_data()
                    myProject.calc_decoding_within(decoder=decoder, n_jobs=-1)
                elif analysis_type == "status":
                    continue
            else:
                print("\n\n", 25*"*",f"{analysis_type} DECODING: using {paired_condition} condition(s) for {mask_name} {get_mask} mask", 25*"*", "\n\n")
                conditions, baseline = myProject.prepare_data(paired_conditions=paired_condition)
                if analysis_type == "category":
                    myProject.calc_decoding_across(decoder=decoder)
                elif analysis_type == "status":
                    myProject.calc_decoding_within(decoder=decoder, n_jobs=-1)
            myProject.calc_errorbars(method="SEM")
            myProject.visual_timeseries(
                rois=rois, figure_size=(8, 6), save_figure=True, save_data=True)
            data = myProject.visual_periods(rois=rois, titles=rois_name, save_figure=True, save_data=True)
            # Statistical testing
            myProject.calc_statistics(data=data)

        # Clean up: FIX IT!
        del myProject.paired_conditions

        # * Representational dissimilarity matrix
        if analysis_type == "category":
            # Reload
            myProject.tcs = myProject.data_store("load", save_name="deconvolved_timecourse", data_content="data")

            # Condition triplets (i.e. individual dropped conditions)
            for _, dropped_cond in enumerate(rdm_drop):
                print("\n\n", 25*"*",f"{analysis_type} RDM: using dropped condition: {dropped_cond} for {mask_name} {get_mask} mask", 25*"*", "\n\n")
                myProject.prepare_RDM(drop_condition=dropped_cond)
                myProject.calc_RDM()
                myProject.visual_RDM_timeseries(save_figure=True, save_data=True)
                corr_matrix, corr_matrix_psubj = myProject.visual_RDM_periods(
                    rois=rois, labels=rois_name, save_figure=True, save_data=True)
                # Do stats
                myProject.calc_RDM_stats(corr_matrix_psubj, shift=-1, rois=rois)
                # Save
                myProject.data_store("save", object2save=corr_matrix, save_name=f"/corr_mat_{dropped_cond}_dropped", data_content="data")
                myProject.data_store("save", object2save=corr_matrix_psubj, save_name=f"/corr_mat_psubj_{dropped_cond}_dropped", data_content="data")

                # * Multidimensional scaling
                myProject.corr_matrix = myProject.data_store("load", save_name=f"corr_mat_{dropped_cond}_dropped", data_content="data")
                myProject.visual_MDS(save_figure=True)

            # Condition pairs
            for _, paired_condition in enumerate(analysis_conditions):
                print("\n\n", 25*"*",f"{analysis_type} RDM: using {paired_condition} condition(s) for {mask_name} {get_mask} mask", 25*"*", "\n\n")
                if paired_condition == "all":
                    myProject.prepare_RDM()
                    save_name_all = "corr_mat"
                    save_name_individual = "corr_mat_psubj"
                else:
                    myProject.prepare_RDM(paired_conditions=paired_condition)
                    save_name_all = f"/corr_mat_{paired_condition[0]}_{paired_condition[1]}"
                    save_name_individual = f"/corr_mat_psubj_{paired_condition[0]}_{paired_condition[1]}"
                myProject.calc_RDM()
                myProject.visual_RDM_timeseries(save_figure=True, save_data=True)
                corr_matrix, corr_matrix_psubj = myProject.visual_RDM_periods(
                    rois=rois, labels=rois_name, save_figure=True, save_data=True)
                # Do stats
                myProject.calc_RDM_stats(corr_matrix_psubj, shift=-1, rois=rois)
                # Save
                myProject.data_store("save", object2save=corr_matrix, save_name=save_name_all, data_content="data")
                myProject.data_store("save", object2save=corr_matrix_psubj, save_name=save_name_individual, data_content="data")

                # * Multidimensional scaling
                myProject.corr_matrix = myProject.data_store("load", object2save=corr_matrix, save_name=save_name_all, data_content="data")
                myProject.visual_MDS(save_figure=True)

            # Clean up
            del myProject.paired_conditions

print("\n--- Analysis finished ---")
