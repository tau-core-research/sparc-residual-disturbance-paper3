#!/usr/bin/env python3
"""Retrospective projection-contrast audit for the frozen Paper 1--3 sample.

This audit does not search for a best classifier.  It evaluates the same
predeclared feature blocks under three fixed regularization strengths and two
fixed source/observability baselines.  The primary comparison is the declared
projection contrast after the MOND/RAR-common residual block has already been
supplied.  It is not a universal test of projection-sensitive rotation
information or of Tau Core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "studies/sparc_taucore_residual_signal_v01/packet_v01_seed"
PAPER1 = ROOT / "studies/sparc_residual_coherence_test_v01/paper_packet_v06_distance_balanced"
PAPER2 = ROOT / "studies/sparc_residual_disturbance_inference_v01/packet_v01_seed"

SOURCE_SUMMARY = PACKET / "paper3_sparc_baryonic_source_summary_v01.csv"
METRICS = PACKET / "paper3_conditional_rotation_information_metrics_v01.csv"
PREDICTIONS = PACKET / "paper3_conditional_rotation_information_predictions_v01.csv"
VERDICT = PACKET / "paper3_conditional_rotation_information_verdict_v01.json"
PROTOCOL = PACKET / "paper3_conditional_rotation_information_protocol_v01.json"
SOURCE_OVERLAP = PACKET / "paper3_label_source_overlap_audit_v01.csv"

GUARDRAIL = "retrospective_conditional_audit_not_tau_validation_not_independent_replication"
SPARC_DOI = "10.5281/zenodo.16284118"
SPARC_ARCHIVE_MD5 = "e4c8b92766026770ed35e5889064e12b"
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7
CV_SEED = 240826
BOOTSTRAP_SEED = 240827
REGULARIZATION_GRID = (0.03, 0.3, 3.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_table1(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            values[fields[0]] = float(fields[1])
    return values


def derive_baryonic_source_summary(rotmod_dir: Path, table1_path: Path) -> pd.DataFrame:
    residuals = pd.read_csv(PAPER2 / "residual_feature_table.csv")
    hubble_type = parse_table1(table1_path)
    rows: list[dict[str, object]] = []
    for name in residuals["GalaxyName"]:
        path = rotmod_dir / f"{name}_rotmod.dat"
        if not path.exists():
            raise FileNotFoundError(path)
        data = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=("R", "Vobs", "eVobs", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"),
        )
        gas2 = data["Vgas"] * np.abs(data["Vgas"])
        disk2 = UPSILON_DISK * data["Vdisk"] ** 2
        bulge2 = UPSILON_BULGE * data["Vbul"] ** 2
        vbar2 = np.maximum(gas2 + disk2 + bulge2, 1.0e-12)
        vbar = np.sqrt(vbar2)
        q33, q67 = data["R"].quantile((0.33, 0.67)).to_numpy()
        inner = data["R"] <= q33
        outer = data["R"] >= q67
        positive_sb = data.loc[data["SBdisk"] > 0, "SBdisk"]
        rows.append(
            {
                "GalaxyName": name,
                "HubType": hubble_type[name],
                "LogVbarMedian": float(np.log(np.median(vbar))),
                "GasFractionMedian": float(np.median(gas2 / vbar2)),
                "BulgeFractionMedian": float(np.median(bulge2 / vbar2)),
                "LogSBdiskMedian": float(np.log(np.median(positive_sb) + 1.0e-12)),
                "VbarOuterInnerRatio": float(np.median(vbar[outer]) / np.median(vbar[inner])),
                "SBdiskOuterInnerLogRatio": float(
                    np.log(
                        (np.median(data.loc[outer, "SBdisk"]) + 1.0e-12)
                        / (np.median(data.loc[inner, "SBdisk"]) + 1.0e-12)
                    )
                ),
                "NRotmodRows": int(len(data)),
                "RotmodSHA256": file_sha256(path),
                "SourceDOI": SPARC_DOI,
                "SourceArchiveMD5": SPARC_ARCHIVE_MD5,
                "UsesObservedVelocity": False,
                "Guardrail": GUARDRAIL,
            }
        )
    output = pd.DataFrame(rows).sort_values("GalaxyName").reset_index(drop=True)
    output.to_csv(SOURCE_SUMMARY, index=False, float_format="%.12g")
    return output


def load_analysis_frame() -> tuple[pd.DataFrame, np.ndarray]:
    residuals = pd.read_csv(PAPER2 / "residual_feature_table.csv")
    observability = pd.read_csv(PAPER2 / "p09_observability_decomposition_join_v01.csv")
    distance = pd.read_csv(PAPER2 / "distance_resolution_environment_join_v01.csv")
    baryonic = pd.read_csv(SOURCE_SUMMARY)
    frame = (
        residuals.merge(observability[["GalaxyName", "InclinationDeg", "EvidenceType"]], on="GalaxyName")
        .merge(distance, on=("GalaxyName", "Class", "NPoints"))
        .merge(baryonic, on="GalaxyName")
        .sort_values("GalaxyName")
        .reset_index(drop=True)
    )
    for column in ("DistanceMpc", "AngularRadiusProxy_KpcPerMpc", "NPoints", "MeanErrVobsKms"):
        frame[f"Log{column}"] = np.log(frame[column].astype(float).clip(lower=1.0e-9))

    suffixes = (
        "RMS",
        "OuterInnerRatio",
        "RadiusPearson",
        "LowAccelerationMean",
        "OutlierFraction_gt_0p15",
    )
    for suffix in suffixes:
        frame[f"StandardMean_{suffix}"] = 0.5 * (
            frame[f"MOND_{suffix}"] + frame[f"RAR_{suffix}"]
        )
        frame[f"ProjectionContrast_{suffix}"] = (
            frame[f"Projection_{suffix}"] - frame[f"StandardMean_{suffix}"]
        )
    target = (frame["Class"] == "C").astype(int).to_numpy()
    if len(frame) != 45 or tuple(np.bincount(target)) != (17, 28):
        raise RuntimeError("The frozen 45-galaxy A/C packet changed.")
    return frame, target


def fixed_blocks() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    observability = [
        "LogDistanceMpc",
        "DistanceFractionalError",
        "InclinationDeg",
        "InclinationErrorDeg",
        "LogNPoints",
        "LogMeanErrVobsKms",
        "LogAngularRadiusProxy_KpcPerMpc",
    ]
    baryonic = [
        "HubType",
        "LogVbarMedian",
        "GasFractionMedian",
        "BulgeFractionMedian",
        "LogSBdiskMedian",
        "VbarOuterInnerRatio",
        "SBdiskOuterInnerLogRatio",
    ]
    suffixes = [
        "RMS",
        "OuterInnerRatio",
        "RadiusPearson",
        "LowAccelerationMean",
        "OutlierFraction_gt_0p15",
    ]
    baselines = {
        "O7_observability": observability,
        "O7B7_observability_plus_baryonic": observability + baryonic,
    }
    blocks = {
        "projection_block": [f"Projection_{suffix}" for suffix in suffixes],
        "mond_block": [f"MOND_{suffix}" for suffix in suffixes],
        "rar_block": [f"RAR_{suffix}" for suffix in suffixes],
        "standard_mean_block": [f"StandardMean_{suffix}" for suffix in suffixes],
        "projection_contrast_only": [f"ProjectionContrast_{suffix}" for suffix in suffixes],
    }
    blocks["standard_plus_projection_contrast"] = (
        blocks["standard_mean_block"] + blocks["projection_contrast_only"]
    )
    return baselines, blocks


def cross_fitted_probabilities(
    frame: pd.DataFrame,
    target: np.ndarray,
    columns: list[str],
    regularization_c: float,
) -> np.ndarray:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=CV_SEED)
    predictions = np.zeros((20, len(frame)), dtype=float)
    counts = np.zeros_like(predictions, dtype=int)
    for split_index, (train, test) in enumerate(splitter.split(frame, target)):
        repeat = split_index // 5
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=regularization_c,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=CV_SEED,
                    ),
                ),
            ]
        )
        model.fit(frame.iloc[train][columns], target[train])
        predictions[repeat, test] = model.predict_proba(frame.iloc[test][columns])[:, 1]
        counts[repeat, test] += 1
    if not np.all(counts == 1):
        raise RuntimeError("Repeated-CV prediction coverage is not exactly one per repeat.")
    return predictions.mean(axis=0)


def metric_triplet(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "AUC": float(roc_auc_score(target, probability)),
        "LogLoss": float(log_loss(target, probability)),
        "Brier": float(brier_score_loss(target, probability)),
    }


def paired_bootstrap_delta(
    target: np.ndarray,
    comparator_probability: np.ndarray,
    candidate_probability: np.ndarray,
    n_bootstrap: int = 5000,
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    class_indices = [np.flatnonzero(target == value) for value in (0, 1)]
    epsilon = np.finfo(float).eps
    p0 = np.clip(comparator_probability, epsilon, 1.0 - epsilon)
    p1 = np.clip(candidate_probability, epsilon, 1.0 - epsilon)
    point_log_loss_0 = -(target * np.log(p0) + (1 - target) * np.log(1.0 - p0))
    point_log_loss_1 = -(target * np.log(p1) + (1 - target) * np.log(1.0 - p1))
    point_brier_0 = (target - p0) ** 2
    point_brier_1 = (target - p1) ** 2
    delta_log_loss: list[float] = []
    delta_brier: list[float] = []
    for _ in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        delta_log_loss.append(
            float(np.mean(point_log_loss_0[sample] - point_log_loss_1[sample]))
        )
        delta_brier.append(float(np.mean(point_brier_0[sample] - point_brier_1[sample])))
    return {
        "DeltaLogLossCI025": float(np.quantile(delta_log_loss, 0.025)),
        "DeltaLogLossCI975": float(np.quantile(delta_log_loss, 0.975)),
        "DeltaBrierCI025": float(np.quantile(delta_brier, 0.025)),
        "DeltaBrierCI975": float(np.quantile(delta_brier, 0.975)),
        "BootstrapKind": "paired_resample_of_fixed_cross_fitted_predictions_not_refit_bootstrap",
    }


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame, target = load_analysis_frame()
    baselines, blocks = fixed_blocks()
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str, float], np.ndarray] = {}

    for c_value in REGULARIZATION_GRID:
        for baseline_name, baseline_columns in baselines.items():
            specs = {
                "baseline": baseline_columns,
                **{name: baseline_columns + columns for name, columns in blocks.items()},
            }
            for model_name, columns in specs.items():
                cache[(baseline_name, model_name, c_value)] = cross_fitted_probabilities(
                    frame, target, columns, c_value
                )

            for model_name in specs:
                candidate = cache[(baseline_name, model_name, c_value)]
                candidate_metrics = metric_triplet(target, candidate)
                comparator_name = (
                    "standard_mean_block"
                    if model_name == "standard_plus_projection_contrast"
                    else "baseline"
                )
                if model_name == "baseline":
                    comparator_metrics = candidate_metrics
                    delta_auc = delta_log_loss = delta_brier = 0.0
                    bootstrap = {
                        "DeltaLogLossCI025": np.nan,
                        "DeltaLogLossCI975": np.nan,
                        "DeltaBrierCI025": np.nan,
                        "DeltaBrierCI975": np.nan,
                        "BootstrapKind": "not_applicable",
                    }
                else:
                    comparator = cache[(baseline_name, comparator_name, c_value)]
                    comparator_metrics = metric_triplet(target, comparator)
                    delta_auc = candidate_metrics["AUC"] - comparator_metrics["AUC"]
                    delta_log_loss = comparator_metrics["LogLoss"] - candidate_metrics["LogLoss"]
                    delta_brier = comparator_metrics["Brier"] - candidate_metrics["Brier"]
                    bootstrap = (
                        paired_bootstrap_delta(target, comparator, candidate)
                        if c_value == 0.3
                        else {
                            "DeltaLogLossCI025": np.nan,
                            "DeltaLogLossCI975": np.nan,
                            "DeltaBrierCI025": np.nan,
                            "DeltaBrierCI975": np.nan,
                            "BootstrapKind": "reported_for_C_0p3_only",
                        }
                    )
                proper_pass = bool(delta_log_loss > 0 and delta_brier > 0) if model_name != "baseline" else False
                rows.append(
                    {
                        "Baseline": baseline_name,
                        "CandidateBlock": model_name,
                        "ComparatorBlock": comparator_name,
                        "RegularizationC": c_value,
                        "N": len(frame),
                        **candidate_metrics,
                        "ComparatorAUC": comparator_metrics["AUC"],
                        "ComparatorLogLoss": comparator_metrics["LogLoss"],
                        "ComparatorBrier": comparator_metrics["Brier"],
                        "DeltaAUC": delta_auc,
                        "DeltaLogLossImprovement": delta_log_loss,
                        "DeltaBrierImprovement": delta_brier,
                        **bootstrap,
                        "DirectionalProperScorePass": proper_pass,
                        "ProjectionContrastTarget": model_name
                        == "standard_plus_projection_contrast",
                        "Interpretation": (
                            "projection_specific_increment_beyond_mond_rar_common_residual"
                            if model_name == "standard_plus_projection_contrast"
                            else "retrospective_conditional_predictive_block_audit"
                        ),
                        "Guardrail": GUARDRAIL,
                    }
                )
                for index, galaxy in enumerate(frame["GalaxyName"]):
                    prediction_rows.append(
                        {
                            "GalaxyName": galaxy,
                            "Class": frame.loc[index, "Class"],
                            "TargetC": int(target[index]),
                            "Baseline": baseline_name,
                            "CandidateBlock": model_name,
                            "RegularizationC": c_value,
                            "CrossFittedProbabilityC": float(candidate[index]),
                            "Guardrail": GUARDRAIL,
                        }
                    )

    metrics = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    contrast_target = metrics[metrics["ProjectionContrastTarget"]]
    generic_projection = metrics[metrics["CandidateBlock"] == "projection_block"]
    primary = metrics[
        (metrics["Baseline"] == "O7B7_observability_plus_baryonic")
        & (metrics["CandidateBlock"] == "standard_plus_projection_contrast")
        & (metrics["RegularizationC"] == 0.3)
    ].iloc[0]
    baseline_primary = metrics[
        (metrics["Baseline"] == "O7B7_observability_plus_baryonic")
        & (metrics["CandidateBlock"] == "baseline")
        & (metrics["RegularizationC"] == 0.3)
    ].iloc[0]
    verdict: dict[str, object] = {
        "status": "RETROSPECTIVE_PROJECTION_CONTRAST_INCREMENT_FAIL_IN_FROZEN_AC_PACKET",
        "n_galaxies": int(len(frame)),
        "class_counts": {"A": int(np.sum(target == 0)), "C": int(np.sum(target == 1))},
        "full_baseline_auc_at_C_0p3": float(baseline_primary["AUC"]),
        "primary_unique_projection_delta_auc": float(primary["DeltaAUC"]),
        "primary_unique_projection_delta_log_loss_improvement": float(
            primary["DeltaLogLossImprovement"]
        ),
        "primary_unique_projection_delta_brier_improvement": float(
            primary["DeltaBrierImprovement"]
        ),
        "projection_contrast_increment_pass_all_baselines_and_regularizations": bool(
            contrast_target["DirectionalProperScorePass"].all()
        ),
        "generic_projection_increment_stable_all_baselines_and_regularizations": bool(
            generic_projection["DirectionalProperScorePass"].all()
        ),
        "permutation_escalation_run": False,
        "permutation_escalation_reason": (
            "The predeclared directional proper-score gate failed; a one-sided positive-increment "
            "permutation claim was therefore not opened."
        ),
        "interpretation": (
            "The frozen residual association remains an unconditional prevalidation result. "
            "In this frozen A/C packet, the declared projection contrast does not supply a "
            "stable positive increment beyond observability, baryonic source summaries, and "
            "the MOND/RAR-common residual block. This is not a test of all projection-sensitive "
            "rotation information or of Tau Core."
        ),
        "remaining_blockers": [
            "A/C labels are not an independent survey-level replication endpoint.",
            "Label evidence and rotation data share a broad H I/kinematic observational ecosystem.",
            "The 45-galaxy sample is too small for a universal conditional-information claim.",
            "A new source-frozen external matched-tracer cohort is required.",
            "The public 17-galaxy Levy KSS is kinematically selected and can serve only as a "
            "conditioned replication or stress packet, not as an independent blind endpoint.",
        ],
        "guardrail": GUARDRAIL,
    }
    return metrics, predictions, verdict


def write_source_overlap_audit() -> None:
    frame = pd.read_csv(PAPER2 / "p09_observability_decomposition_join_v01.csv")
    counts = (
        frame.groupby(["EvidenceType", "Class"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
        .sort_values(["Class", "EvidenceType"])
    )
    counts["IndependentSurveyReplication"] = False
    counts["Interpretation"] = "label_evidence_family_count_not_independent_endpoint"
    counts["Guardrail"] = GUARDRAIL
    counts.to_csv(SOURCE_OVERLAP, index=False)


def write_protocol() -> None:
    baselines, blocks = fixed_blocks()
    protocol = {
        "version": "v01",
        "analysis_kind": "retrospective_fixed_block_conditional_predictive_audit",
        "source_dataset": {
            "doi": SPARC_DOI,
            "rotmod_archive_md5": SPARC_ARCHIVE_MD5,
            "raw_data_redistributed": False,
            "derived_baryonic_summary": SOURCE_SUMMARY.name,
        },
        "sample": "frozen Paper 1--2 A/C sample; N=45 (A=17, C=28)",
        "baselines": baselines,
        "candidate_blocks": blocks,
        "primary_projection_contrast_comparison": (
            "O7B7 + standard_mean_block versus O7B7 + standard_mean_block + "
            "projection_contrast_only"
        ),
        "projection_contrast_definition": "Projection feature minus mean(MOND feature, RAR feature)",
        "model": "median imputation + z scaling + L2 logistic regression",
        "regularization_grid_fixed_not_selected": list(REGULARIZATION_GRID),
        "cross_validation": {
            "scheme": "20x repeated stratified five-fold",
            "seed": CV_SEED,
            "prediction_aggregation": "mean of 20 out-of-fold probabilities per galaxy",
        },
        "proper_score_pass": "DeltaLogLossImprovement > 0 and DeltaBrierImprovement > 0",
        "claim_rule": (
            "A positive claim for the declared projection-contrast increment requires the "
            "comparison to pass under both baselines and all fixed regularization strengths. "
            "Failure is local to this frozen A/C packet and does not establish absence of all "
            "projection-sensitive rotation information."
        ),
        "guardrail": GUARDRAIL,
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotmod-dir", type=Path)
    parser.add_argument("--table1", type=Path)
    args = parser.parse_args()
    if args.rotmod_dir is not None:
        if args.table1 is None:
            parser.error("--table1 is required when --rotmod-dir is supplied")
        derive_baryonic_source_summary(args.rotmod_dir, args.table1)
    if not SOURCE_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_SUMMARY}. Re-run with --rotmod-dir and --table1 to derive it."
        )
    write_protocol()
    write_source_overlap_audit()
    metrics, predictions, verdict = run_audit()
    metrics.to_csv(METRICS, index=False, float_format="%.12g")
    predictions.to_csv(PREDICTIONS, index=False, float_format="%.12g")
    VERDICT.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
