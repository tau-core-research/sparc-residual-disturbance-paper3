#!/usr/bin/env python3
"""Generate Paper 3 seed packet, LaTeX source, figures, arXiv ZIP, and PDF."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "studies/sparc_taucore_residual_signal_v01/packet_v01_seed"
SOURCE = ROOT / "paper3_submission_source"
SOURCE_FIGURES = SOURCE / "figures"
PUBLIC_FIGURES = ROOT / "figures"
PAPER1 = ROOT / "studies/sparc_residual_coherence_test_v01/paper_packet_v06_distance_balanced"
PAPER2 = ROOT / "studies/sparc_residual_disturbance_inference_v01/packet_v01_seed"
RADIAL = ROOT / "studies/sparc_radial_s_tau_pilot_v01/packet_v01_seed"
ARXIV_ZIP = ROOT / "arxiv_submission_source.zip"
DEFAULT_LOCAL_SPARC_ROTMOD = ROOT.parent / "tau-core/data/sparc/Rotmod_LTG"
LOCAL_SPARC_ROTMOD = Path(
    os.environ.get(
        "PAPER3_SPARC_ROTMOD_DIR",
        str(DEFAULT_LOCAL_SPARC_ROTMOD),
    )
)

GUARDRAIL = "paper3_seed_candidate_search_not_tau_core_validation"
KPC_METERS = 3.0856775814913673e19
KM_PER_SEC_TO_M_PER_SEC = 1000.0
A0_M_S2 = 1.2e-10
ALPHA_TPG = 0.360
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7

FAMILY_LABELS = {
    "fixed_s1": "fixed S=1",
    "galaxy_constant": "galaxy constant",
    "linear_radius": "linear radius",
    "quadratic_radius": "quadratic radius",
    "linear_acceleration": "linear acceleration",
    "quadratic_acceleration": "quadratic acceleration",
    "radius_plus_acceleration": "radius + acceleration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fnum(value: object, digits: int = 9) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def as_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_rms(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.sqrt(sum(value * value for value in values) / len(values))


def solve_least_squares(rows: list[dict[str, float]], feature_keys: list[str]) -> list[float]:
    n = len(feature_keys)
    ata = [[0.0 for _ in range(n)] for _ in range(n)]
    aty = [0.0 for _ in range(n)]
    for row in rows:
        features = [row[key] for key in feature_keys]
        target = row["target_y"]
        for i in range(n):
            aty[i] += features[i] * target
            for j in range(n):
                ata[i][j] += features[i] * features[j]
    return gaussian_solve(ata, aty)


def gaussian_solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    ridge = 1e-9
    for i in range(n):
        aug[i][i] += ridge
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return [0.0 for _ in range(n)]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx = sum(xs2) / len(xs2)
    my = sum(ys2) / len(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys2))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def residual_onset_catalog() -> list[dict[str, object]]:
    point_rows = read_csv(PAPER1 / "taucore_specificity_point_map.csv")
    grouped = group_by(point_rows, "GalaxyName")
    output: list[dict[str, object]] = []
    for name, rows in sorted(grouped.items()):
        above = [row for row in rows if float(row["AbsResidualProjection"]) >= 0.15]
        first = min(above, key=lambda r: float(r["RadiusFraction"])) if above else None
        inner = [
            float(row["AbsResidualProjection"])
            for row in rows
            if row["RadiusBin"] == "inner_R<0.33Rmax"
        ]
        mid = [
            float(row["AbsResidualProjection"])
            for row in rows
            if row["RadiusBin"] == "middle_0.33-0.67Rmax"
        ]
        outer = [
            float(row["AbsResidualProjection"])
            for row in rows
            if row["RadiusBin"] == "outer_R>=0.66Rmax"
        ]
        low_acc = [
            float(row["AbsResidualProjection"])
            for row in rows
            if row["AccelerationBin"] == "aN/a0<0.1"
        ]
        delta_mond = [float(row["ProjectionMinusMONDAbs"]) for row in rows]
        delta_rar = [float(row["ProjectionMinusRARAbs"]) for row in rows]
        onset_bin = first["RadiusBin"] if first else "no_projection_residual_ge_0p15"
        output.append(
            {
                "GalaxyName": name,
                "Class": rows[0]["Class"],
                "NPoints": len(rows),
                "FirstRadiusFraction_ge_0p15": fnum(first["RadiusFraction"], 6) if first else "",
                "FirstRadiusBin_ge_0p15": onset_bin,
                "MedianInnerProjectionResidual": fnum(median(inner), 9) if inner else "",
                "MedianMidProjectionResidual": fnum(median(mid), 9) if mid else "",
                "MedianOuterProjectionResidual": fnum(median(outer), 9) if outer else "",
                "OuterMinusInnerProjectionResidual": fnum((median(outer) - median(inner)), 9)
                if inner and outer
                else "",
                "MedianLowAccelerationProjectionResidual": fnum(median(low_acc), 9) if low_acc else "",
                "MedianProjectionMinusMONDAbs": fnum(median(delta_mond), 9),
                "MedianProjectionMinusRARAbs": fnum(median(delta_rar), 9),
                "OnsetUse": "residual_shape_triage_not_physical_attribution",
                "Guardrail": GUARDRAIL,
            }
        )
    return output


def signal_candidate_table(onsets: list[dict[str, object]]) -> list[dict[str, object]]:
    features = {row["GalaxyName"]: row for row in read_csv(PAPER2 / "residual_feature_table.csv")}
    distance = {row["GalaxyName"]: row for row in read_csv(PAPER2 / "distance_resolution_environment_join_v01.csv")}
    p09 = {row["GalaxyName"]: row for row in read_csv(PAPER2 / "p09_observability_decomposition_join_v01.csv")}
    onset_map = {str(row["GalaxyName"]): row for row in onsets}

    projection_values = [float(row["Projection_RMS"]) for row in features.values()]
    tpg_excess_values = []
    for row in features.values():
        best_low = min(float(row["MOND_RMS"]), float(row["RAR_RMS"]))
        tpg_excess_values.append(float(row["Projection_RMS"]) - best_low)
    q_projection_hi = percentile(projection_values, 0.75)
    q_projection_lo = percentile(projection_values, 0.25)
    q_excess_hi = percentile(tpg_excess_values, 0.75)
    env_values = [as_float(row["EnvMaxTheta"]) for row in distance.values() if row["EnvMaxTheta"] != ""]
    q_env_hi = percentile(env_values, 0.67)
    risk_median = percentile([as_float(row["ReconstructionRiskChannel_v01"]) for row in distance.values()], 0.5)

    output: list[dict[str, object]] = []
    for name in sorted(features):
        row = features[name]
        dist = distance[name]
        obs = p09[name]
        onset = onset_map[name]
        projection = float(row["Projection_RMS"])
        mond = float(row["MOND_RMS"])
        rar = float(row["RAR_RMS"])
        best_low = min(mond, rar)
        tpg_excess = projection - best_low
        env = as_float(dist["EnvMaxTheta"])
        risk = as_float(dist["ReconstructionRiskChannel_v01"])
        w_abs = as_float(dist["W_tau_eff_abs_v01"])
        mean_err = as_float(dist["MeanErrVobsKms"])
        tau_score = projection + max(tpg_excess, 0.0) + 0.15 * env + 0.10 * w_abs
        env_present = dist["EnvMaxTheta"] != ""
        if projection <= q_projection_lo and tpg_excess <= 0:
            candidate_class = "tpg_success_control"
        elif projection >= q_projection_hi and tpg_excess >= q_excess_hi and risk <= risk_median:
            candidate_class = "clean_tau_candidate"
        elif projection >= q_projection_hi and env_present and env >= q_env_hi:
            candidate_class = "environment_tau_candidate"
        elif row["Class"] == "C" and risk > risk_median:
            candidate_class = "disturbance_systematics_candidate"
        elif tpg_excess > 0:
            candidate_class = "tpg_divergence_followup"
        else:
            candidate_class = "background_control"
        output.append(
            {
                "GalaxyName": name,
                "Class": row["Class"],
                "NPoints": row["NPoints"],
                "ProjectionRMS_TPG": fnum(projection, 9),
                "MONDSimpleRMS": fnum(mond, 9),
                "RARLikeRMS": fnum(rar, 9),
                "TPGMinusBestLowAccelRMS": fnum(tpg_excess, 9),
                "ProjectionMinusMOND_Mean": row["ProjectionMinusMOND_Mean"],
                "ProjectionMinusRAR_Mean": row["ProjectionMinusRAR_Mean"],
                "TauResidualCandidateScore": fnum(tau_score, 9),
                "DistanceMpc": dist["DistanceMpc"],
                "EnvMaxTheta": dist["EnvMaxTheta"],
                "EnvMainDisturber": dist["EnvMainDisturber"],
                "EnvironmentCuePresent": dist["EnvironmentCuePresent"],
                "W_tau_eff_abs_v01": dist["W_tau_eff_abs_v01"],
                "ReconstructionRiskChannel_v01": dist["ReconstructionRiskChannel_v01"],
                "MeanErrVobsKms": fnum(mean_err, 6),
                "InclinationDeg": obs["InclinationDeg"],
                "InclinationErrorDeg": obs["InclinationErrorDeg"],
                "FirstRadiusBin_ge_0p15": onset["FirstRadiusBin_ge_0p15"],
                "FirstRadiusFraction_ge_0p15": onset["FirstRadiusFraction_ge_0p15"],
                "OuterMinusInnerProjectionResidual": onset["OuterMinusInnerProjectionResidual"],
                "CandidateClass": candidate_class,
                "CandidateUse": "triage_not_detection",
                "Guardrail": GUARDRAIL,
            }
        )
    return output


def candidate_shortlist(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    priority = {
        "clean_tau_candidate": 0,
        "environment_tau_candidate": 1,
        "tpg_divergence_followup": 2,
        "disturbance_systematics_candidate": 3,
        "tpg_success_control": 4,
        "background_control": 5,
    }
    selected = [
        row
        for row in rows
        if row["CandidateClass"]
        in {
            "clean_tau_candidate",
            "environment_tau_candidate",
            "tpg_divergence_followup",
            "tpg_success_control",
        }
    ]
    selected.sort(
        key=lambda row: (
            priority[str(row["CandidateClass"])],
            -float(row["TauResidualCandidateScore"]),
        )
    )
    return selected[:15]


def environment_stress(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = {
        "ProjectionRMS_TPG": [float(row["ProjectionRMS_TPG"]) for row in rows],
        "TPGMinusBestLowAccelRMS": [float(row["TPGMinusBestLowAccelRMS"]) for row in rows],
        "TauResidualCandidateScore": [float(row["TauResidualCandidateScore"]) for row in rows],
    }
    covariates = {
        "DistanceMpc": [as_float(row["DistanceMpc"]) for row in rows],
        "EnvMaxTheta": [as_float(row["EnvMaxTheta"]) for row in rows],
        "W_tau_eff_abs_v01": [as_float(row["W_tau_eff_abs_v01"]) for row in rows],
        "ReconstructionRiskChannel_v01": [as_float(row["ReconstructionRiskChannel_v01"]) for row in rows],
        "MeanErrVobsKms": [as_float(row["MeanErrVobsKms"]) for row in rows],
        "InclinationErrorDeg": [as_float(row["InclinationErrorDeg"]) for row in rows],
    }
    output: list[dict[str, object]] = []
    for metric, ys in metrics.items():
        for covariate, xs in covariates.items():
            output.append(
                {
                    "Metric": metric,
                    "Covariate": covariate,
                    "N": len(rows),
                    "Pearson": fnum(pearson(xs, ys), 9),
                    "Interpretation": "screening_correlation_not_causal_attribution",
                    "Guardrail": GUARDRAIL,
                }
            )
    return output


def parse_rotmod(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 8:
            continue
        rows.append(
            {
                "radius_kpc": float(tokens[0]),
                "vobs_kms": max(0.0, float(tokens[1])),
                "err_vobs_kms": max(0.0, float(tokens[2])),
                "vgas_kms": float(tokens[3]),
                "vdisk_kms": max(0.0, float(tokens[4])),
                "vbul_kms": max(0.0, float(tokens[5])),
            }
        )
    return rows


def baryonic_vn2(row: dict[str, float]) -> float:
    gas2 = row["vgas_kms"] * abs(row["vgas_kms"])
    disk2 = UPSILON_DISK * row["vdisk_kms"] * row["vdisk_kms"]
    bulge2 = UPSILON_BULGE * row["vbul_kms"] * row["vbul_kms"]
    return max(0.0, gas2 + disk2 + bulge2)


def s_tau_point_rows_from_raw(candidate_names: set[str]) -> list[dict[str, object]]:
    if not LOCAL_SPARC_ROTMOD.exists():
        existing = PACKET / "paper3_s_tau_required_points.csv"
        if existing.exists():
            return read_csv(existing)  # type: ignore[return-value]
        return []

    output: list[dict[str, object]] = []
    for path in sorted(LOCAL_SPARC_ROTMOD.glob("*_rotmod.dat")):
        name = path.name.split("_rotmod", maxsplit=1)[0]
        if name not in candidate_names:
            continue
        raw_rows = parse_rotmod(path)
        if not raw_rows:
            continue
        rmax = max(row["radius_kpc"] for row in raw_rows)
        for row in raw_rows:
            vn2 = baryonic_vn2(row)
            if vn2 <= 0 or row["radius_kpc"] <= 0 or row["vobs_kms"] <= 0:
                continue
            vn = math.sqrt(vn2)
            radius_m = row["radius_kpc"] * KPC_METERS
            a_n = (vn * KM_PER_SEC_TO_M_PER_SEC) ** 2 / radius_m
            if a_n <= 0:
                continue
            log_kernel = ALPHA_TPG * math.log(1.0 + A0_M_S2 / a_n)
            if abs(log_kernel) < 1e-12:
                continue
            required_s = (row["vobs_kms"] / vn - 1.0) / log_kernel
            factor_fixed = 1.0 + log_kernel
            factor_required = 1.0 + required_s * log_kernel
            if factor_fixed <= 0 or factor_required <= 0:
                continue
            v_fixed = vn * factor_fixed
            output.append(
                {
                    "GalaxyName": name,
                    "RadiusKpc": fnum(row["radius_kpc"], 9),
                    "RadiusFraction": fnum(row["radius_kpc"] / rmax, 9),
                    "VnKms": fnum(vn, 9),
                    "VobsKms": fnum(row["vobs_kms"], 9),
                    "aN_over_a0": fnum(a_n / A0_M_S2, 9),
                    "LogKernelAlphaLn": fnum(log_kernel, 9),
                    "RequiredS_tau": fnum(required_s, 9),
                    "FixedTPGLogResidual": fnum(math.log(row["vobs_kms"] / v_fixed), 9),
                    "RequiredSLogResidual": fnum(math.log(row["vobs_kms"] / (vn * factor_required)), 9),
                    "Source": "local_raw_sparc_rotmod_derived_no_raw_redistribution",
                    "Guardrail": GUARDRAIL,
                }
            )
    return output


def eval_s_tau_family(points: list[dict[str, float]], coeffs: list[float], feature_keys: list[str]) -> float:
    residuals: list[float] = []
    for row in points:
        s_value = sum(coeff * row[key] for coeff, key in zip(coeffs, feature_keys))
        factor = 1.0 + s_value * row["kernel_x"]
        if factor <= 0:
            continue
        pred = row["vn_kms"] * factor
        if pred <= 0:
            continue
        residuals.append(math.log(row["vobs_kms"] / pred))
    return safe_rms(residuals)


def fit_s_tau_diagnostics(point_rows: list[dict[str, object]], candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    class_by_name = {str(row["GalaxyName"]): str(row["Class"]) for row in candidates}
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in point_rows:
        name = str(row["GalaxyName"])
        radius_fraction = as_float(row["RadiusFraction"])
        a_ratio = max(as_float(row["aN_over_a0"]), 1e-12)
        kernel = as_float(row["LogKernelAlphaLn"])
        vn = as_float(row["VnKms"])
        vobs = as_float(row["VobsKms"])
        if kernel <= 0 or vn <= 0 or vobs <= 0:
            continue
        target_y = vobs / vn - 1.0
        grouped.setdefault(name, []).append(
            {
                "radius_fraction": radius_fraction,
                "log_a0_over_aN": math.log10(1.0 / a_ratio),
                "kernel_x": kernel,
                "target_y": target_y,
                "required_s": as_float(row["RequiredS_tau"]),
                "vn_kms": vn,
                "vobs_kms": vobs,
                "const": kernel,
                "radius": kernel * radius_fraction,
                "radius2": kernel * radius_fraction * radius_fraction,
                "accel": kernel * math.log10(1.0 / a_ratio),
                "accel2": kernel * math.log10(1.0 / a_ratio) * math.log10(1.0 / a_ratio),
            }
        )

    summary: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    family_scores: dict[str, list[float]] = {
        "fixed_s1": [],
        "galaxy_constant": [],
        "linear_radius": [],
        "quadratic_radius": [],
        "linear_acceleration": [],
        "quadratic_acceleration": [],
        "radius_plus_acceleration": [],
    }

    family_defs = {
        "galaxy_constant": ["const"],
        "linear_radius": ["const", "radius"],
        "quadratic_radius": ["const", "radius", "radius2"],
        "linear_acceleration": ["const", "accel"],
        "quadratic_acceleration": ["const", "accel", "accel2"],
        "radius_plus_acceleration": ["const", "radius", "accel"],
    }

    for name, rows in sorted(grouped.items()):
        if len(rows) < 4:
            continue
        fixed_rms = eval_s_tau_family(rows, [1.0], ["const"])
        s_values = [row["required_s"] for row in rows]
        s_median = median(s_values)
        s_q25 = percentile(s_values, 0.25)
        s_q75 = percentile(s_values, 0.75)
        s_iqr = s_q75 - s_q25
        out_of_unit = sum(1 for value in s_values if value < 0.0 or value > 1.0) / len(s_values)
        out_of_two = sum(1 for value in s_values if value < 0.0 or value > 2.0) / len(s_values)
        best_family = "fixed_s1"
        best_rms = fixed_rms
        row_out: dict[str, object] = {
            "GalaxyName": name,
            "Class": class_by_name.get(name, ""),
            "NPoints": len(rows),
            "MedianRequiredS_tau": fnum(s_median, 9),
            "IQRRequiredS_tau": fnum(s_iqr, 9),
            "Q25RequiredS_tau": fnum(s_q25, 9),
            "Q75RequiredS_tau": fnum(s_q75, 9),
            "FractionOutside_0_1": fnum(out_of_unit, 9),
            "FractionOutside_0_2": fnum(out_of_two, 9),
            "FixedS1_RMSLog": fnum(fixed_rms, 9),
            "Guardrail": GUARDRAIL,
        }
        family_scores["fixed_s1"].append(fixed_rms)
        for family, keys in family_defs.items():
            coeffs = solve_least_squares(rows, keys)
            rms = eval_s_tau_family(rows, coeffs, keys)
            family_scores[family].append(rms)
            row_out[f"{family}_RMSLog"] = fnum(rms, 9)
            row_out[f"{family}_Coefficients"] = ";".join(f"{coeff:.9f}" for coeff in coeffs)
            if rms < best_rms:
                best_rms = rms
                best_family = family
            long_rows.append(
                {
                    "GalaxyName": name,
                    "Class": class_by_name.get(name, ""),
                    "Family": family,
                    "NPoints": len(rows),
                    "RMSLog": fnum(rms, 9),
                    "ImprovementVsFixedS1": fnum(fixed_rms - rms, 9),
                    "Coefficients": row_out[f"{family}_Coefficients"],
                    "FitUse": "in_sample_shape_diagnostic_not_model_selection",
                    "Guardrail": GUARDRAIL,
                }
            )
        row_out["BestFamily"] = best_family
        row_out["BestFamilyRMSLog"] = fnum(best_rms, 9)
        row_out["BestImprovementVsFixedS1"] = fnum(fixed_rms - best_rms, 9)
        if s_iqr <= 0.25 and out_of_unit <= 0.25:
            verdict = "constant_s_tau_plausible"
        elif row_out.get("linear_radius_RMSLog") and as_float(row_out["linear_radius_RMSLog"]) < fixed_rms * 0.80:
            verdict = "radial_function_preferred"
        elif row_out.get("linear_acceleration_RMSLog") and as_float(row_out["linear_acceleration_RMSLog"]) < fixed_rms * 0.80:
            verdict = "acceleration_function_preferred"
        else:
            verdict = "function_or_systematics_needed"
        row_out["S_tauVerdict"] = verdict
        summary.append(row_out)

    comparison: list[dict[str, object]] = []
    for family, values in family_scores.items():
        if not values:
            continue
        comparison.append(
            {
                "Family": family,
                "NGalaxies": len(values),
                "MedianRMSLog": fnum(median(values), 9),
                "MeanRMSLog": fnum(sum(values) / len(values), 9),
                "Interpretation": "in_sample_required_s_tau_shape_diagnostic",
                "Guardrail": GUARDRAIL,
            }
        )

    return summary, long_rows, comparison


def model_comparator_status() -> list[dict[str, object]]:
    return [
        {
            "Comparator": "TPG/projection",
            "ComputationStatus": "available_pointwise_and_galaxy_level",
            "Role": "primary inherited baseline",
            "CurrentUse": "candidate residual map",
            "Blocker": "",
            "Guardrail": GUARDRAIL,
        },
        {
            "Comparator": "MOND-simple",
            "ComputationStatus": "available_galaxy_level_and_pointwise_abs_residual",
            "Role": "low-acceleration baseline",
            "CurrentUse": "specificity control",
            "Blocker": "",
            "Guardrail": GUARDRAIL,
        },
        {
            "Comparator": "empirical RAR-like",
            "ComputationStatus": "available_galaxy_level_and_pointwise_abs_residual",
            "Role": "SPARC/RAR baseline",
            "CurrentUse": "specificity control",
            "Blocker": "",
            "Guardrail": GUARDRAIL,
        },
        {
            "Comparator": "RMOND",
            "ComputationStatus": "blocked_no_unique_velocity_law",
            "Role": "requested theory comparator",
            "CurrentUse": "bridge audit only",
            "Blocker": "Current local bridge gives Lagrangian-level compatibility and RMOND-MTW architecture notes, but not a unique frozen V(R) prediction independent of TPG.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Comparator": "observed SPARC Vobs",
            "ComputationStatus": "represented_through_frozen_residual_tables",
            "Role": "measurement reference",
            "CurrentUse": "residual endpoint reference",
            "Blocker": "Raw rotmod files are not redistributed in this slim repository.",
            "Guardrail": GUARDRAIL,
        },
    ]


def rmond_bridge_audit() -> list[dict[str, object]]:
    return [
        {
            "Gate": "RMOND_bridge_theory",
            "Finding": "TPG-RMOND bridge documents structural compatibility with RMOND/SZ20 leading scalar and vector exponents.",
            "NumericalEndpointStatus": "not_velocity_law",
            "ImplicationForPaper3": "Useful theory motivation, but cannot be used as a pointwise residual comparator by itself.",
            "NextRequirement": "Freeze a V_RMOND(R) prescription before computing residuals.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Gate": "RMOND_MTW_hybrid_fvec0",
            "Finding": "Local RMOND-MTW analysis states f_vec=0 reproduces pure MTW/TPG exactly.",
            "NumericalEndpointStatus": "degenerate_with_tpg",
            "ImplicationForPaper3": "Not independent evidence; using f_vec=0 as RMOND would double-count TPG.",
            "NextRequirement": "Do not report as separate comparator.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Gate": "RMOND_MTW_hybrid_nonzero_fvec",
            "Finding": "Nonzero vector contribution requires a frozen halo/vector prescription such as f_vec and screening.",
            "NumericalEndpointStatus": "not_frozen_for_public_packet",
            "ImplicationForPaper3": "Could become a comparator only after parameter policy and raw-data regeneration are frozen.",
            "NextRequirement": "Choose no-fit f_vec/screening rule or mark as exploratory grid outside the primary endpoint.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Gate": "finite_T_scalar_bridge",
            "Finding": "Finite-T scalar coefficient is positive, monotonic, saturating, and MOND/RMOND-admissible in shape.",
            "NumericalEndpointStatus": "lagrangian_coefficient_not_speed",
            "ImplicationForPaper3": "Supports theory motivation but cannot be inserted as V_pred without an additional dynamical map.",
            "NextRequirement": "Derive or freeze the acceleration/velocity map from the coefficient.",
            "Guardrail": GUARDRAIL,
        },
    ]


def next_gate_rows() -> list[dict[str, object]]:
    return [
        {
            "Priority": "P0",
            "Gate": "RMOND velocity-law freeze",
            "Action": "Define whether Paper 3 uses no RMOND numeric comparator, f_vec=0 degeneracy documentation, or a predeclared screened-vector RMOND-MTW V(R) law.",
            "PassCondition": "A single formula produces V_RMOND at every SPARC radius without reading target residual outcomes.",
            "FailCondition": "Formula requires post-hoc tuning or remains only Lagrangian-level motivation.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Priority": "P1",
            "Gate": "raw-to-derived regeneration",
            "Action": "If a velocity law is frozen, regenerate a derived pointwise RMOND residual table from local raw SPARC rotmod files, but publish only derived residual summaries.",
            "PassCondition": "Derived table includes galaxy, radius, residual, and source formula hash; raw rotmod files remain excluded.",
            "FailCondition": "Cannot reproduce without private workspace state.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Priority": "P2",
            "Gate": "Tau-specific candidate retest",
            "Action": "Rerun candidate classes after adding RMOND side-by-side with TPG, MOND-simple, and RAR-like baselines.",
            "PassCondition": "Tau candidates retain structured excess not shared by MOND/RAR/RMOND and not explained by observability proxies.",
            "FailCondition": "Candidate excess collapses into RMOND/MOND/RAR or systematics.",
            "Guardrail": GUARDRAIL,
        },
    ]


def literature_map() -> list[dict[str, object]]:
    return [
        {
            "Theme": "SPARC/RAR baseline",
            "CitationKey": "McGaugh2016RAR",
            "UseInPaper3": "Defines the strongest low-acceleration empirical baseline that any projection-sensitive residual interpretation must distinguish itself from.",
            "URL": "https://arxiv.org/abs/1609.05917",
        },
        {
            "Theme": "SPARC mass models",
            "CitationKey": "Lelli2016SPARC",
            "UseInPaper3": "Provides the rotation-curve and baryonic mass-model context inherited by all residual scores.",
            "URL": "https://doi.org/10.3847/0004-6256/152/6/157",
        },
        {
            "Theme": "MOND baseline",
            "CitationKey": "Milgrom1983MOND",
            "UseInPaper3": "Provides the canonical modified-dynamics low-acceleration comparator family.",
            "URL": "https://doi.org/10.1086/161130",
        },
        {
            "Theme": "SPARC individual RAR fits",
            "CitationKey": "Li2018RARFits",
            "UseInPaper3": "Motivates galaxy-to-galaxy variation in low-acceleration fits and warns against overclaiming uniqueness.",
            "URL": "https://arxiv.org/abs/1803.00022",
        },
        {
            "Theme": "HI non-circular motions",
            "CitationKey": "Trachternach2008THINGS",
            "UseInPaper3": "Defines a key systematics channel that can mimic residual structure.",
            "URL": "https://doi.org/10.1088/0004-6256/136/6/2720",
        },
        {
            "Theme": "Dwarf rotation-curve systematics",
            "CitationKey": "Oman2019NonCircular",
            "UseInPaper3": "Shows how non-circular motions can alter inferred rotation-curve diversity.",
            "URL": "https://arxiv.org/abs/1706.07478",
        },
        {
            "Theme": "Beam smearing and LSB rotation curves",
            "CitationKey": "deBlok1997Beam",
            "UseInPaper3": "Constrains observational failure modes before interpreting residuals as projection-sensitive candidates.",
            "URL": "https://arxiv.org/abs/astro-ph/9704274",
        },
        {
            "Theme": "JWST Zone of Avoidance",
            "CitationKey": "NiloCastellon2025ZoA",
            "UseInPaper3": "Supports the observer/line-of-sight theme: hidden foreground structure and extinction can change what the observer can map.",
            "URL": "https://arxiv.org/abs/2510.12488",
        },
    ]


def claim_boundary() -> list[dict[str, object]]:
    return [
        {
            "Status": "allowed",
            "Claim": "TPG/projection residual structure can be triaged against MOND-simple and RAR-like residuals to identify projection-sensitive follow-up candidates.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "allowed",
            "Claim": "Distance, environment, observer geometry, and residual-onset summaries can be inspected as candidate observer/environment-sensitive residual patterns.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "allowed",
            "Claim": "The current packet defines a reproducible Paper 3 seed, not a final validation result.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "forbidden",
            "Claim": "The packet validates a physical theory.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "forbidden",
            "Claim": "The packet shows TPG/projection is uniquely preferred over MOND/RAR.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "forbidden",
            "Claim": "RMOND has been numerically tested in this packet.",
            "Guardrail": GUARDRAIL,
        },
    ]


def readiness_table(pdf_status: str = "not_run") -> list[dict[str, object]]:
    return [
        {
            "Item": "paper3_seed_packet",
            "Status": "ready",
            "Detail": "Derived candidate, onset, comparator, stress, literature, and claim-boundary tables regenerate.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "latex_source",
            "Status": "ready",
            "Detail": "main.tex and references.bib regenerate from the source script.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "arxiv_source_zip",
            "Status": "ready",
            "Detail": "ZIP contains TeX, bibliography, and PDF figures only.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "pdf_compile",
            "Status": pdf_status,
            "Detail": "Tectonic build status for paper3_submission_source/main.pdf.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "rmond_numeric_endpoint",
            "Status": "blocked",
            "Detail": "Need a unique frozen RMOND V(R) law; current bridge is theory-compatible but not an independent pointwise velocity comparator.",
            "Guardrail": GUARDRAIL,
        },
    ]


def make_figures(rows: list[dict[str, object]], stress: list[dict[str, object]]) -> None:
    SOURCE_FIGURES.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )

    classes = {"A": "#2c7fb8", "C": "#d95f0e"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for klass in ["A", "C"]:
        subset = [row for row in rows if row["Class"] == klass]
        ax.scatter(
            [as_float(row["EnvMaxTheta"]) for row in subset],
            [as_float(row["ProjectionRMS_TPG"]) for row in subset],
            s=46,
            alpha=0.82,
            label=f"{klass} class",
            color=classes[klass],
            edgecolor="white",
            linewidth=0.5,
        )
    for row in sorted(rows, key=lambda r: float(r["TauResidualCandidateScore"]), reverse=True)[:5]:
        ax.annotate(str(row["GalaxyName"]), (as_float(row["EnvMaxTheta"]), as_float(row["ProjectionRMS_TPG"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Environment proxy: max theta")
    ax.set_ylabel("TPG/projection RMS residual")
    ax.set_title("Residual burden versus environment proxy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "paper3_environment_residual_scatter")

    sorted_rows = sorted(rows, key=lambda r: float(r["TPGMinusBestLowAccelRMS"]), reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = [str(row["GalaxyName"]) for row in sorted_rows]
    values = [float(row["TPGMinusBestLowAccelRMS"]) for row in sorted_rows]
    colors = [classes[str(row["Class"])] for row in sorted_rows]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.86)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("TPG/projection RMS minus best MOND/RAR-like RMS")
    ax.set_title("Largest TPG-specific residual excess candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "paper3_tpg_specific_excess")

    onset_counts: dict[str, int] = {}
    for row in rows:
        onset_counts[str(row["FirstRadiusBin_ge_0p15"])] = onset_counts.get(str(row["FirstRadiusBin_ge_0p15"]), 0) + 1
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    labels = list(onset_counts)
    values = [onset_counts[label] for label in labels]
    ax.bar(labels, values, color="#4d9221", alpha=0.86)
    ax.set_ylabel("Number of galaxies")
    ax.set_title("Where the TPG residual first exceeds 0.15 dex")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_figure(fig, "paper3_residual_onset_bins")

    stress_rows = [row for row in stress if row["Metric"] == "ProjectionRMS_TPG"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [str(row["Covariate"]) for row in stress_rows]
    values = [float(row["Pearson"]) for row in stress_rows]
    ax.bar(labels, values, color="#756bb1", alpha=0.86)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pearson r")
    ax.set_title("Screening correlations for TPG residual burden")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "paper3_observability_stress")


def make_s_tau_figures(summary: list[dict[str, object]], comparison: list[dict[str, object]]) -> None:
    if not summary or not comparison:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = {"A": "#2c7fb8", "C": "#d95f0e"}
    for klass in ["A", "C"]:
        subset = [row for row in summary if row["Class"] == klass]
        ax.scatter(
            [as_float(row["MedianRequiredS_tau"]) for row in subset],
            [as_float(row["IQRRequiredS_tau"]) for row in subset],
            s=44,
            alpha=0.82,
            label=f"{klass} class",
            color=colors[klass],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"Median required $S_\tau$")
    ax.set_ylabel(r"Within-galaxy IQR of required $S_\tau$")
    ax.set_title(r"Does each galaxy need constant or varying $S_\tau$?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "paper3_required_s_tau_scatter")

    ordered = sorted(comparison, key=lambda row: as_float(row["MedianRMSLog"]))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = [FAMILY_LABELS.get(str(row["Family"]), str(row["Family"]).replace("_", " ")) for row in ordered]
    values = [as_float(row["MedianRMSLog"]) for row in ordered]
    ax.barh(labels[::-1], values[::-1], color="#1b9e77", alpha=0.86)
    ax.set_xlabel("Median in-sample RMS log residual")
    ax.set_title(r"Required $S_\tau$ function-family diagnostics")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "paper3_s_tau_family_comparison")


def rotation_curve_model_rows() -> dict[str, list[dict[str, str]]]:
    sources = {
        "DDO126": PACKET / "paper3_tau_signal_ddo126_scoring_pilot_points_v01.csv",
        "DDO50": PACKET / "paper3_tau_signal_priority_ddo50_scoring_pilot_points_v01.csv",
    }
    return {galaxy: read_csv(path) for galaxy, path in sources.items() if path.exists()}


def make_rotation_curve_figure() -> None:
    model_rows = rotation_curve_model_rows()
    if not {"DDO126", "DDO50"} <= set(model_rows):
        return

    model_styles = {
        "NewtonianBaryonic": ("Newtonian baryonic", "#6b7280", "--"),
        "MONDSimpleMu": ("MOND simple-$\\mu$", "#2563eb", "-."),
        "EmpiricalRARLike": ("RAR-like", "#0891b2", ":"),
        "FixedTPG_S1": ("fixed TPG", "#b91c1c", "-"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex="col")
    for col, galaxy in enumerate(["DDO126", "DDO50"]):
        grouped: dict[float, dict[str, object]] = {}
        for row in model_rows[galaxy]:
            radius = as_float(row["RadiusKpc"])
            grouped.setdefault(
                radius,
                {
                    "RadiusKpc": radius,
                    "VobsKms": as_float(row["VobsKms"]),
                    "ErrVobsKms": as_float(row["ErrVobsKms"]),
                    "RequiredS_tauDiagnostic": as_float(row["RequiredS_tauDiagnostic"]),
                },
            )
            grouped[radius][str(row["Model"])] = as_float(row["VmodelKms"])

        points = [grouped[key] for key in sorted(grouped)]
        radii = [float(point["RadiusKpc"]) for point in points]
        vobs = [float(point["VobsKms"]) for point in points]
        verr = [float(point["ErrVobsKms"]) for point in points]
        req_s = [float(point["RequiredS_tauDiagnostic"]) for point in points]

        ax_curve = axes[0][col]
        ax_curve.errorbar(
            radii,
            vobs,
            yerr=verr,
            fmt="o",
            ms=3.3,
            lw=0.8,
            color="#111827",
            ecolor="#9ca3af",
            capsize=1.4,
            label="$V_{\\rm obs}$",
            zorder=5,
        )
        for model, (label, color, linestyle) in model_styles.items():
            xs = [float(point["RadiusKpc"]) for point in points if model in point]
            values = [float(point[model]) for point in points if model in point]
            ax_curve.plot(xs, values, linestyle=linestyle, color=color, lw=1.35, label=label)
        ax_curve.set_title(f"{galaxy} rotation curve", fontsize=11)
        ax_curve.set_ylabel("velocity [km s$^{-1}$]")
        ax_curve.grid(True, alpha=0.22)
        if col == 1:
            ax_curve.legend(frameon=False, fontsize=7.6, loc="best")

        ax_diag = axes[1][col]
        fixed_points = [point for point in points if "FixedTPG_S1" in point]
        fixed_x = [float(point["RadiusKpc"]) for point in fixed_points]
        fixed_residual = [
            math.log(float(point["VobsKms"]) / float(point["FixedTPG_S1"]))
            for point in fixed_points
            if float(point["FixedTPG_S1"]) > 0
        ]
        ax_diag.axhline(0.0, color="#111827", lw=0.7, alpha=0.5)
        ax_diag.plot(
            fixed_x[: len(fixed_residual)],
            fixed_residual,
            color="#b91c1c",
            lw=1.2,
            label="fixed TPG log residual",
        )
        ax_diag.set_ylabel("log residual")
        ax_diag.set_xlabel("radius [kpc]")
        ax_diag.grid(True, alpha=0.22)
        twin = ax_diag.twinx()
        twin.plot(radii, req_s, color="#047857", lw=1.1, alpha=0.85, label="required $S_\\tau$")
        twin.set_ylabel("required $S_\\tau$")
        if col == 1:
            lines, labels = ax_diag.get_legend_handles_labels()
            lines2, labels2 = twin.get_legend_handles_labels()
            ax_diag.legend(lines + lines2, labels + labels2, frameon=False, fontsize=7.6, loc="best")

    fig.suptitle("Anchor/control rotation-curve diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "paper3_anchor_control_rotation_curves")


def save_figure(fig: plt.Figure, stem: str) -> None:
    pdf = SOURCE_FIGURES / f"{stem}.pdf"
    svg = PUBLIC_FIGURES / f"{stem}.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)


def latex_table_shortlist(rows: list[dict[str, object]], limit: int = 8) -> str:
    def esc(value: object) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("<", r"$<$").replace(">", r"$>$")

    display_class = {
        "clean_tau_candidate": "clean projection",
        "environment_tau_candidate": "environment linked",
        "tpg_divergence_followup": "TPG divergence",
        "tpg_success_control": "TPG success",
        "disturbance_systematics_candidate": "systematics",
        "background_control": "control",
    }
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"{esc(row['GalaxyName'])} & {esc(row['Class'])} & {as_float(row['ProjectionRMS_TPG']):.3f} & "
            f"{as_float(row['TPGMinusBestLowAccelRMS']):.3f} & {as_float(row['EnvMaxTheta']):.2f} & "
            f"{esc(row['FirstRadiusBin_ge_0p15'])} & {esc(display_class.get(str(row['CandidateClass']), row['CandidateClass']))} \\\\"
        )
    return "\n".join(lines)


def latex_stress_rows(stress: list[dict[str, object]]) -> str:
    def esc(value: object) -> str:
        return str(value).replace("_", r"\_")

    rows = [row for row in stress if row["Metric"] == "ProjectionRMS_TPG"]
    return "\n".join(
        f"{esc(row['Covariate'])} & {float(row['Pearson']):.3f} & {esc(row['Interpretation'])} \\\\"
        for row in rows
    )


def latex_s_tau_comparison_rows(comparison: list[dict[str, object]]) -> str:
    rows = sorted(comparison, key=lambda row: as_float(row["MedianRMSLog"]))
    lines = []
    for row in rows:
        family = FAMILY_LABELS.get(str(row["Family"]), str(row["Family"]).replace("_", " "))
        lines.append(
            f"{family} & {as_float(row['MedianRMSLog']):.3f} & "
            f"{as_float(row['MeanRMSLog']):.3f} & {row['NGalaxies']} \\\\"
        )
    return "\n".join(lines)


def write_references() -> None:
    text = r"""@article{Lelli2016SPARC,
  author = {Lelli, Federico and McGaugh, Stacy S. and Schombert, James M.},
  title = {{SPARC}: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves},
  journal = {The Astronomical Journal},
  volume = {152},
  pages = {157},
  year = {2016},
  doi = {10.3847/0004-6256/152/6/157}
}

@article{McGaugh2016RAR,
  author = {McGaugh, Stacy S. and Lelli, Federico and Schombert, James M.},
  title = {Radial Acceleration Relation in Rotationally Supported Galaxies},
  journal = {Physical Review Letters},
  volume = {117},
  pages = {201101},
  year = {2016},
  doi = {10.1103/PhysRevLett.117.201101},
  eprint = {1609.05917},
  archivePrefix = {arXiv}
}

@article{Milgrom1983MOND,
  author = {Milgrom, M.},
  title = {A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis},
  journal = {The Astrophysical Journal},
  volume = {270},
  pages = {365--370},
  year = {1983},
  doi = {10.1086/161130}
}

@article{Li2018RARFits,
  author = {Li, Pengfei and Lelli, Federico and McGaugh, Stacy S. and Schombert, James M.},
  title = {Fitting the radial acceleration relation to individual {SPARC} galaxies},
  journal = {Astronomy and Astrophysics},
  volume = {615},
  pages = {A3},
  year = {2018},
  doi = {10.1051/0004-6361/201732547},
  eprint = {1803.00022},
  archivePrefix = {arXiv}
}

@article{Trachternach2008THINGS,
  author = {Trachternach, C. and de Blok, W. J. G. and Walter, F. and Brinks, E. and Kennicutt, Jr., R. C.},
  title = {Dynamical Centers and Noncircular Motions in {THINGS} Galaxies: Implications for Dark Matter Halos},
  journal = {The Astronomical Journal},
  volume = {136},
  pages = {2720--2760},
  year = {2008},
  doi = {10.1088/0004-6256/136/6/2720}
}

@article{Oman2019NonCircular,
  author = {Oman, Kyle A. and Marasco, Antonino and Navarro, Julio F. and Frenk, Carlos S. and Schaye, Joop and Benitez-Llambay, Alejandro},
  title = {Non-circular motions and the diversity of dwarf galaxy rotation curves},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume = {482},
  pages = {821--847},
  year = {2019},
  doi = {10.1093/mnras/sty2687},
  eprint = {1706.07478},
  archivePrefix = {arXiv}
}

@article{deBlok1997Beam,
  author = {de Blok, W. J. G. and McGaugh, Stacy S.},
  title = {The dark and visible matter content of low surface brightness disc galaxies},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume = {290},
  pages = {533--552},
  year = {1997},
  doi = {10.1093/mnras/290.3.533},
  eprint = {astro-ph/9704274},
  archivePrefix = {arXiv}
}

@article{NiloCastellon2025ZoA,
  author = {Nilo-Castellon, J. L. and Alonso, M. V. and Baravalle, L. D. and Villalon, C. and Willmer, C. N. A. and Valotto, C. and Soto, M. and Minniti, D. and Sgro, M. A. and others},
  title = {Faint galaxies in the Zone of Avoidance revealed by {JWST}/{NIRCam}},
  journal = {Astronomy and Astrophysics},
  volume = {704},
  pages = {A209},
  year = {2025},
  doi = {10.1051/0004-6361/202556688},
  eprint = {2510.12488},
  archivePrefix = {arXiv}
}
"""
    (SOURCE / "references.bib").write_text(text, encoding="utf-8")


def write_main_tex(
    shortlist: list[dict[str, object]],
    stress: list[dict[str, object]],
    s_tau_comparison: list[dict[str, object]],
) -> None:
    tex = r"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{float}}
\usepackage{{array}}

\title{{A reproducible candidate framework for projection-sensitive residual structure in SPARC rotation curves}}
\author{{Jozsef Olcsak}}
\date{{2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Paper 1 found that externally reviewed structural disturbance is associated with larger low-acceleration residual scatter in SPARC. Paper 2 reversed the question and showed that fixed residual-shape features can recover those A/C labels better than chance, while remaining explicitly non-unique with respect to MOND-simple and empirical RAR-like baselines. This Paper 3 seed asks a narrower follow-up question: where do TPG/projection residuals, MOND-simple residuals, RAR-like residuals, and measured rotation curves point to reproducible projection-sensitive residual candidates? The current packet identifies candidate galaxies, radial onset categories, and environment/observability stress channels. It is a candidate-support methodology and registry paper. It does not validate a physical theory, does not numerically test RMOND, and does not claim gravity-model selection.
\end{{abstract}}

\section{{Purpose and claim boundary}}

The working hypothesis is that the TPG/projection baseline may capture part of a local projection-sensitive residual structure, while the remaining TPG residual may carry missing observer- and environment-dependent structure. In this manuscript, ``TPG/projection'' denotes a frozen operational projection baseline used for residual comparison rather than a validated gravity model; TPG is retained as a historical operational label from earlier work and does not imply a validated gravity theory. This is a candidate-search statement, not a validation result. A defensible Paper 3 must therefore separate three layers:
\begin{{enumerate}}
\item an operational residual layer, based on frozen TPG/projection, MOND-simple, and RAR-like residual maps;
\item a systematics layer, based on distance, inclination, point count, H\,I kinematics, beam smearing, and non-circular motions;
\item an interpretation layer, where any projection-sensitive residual interpretation is allowed only if the residual pattern survives the systematics layer and is more specific than ordinary RAR/MOND behavior.
\end{{enumerate}}

The permitted claim is that this packet defines a reproducible candidate-selection and control framework for projection-sensitive residual structures. The forbidden claim is that the candidates already validate any physical theory or gravity model.

\section{{Data inherited from Papers 1 and 2}}

The packet inherits derived SPARC residual artifacts from the Paper 1 and Paper 2 public packages. Raw SPARC rotmod files and raw H\,I survey products are not redistributed. The retained derived tables include pointwise absolute residuals for TPG/projection, MOND-simple, and empirical RAR-like families; galaxy-level residual features; distance, observability, and environment summaries; and small radial-pilot control summaries.

The main residual quantity is
\[
r_{{m,gi}}=\left|\log V_{{\rm obs},gi}-\log V_{{m,gi}}\right|,
\]
with galaxy-level burden
\[
{\rm RMS}_{{m,g}}=\sqrt{{1\over N_g}\sum_i r_{{m,gi}}^2}.
\]
The TPG-specific excess used for candidate triage is
\[
\Delta_{{\rm TPG-low},g}=
{\rm RMS}_{{\rm TPG},g}-
\min\left({\rm RMS}_{{\rm MOND},g},{\rm RMS}_{{\rm RAR},g}\right).
\]
Positive values mark galaxies where the TPG/projection residual is larger than the best available low-acceleration comparator in this packet. Negative values mark TPG success controls.

\section{{Theory motivation}}

The relevant external baseline is the SPARC radial acceleration relation \cite{{McGaugh2016RAR,Lelli2016SPARC}} and the broader MOND/RAR literature \cite{{Milgrom1983MOND,Li2018RARFits}}. These works are not optional background; they define the strongest ordinary low-acceleration competitor. If a projection-sensitive residual hypothesis only reproduces the same residual ordering as RAR/MOND, it is not yet distinguished.

The systematics literature is equally central. Non-circular motions in H\,I velocity fields \cite{{Trachternach2008THINGS,Oman2019NonCircular}} and beam-smearing/rotation-curve quality effects \cite{{deBlok1997Beam}} can generate residual structure without new physics. The JWST/NIRCam Zone-of-Avoidance result \cite{{NiloCastellon2025ZoA}} is used only as an observer/line-of-sight motivation: foreground obscuration and hidden structure can materially affect what an observer can map. It is not evidence for the projection-sensitive residual interpretation by itself.

\section{{Candidate construction}}

For each galaxy we record TPG/projection RMS, MOND-simple RMS, RAR-like RMS, TPG-specific excess, environment proxy, distance, reconstruction-risk proxy, inclination, and the first radial bin where the TPG/projection absolute residual exceeds 0.15 dex. Candidate classes are assigned by frozen screening rules:
\begin{{itemize}}
\item \textit{{clean projection-sensitive candidate}}: high TPG residual, high TPG-specific excess, and non-high reconstruction risk;
\item \textit{{environment-linked projection candidate}}: high residual burden with high environment cue;
\item \textit{{TPG divergence follow-up}}: positive TPG-specific excess without enough cleanliness for the first two labels;
\item \textit{{TPG success control}}: low residual burden where TPG is no worse than MOND/RAR-like baselines.
\end{{itemize}}

These classes are triage labels. They are not physical classifications.

\begin{{table}}[H]
\centering
\caption{{Candidate shortlist from the regenerated seed packet.}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llrrrp{{0.22\linewidth}}p{{0.18\linewidth}}}}
\toprule
Galaxy & Class & TPG RMS & TPG excess & Env. & Onset & Candidate class\\
\midrule
SHORTLIST_ROWS
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Residual and environment readouts}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/paper3_environment_residual_scatter.pdf}}
\caption{{TPG/projection residual burden versus the inherited environment proxy. Named points are the largest seed-candidate scores.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/paper3_tpg_specific_excess.pdf}}
\caption{{Galaxies with the largest TPG/projection residual excess relative to the better of MOND-simple and RAR-like RMS.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{figures/paper3_residual_onset_bins.pdf}}
\caption{{Radial bin where the TPG/projection absolute residual first exceeds 0.15 dex. This is a morphology-of-failure diagnostic, not a detection statistic.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{Screening correlations for TPG/projection residual burden.}}
\footnotesize
\begin{{tabular}}{{lrl}}
\toprule
Covariate & Pearson $r$ & Interpretation\\
\midrule
STRESS_ROWS
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/paper3_observability_stress.pdf}}
\caption{{Screening correlations with the TPG/projection residual burden. These are not causal estimates.}}
\end{{figure}}

One covariate in this table, W\_tau\_eff\_abs\_v01, is intentionally treated as an internal diagnostic rather than independent evidence. Its correlation with projection residual burden is very high, and the variable is not allowed to support any validation claim until a leakage audit proves that it is not partly defined from the same residual endpoint. In the present manuscript it is therefore a bookkeeping stress term only.

\section{{Endpoint-conditioned $S_\tau$ diagnostic}}

The descriptive projection-sensitive residual form can be written operationally as
\[
F_\tau(a_N,R)=1+S_\tau(R)\,\alpha\ln\left(1+{a_0\over a_N}\right).
\]
For each SPARC point with usable baryonic baseline, the endpoint-conditioned descriptive value is
\[
S_{\tau,\mathrm{req}}(R)=
{V_{\rm obs}(R)/V_N(R)-1\over \alpha\ln\left(1+a_0/a_N(R)\right)}.
\]
This is an inverse residual-absorption diagnostic. It is descriptive and endpoint-conditioned. It uses the measured endpoint and therefore is not a predictive field reconstruction, not independent evidence for a physical theory, and not a validation of the projection ansatz. Its role is only to ask what kind of future predictive rule would be needed: a galaxy-level constant, a radial function, an acceleration function, or an environment-coupled function.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\linewidth]{{figures/paper3_required_s_tau_scatter.pdf}}
\caption{{Median required $S_\tau$ versus within-galaxy spread. A low spread near $S_\tau=1$ would support a constant correction; large spread indicates that a function is needed.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{In-sample required-$S_\tau$ function-family diagnostics. Lower RMS means that family can absorb the measured TPG residual more compactly.}}
\footnotesize
\begin{{tabular}}{{lrrr}}
\toprule
Family & Median RMS & Mean RMS & Galaxies\\
\midrule
S_TAU_COMPARISON_ROWS
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/paper3_s_tau_family_comparison.pdf}}
\caption{{Diagnostic comparison of constant and simple functional $S_\tau$ families. This is not model selection because the measured velocities are used to infer $S_\tau$.}}
\end{{figure}}

\section{{Small-residual controls}}

The same diagnostic can be read from the opposite direction. The lowest-quartile TPG/projection RMS objects form a small-residual control set of 12 galaxies, with median projection RMS $0.099854397$ and median required $S_\tau=1.126924435$. These objects ask why the TPG/projection baseline can remain close to the measured curve, rather than why it fails.

The audit gives a cautious answer. One object is classified as a possible local-capture case, where the local TPG/projection baseline may already absorb the dominant local projection-sensitive structure. Three objects are better described as cases where the broader low-acceleration family tracks the measurement well. The remaining eight small-residual objects still carry unresolved radial or external residual structure. Thus good TPG agreement is useful control context, not validation. These rows must remain paired with the high-residual candidates in any future source-native endpoint test.

\section{{Heuristic residual-pattern decomposition}}

The DDO75/Sextans A stress case motivates a compact heuristic decomposition of the unknown residual pattern. This subsection is not a model-building result and is not used as evidence in the present packet. It is included only to state what a future nonleaky predictor might have to separate. In the working form above, the logarithmic low-acceleration kernel is universal, while the remaining factor $S_\tau(R)$ is allowed to carry source and observer dependence:
\[
S_\tau(R)=S_0
+\beta_{\rm src}C_{\rm src}(R)
+\beta_{\rm path}C_{\rm path}(D)
+\beta_{\rm proj}C_{\rm proj}(B,R_{\max})
+\beta_{\rm drift}C_{\rm drift}(R).
\]
Here $C_{\rm src}$ is source-side radial structure, $C_{\rm path}$ is observer path length or distance, $C_{\rm proj}$ is projection/resolution, and $C_{\rm drift}$ is radial-drift or outer-curve morphology. All four terms are placeholders for future preregistered predictors. None is allowed to use target residuals, required $S_\tau$, or post-hoc endpoint gains as an input.

\section{{RMOND comparator status}}

RMOND is retained only as a roadmap comparator. The current packet does not contain a frozen pointwise $V_{\rm RMOND}(R)$ law, and therefore it does not contain an RMOND residual endpoint. Local bridge notes show theory-level compatibility and also warn about double counting: at $f_{\rm vec}=0$ the hybrid reduces to the TPG/MTW case, while nonzero vector terms require an independently frozen halo/vector prescription. A future version may add RMOND only after a unique velocity law is preregistered and evaluated on the same SPARC radii as TPG/projection, MOND-simple, and RAR-like baselines.

\section{{Post-repair status and next gate}}

After the initial seed construction, one repaired branch, DDO168, was rescored under frozen rules as a ladder-consistency example rather than as a primary endpoint. The repaired DDO168 input layer removes the previous mass-closure caveat and gives a single-object specificity-support result: empirical RAR-like remains the best low-acceleration score, but fixed TPG lies within the frozen tie tolerance of the best low-acceleration baseline, with $\Delta_{\rm TPG-best}=0.001509618$ in RMS-log units. The Newtonian baryonic score is worse than fixed TPG by $0.089064802$, and the inverse required-$S_\tau$ diagnostic remains strongly radial, with Pearson $r=0.927628669$.

This result is useful only in context because it places DDO168 beside visible anchors and controls. DDO126 is a positive fixed-TPG public endpoint, DDO50 is a quiet Newtonian-best control with near-zero required-$S_\tau$, DDO154 is a near-one control/countercontrol, and NGC2366 is a caveated lower-amplitude radial candidate. DDO168 therefore serves as one rung in the candidate ladder, not as a hero object. It is not validation evidence by itself, and it does not select TPG over RAR/MOND as a gravity model.

The next gate is therefore two-track. The paper can now be written as a narrow status or seed paper whose claim is candidate specificity support in context. Any stronger claim requires the already frozen eight-object clean endpoint expansion: DDO154, DDO168, NGC2366, WLM, IC1613, DDO47, DDO50, and DDO126, with auditable component-velocity inputs or frozen reconstructions before running the LOGO TPG/MOND/RAR/radial-$S_\tau$ endpoint.

\section{{Post-external-support status}}

The later external-support branch adds a useful but deliberately limited result. A preregistered external-only screen is directionally positive: the candidate side has larger endpoint burden than the mandatory controls and countercases under the frozen readout. This supports the idea that the residual candidates are not purely random bookkeeping artifacts. However, the same branch remains observability-caveated. Distance, inclination, point count, H\,I resolution, and source visibility remain possible ordinary explanations, and the external branch cannot substitute for accepted baryonic component data.

For this reason, the current Paper 3 claim is bounded as follows. The manuscript may report a coherent candidate-support pattern in residual and external-proxy diagnostics. It may also report that a future external-plus-component readout has been frozen as a nonexecuted contract. It may not claim that a physical theory has been validated, that TPG has been selected over MOND/RAR, or that the missing source-native component endpoint has been satisfied.

The frozen combined contract compares three future predictors only after accepted component inputs pass intake: an external-only predictor, a component-only predictor, and an external-plus-component predictor. Endpoint outputs and diagnostics such as Vobs residuals, FunctionGain, ProjectionRMS\_TPG, and required $S_\tau$ are forbidden as predictor inputs. This keeps the validation route alive without allowing endpoint-driven retuning.

\section{{Anchor/control rotation curves}}

Figure~\ref{{fig:anchor-control-rotation}} shows the two most important current visual endpoints. DDO126 is the positive public-only anchor candidate, and DDO50 is the required control. The panels are diagnostic visualizations of the frozen candidate/control ladder. They are not independent validation and they do not replace the component-input gate.

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.94\linewidth]{{figures/paper3_anchor_control_rotation_curves.pdf}}
\caption{{Anchor/control rotation-curve diagnostics. Top panels compare observed rotation speeds with the frozen Newtonian baryonic, MOND-simple, empirical RAR-like, and fixed-TPG curves. Bottom panels show the fixed-TPG log residual and the inverse required-$S_\tau$ diagnostic. The figure is a visual check on the candidate/control ladder, not a validation endpoint.}}
\label{{fig:anchor-control-rotation}}
\end{{figure}}

\section{{Current candidate-ladder status}}

The later DDO126/DDO50 and held-out readiness audits sharpen the current claim boundary. The manuscript can report a reproducible candidate ladder, not a validation result. DDO126 is the strongest public-only positive anchor candidate in the present packet, while DDO50 is the required control branch. WLM is retained as a geometry/observability stress context. NGC2366 remains caveated radial support and must be reported with DDO154 as its countercontrol. IC1613 and DDO47 remain sensitivity and failure-mode objects until their stellar or component-input blockers are repaired.

This status does not add a new endpoint score. It is a claim-control layer. The input watchlist defines what would be needed for a future upgrade: schema-clean component intake for DDO126/DDO50, or a broader component-confirmed held-out pool. Any new file must pass provenance, column-schema, unit/radius, geometry/systematics, and control-visibility checks before model scores are rerun.

Therefore the current paper-level claim remains deliberately narrow. The result is not a physical validation, not gravity-model selection, and not evidence that the residual pattern is unique to a projection-sensitive interpretation. Additional scoring without new accepted component or stellar inputs is not a claim-upgrade path.

\section{{Predictive validation gate}}

The present manuscript is intentionally weaker than a validation paper. It defines where a projection-sensitive residual signal might be looked for, but it does not yet supply a predictive residual-pattern rule. A validation-oriented version must pass a stricter gate before any discovery language is appropriate.

First, the residual-pattern rule must be frozen before reading the endpoint residuals. The rule may use source-side structure, distance or line-of-sight information, projection/observability quantities, and preregistered radial-shape information, but it may not use Vobs residuals, FunctionGain, ProjectionRMS\_TPG, or the inverse required-$S_\tau$ diagnostic as predictor inputs. Second, the rule must be evaluated on a predeclared clean candidate/control set, including both positive anchors and controls. Third, the result must improve the frozen endpoint relative to TPG/projection, MOND-simple, and empirical RAR-like baselines, not merely reproduce a generic low-acceleration ordering. Fourth, the improvement must survive the ordinary observability nulls: distance, inclination, point count, beam smearing, H\,I resolution, and non-circular motion.

This predictive gate is the main upgrade path from a seed paper to a validation paper. In practical terms, the next claim-raising experiment is: freeze a predictive projection-sensitive residual rule, then test whether it improves the predeclared clean candidate/control set beyond TPG/projection, MOND-simple, and RAR-like baselines. Until that experiment is passed, the correct status is candidate-support methodology rather than physical evidence.

\section{{Interpretation}}

The most useful interpretation is modest. TPG success controls show where the local projection baseline is already adequate. TPG divergence follow-ups show where the residual has structure that might encode missing observer/environment-sensitive information, but these are exactly the cases where ordinary systematics can also enter. A candidate becomes interesting only when three facts hold together: the TPG residual diverges in a structured radial way, the divergence is not shared by all low-acceleration baselines, and the object has a plausible projection-sensitive residual pattern that is not reducible to distance, inclination, beam smearing, or non-circular motion.

\section{{Conclusion}}

This seed opens Paper 3 as a reproducible candidate-search project. It carries forward the discipline learned from Papers 1 and 2: freeze the endpoint, compare against MOND/RAR baselines, name the systematics, and avoid promoting a diagnostic pattern into physical evidence. The repaired DDO168 branch, the later external-support branch, and the DDO126/DDO50 candidate-ladder lock strengthen the candidate-specificity story in context, but the current evidence is still not enough to claim a detected physical field.

The next paper-grade step is the eight-object clean endpoint expansion, with source-native component tables or frozen public reconstructions wherever source-native tables are unavailable. The immediate action is therefore write-or-wait: update the manuscript with candidate-support language, or wait for source-native component input and run the intake gate. A frozen RMOND residual table and a held-out environment/line-of-sight validation rule remain important parallel upgrades. Only after those gates can the paper ask whether the residual candidates are truly projection-sensitive rather than ordinary low-acceleration or H\,I-systematics behavior.

\bibliographystyle{{plain}}
\bibliography{{references}}

\end{{document}}
"""
    tex = tex.replace("{{", "{").replace("}}", "}")
    tex = tex.replace(
        r"r_{m,gi}=\left|\log V_{\rm obs},gi}-\log V_{m,gi}\right|,",
        r"r_{m,gi}=\left|\log V_{{\rm obs},gi}-\log V_{m,gi}\right|,",
    )
    tex = tex.replace(
        r"{\rm RMS}_{m,g}=\sqrt{1\over N_g}\sum_i r_{m,gi}^2}.",
        r"{\rm RMS}_{m,g}=\sqrt{{1\over N_g}\sum_i r_{m,gi}^2}.",
    )
    tex = tex.replace(
        r"\min\left({\rm RMS}_{\rm MOND},g},{\rm RMS}_{\rm RAR},g}\right).",
        r"\min\left({\rm RMS}_{{\rm MOND},g},{\rm RMS}_{{\rm RAR},g}\right).",
    )
    tex = tex.replace(
        r"{\rm RMS}_{\rm TPG},g}-",
        r"{\rm RMS}_{{\rm TPG},g}-",
    )
    tex = tex.replace(
        r"\Delta_{\rm TPG-low},g}=",
        r"\Delta_{{\rm TPG-low},g}=",
    )
    tex = tex.replace(
        r"\section{Required S_tau diagnostic}",
        r"\section{Required $S_\tau$ diagnostic}",
    )
    tex = tex.replace(
        r"S_{\tau,\mathrm{req}(R)=",
        r"S_{\tau,\mathrm{req}}(R)=",
    )
    tex = tex.replace("SHORTLIST_ROWS", latex_table_shortlist(shortlist))
    tex = tex.replace("STRESS_ROWS", latex_stress_rows(stress))
    tex = tex.replace("S_TAU_COMPARISON_ROWS", latex_s_tau_comparison_rows(s_tau_comparison))
    (SOURCE / "main.tex").write_text(tex, encoding="utf-8")


def write_manifest(pdf_status: str) -> None:
    manifest = {
        "packet": "sparc_taucore_residual_signal_v01",
        "version": "paper3_seed_v01",
        "guardrail": GUARDRAIL,
        "generated_files": [
            "paper3_signal_candidate_table.csv",
            "paper3_candidate_shortlist.csv",
            "paper3_residual_onset_catalog.csv",
            "paper3_environment_observability_stress.csv",
            "paper3_s_tau_required_points.csv",
            "paper3_s_tau_required_galaxy_summary.csv",
            "paper3_s_tau_family_fit_long.csv",
            "paper3_s_tau_function_family_comparison.csv",
            "paper3_model_comparator_status.csv",
            "paper3_rmond_bridge_audit.csv",
            "paper3_next_gate.csv",
            "paper3_related_literature_map.csv",
            "paper3_claim_boundary.csv",
            "paper3_readiness_table.csv",
        ],
        "pdf_status": pdf_status,
    }
    (PACKET / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def compile_pdf() -> str:
    if shutil.which("tectonic") is None:
        return "blocked_tectonic_not_installed"
    log = SOURCE / "tectonic_build.log"
    result = subprocess.run(
        ["tectonic", "main.tex"],
        cwd=SOURCE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        return "blocked_compile_failed"
    return "ready"


def build_arxiv_zip() -> None:
    if ARXIV_ZIP.exists():
        ARXIV_ZIP.unlink()
    with ZipFile(ARXIV_ZIP, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(SOURCE / "main.tex", "main.tex")
        zf.write(SOURCE / "references.bib", "references.bib")
        for figure in sorted(SOURCE_FIGURES.glob("*.pdf")):
            zf.write(figure, f"figures/{figure.name}")


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    SOURCE_FIGURES.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir(parents=True, exist_ok=True)

    onsets = residual_onset_catalog()
    candidates = signal_candidate_table(onsets)
    shortlist = candidate_shortlist(candidates)
    stress = environment_stress(candidates)
    s_tau_points = s_tau_point_rows_from_raw({str(row["GalaxyName"]) for row in candidates})
    s_tau_summary, s_tau_long, s_tau_comparison = fit_s_tau_diagnostics(s_tau_points, candidates)

    write_csv(
        PACKET / "paper3_residual_onset_catalog.csv",
        onsets,
        [
            "GalaxyName",
            "Class",
            "NPoints",
            "FirstRadiusFraction_ge_0p15",
            "FirstRadiusBin_ge_0p15",
            "MedianInnerProjectionResidual",
            "MedianMidProjectionResidual",
            "MedianOuterProjectionResidual",
            "OuterMinusInnerProjectionResidual",
            "MedianLowAccelerationProjectionResidual",
            "MedianProjectionMinusMONDAbs",
            "MedianProjectionMinusRARAbs",
            "OnsetUse",
            "Guardrail",
        ],
    )
    candidate_fields = [
        "GalaxyName",
        "Class",
        "NPoints",
        "ProjectionRMS_TPG",
        "MONDSimpleRMS",
        "RARLikeRMS",
        "TPGMinusBestLowAccelRMS",
        "ProjectionMinusMOND_Mean",
        "ProjectionMinusRAR_Mean",
        "TauResidualCandidateScore",
        "DistanceMpc",
        "EnvMaxTheta",
        "EnvMainDisturber",
        "EnvironmentCuePresent",
        "W_tau_eff_abs_v01",
        "ReconstructionRiskChannel_v01",
        "MeanErrVobsKms",
        "InclinationDeg",
        "InclinationErrorDeg",
        "FirstRadiusBin_ge_0p15",
        "FirstRadiusFraction_ge_0p15",
        "OuterMinusInnerProjectionResidual",
        "CandidateClass",
        "CandidateUse",
        "Guardrail",
    ]
    write_csv(PACKET / "paper3_signal_candidate_table.csv", candidates, candidate_fields)
    write_csv(PACKET / "paper3_candidate_shortlist.csv", shortlist, candidate_fields)
    write_csv(
        PACKET / "paper3_environment_observability_stress.csv",
        stress,
        ["Metric", "Covariate", "N", "Pearson", "Interpretation", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_s_tau_required_points.csv",
        s_tau_points,
        [
            "GalaxyName",
            "RadiusKpc",
            "RadiusFraction",
            "VnKms",
            "VobsKms",
            "aN_over_a0",
            "LogKernelAlphaLn",
            "RequiredS_tau",
            "FixedTPGLogResidual",
            "RequiredSLogResidual",
            "Source",
            "Guardrail",
        ],
    )
    s_tau_summary_fields = [
        "GalaxyName",
        "Class",
        "NPoints",
        "MedianRequiredS_tau",
        "IQRRequiredS_tau",
        "Q25RequiredS_tau",
        "Q75RequiredS_tau",
        "FractionOutside_0_1",
        "FractionOutside_0_2",
        "FixedS1_RMSLog",
        "galaxy_constant_RMSLog",
        "galaxy_constant_Coefficients",
        "linear_radius_RMSLog",
        "linear_radius_Coefficients",
        "quadratic_radius_RMSLog",
        "quadratic_radius_Coefficients",
        "linear_acceleration_RMSLog",
        "linear_acceleration_Coefficients",
        "quadratic_acceleration_RMSLog",
        "quadratic_acceleration_Coefficients",
        "radius_plus_acceleration_RMSLog",
        "radius_plus_acceleration_Coefficients",
        "BestFamily",
        "BestFamilyRMSLog",
        "BestImprovementVsFixedS1",
        "S_tauVerdict",
        "Guardrail",
    ]
    write_csv(PACKET / "paper3_s_tau_required_galaxy_summary.csv", s_tau_summary, s_tau_summary_fields)
    write_csv(
        PACKET / "paper3_s_tau_family_fit_long.csv",
        s_tau_long,
        ["GalaxyName", "Class", "Family", "NPoints", "RMSLog", "ImprovementVsFixedS1", "Coefficients", "FitUse", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_s_tau_function_family_comparison.csv",
        s_tau_comparison,
        ["Family", "NGalaxies", "MedianRMSLog", "MeanRMSLog", "Interpretation", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_model_comparator_status.csv",
        model_comparator_status(),
        ["Comparator", "ComputationStatus", "Role", "CurrentUse", "Blocker", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_rmond_bridge_audit.csv",
        rmond_bridge_audit(),
        ["Gate", "Finding", "NumericalEndpointStatus", "ImplicationForPaper3", "NextRequirement", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_next_gate.csv",
        next_gate_rows(),
        ["Priority", "Gate", "Action", "PassCondition", "FailCondition", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_related_literature_map.csv",
        literature_map(),
        ["Theme", "CitationKey", "UseInPaper3", "URL"],
    )
    write_csv(PACKET / "paper3_claim_boundary.csv", claim_boundary(), ["Status", "Claim", "Guardrail"])

    make_figures(candidates, stress)
    make_s_tau_figures(s_tau_summary, s_tau_comparison)
    make_rotation_curve_figure()
    write_references()
    write_main_tex(shortlist, stress, s_tau_comparison)
    build_arxiv_zip()
    pdf_status = compile_pdf()
    write_csv(PACKET / "paper3_readiness_table.csv", readiness_table(pdf_status), ["Item", "Status", "Detail", "Guardrail"])
    write_manifest(pdf_status)


if __name__ == "__main__":
    main()
