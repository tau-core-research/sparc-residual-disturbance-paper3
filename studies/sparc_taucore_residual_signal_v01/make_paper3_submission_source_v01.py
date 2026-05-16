#!/usr/bin/env python3
"""Generate Paper 3 seed packet, LaTeX source, figures, arXiv ZIP, and PDF."""

from __future__ import annotations

import csv
import json
import math
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

GUARDRAIL = "paper3_seed_candidate_search_not_tau_core_validation"


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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
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
            "ComputationStatus": "blocked_no_pointwise_residual_table",
            "Role": "requested theory comparator",
            "CurrentUse": "not used as numeric endpoint",
            "Blocker": "Need frozen pointwise RMOND prediction/residual table on the same SPARC radii.",
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


def literature_map() -> list[dict[str, object]]:
    return [
        {
            "Theme": "SPARC/RAR baseline",
            "CitationKey": "McGaugh2016RAR",
            "UseInPaper3": "Defines the strongest low-acceleration empirical baseline that Tau Core must distinguish itself from.",
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
            "UseInPaper3": "Constrains observational failure modes before calling residuals Tau Core signal.",
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
            "Claim": "TPG/projection residual structure can be triaged against MOND-simple and RAR-like residuals to identify Tau Core-compatible follow-up candidates.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "allowed",
            "Claim": "Distance, environment, observer geometry, and residual-onset summaries can be inspected as candidate Tau Core weighting channels.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "allowed",
            "Claim": "The current packet defines a reproducible Paper 3 seed, not a final physical proof.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Status": "forbidden",
            "Claim": "The packet proves Tau Core.",
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
            "Detail": "Need frozen pointwise RMOND residual table before RMOND can be used as a numeric comparator.",
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
        "clean_tau_candidate": "clean tau",
        "environment_tau_candidate": "environment",
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


def write_main_tex(shortlist: list[dict[str, object]], stress: list[dict[str, object]]) -> None:
    tex = r"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{float}}
\usepackage{{array}}

\title{{Searching for Tau Core signal candidates in SPARC residual structure}}
\author{{Jozsef Olcsak}}
\date{{2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Paper 1 found that externally reviewed structural disturbance is associated with larger low-acceleration residual scatter in SPARC. Paper 2 reversed the question and showed that fixed residual-shape features can recover those A/C labels better than chance, while remaining explicitly non-unique with respect to MOND-simple and empirical RAR-like baselines. This Paper 3 seed asks a narrower follow-up question: where do TPG/projection residuals, MOND-simple residuals, RAR-like residuals, and measured rotation curves point to candidate residual structure that could plausibly carry the missing observer- and environment-dependent Tau Core weight? The current packet identifies candidate galaxies, radial onset categories, and environment/observability stress channels. It does not validate Tau Core, does not numerically test RMOND, and does not claim gravity-model selection.
\end{{abstract}}

\section{{Purpose and claim boundary}}

The working hypothesis is that the TPG/projection baseline already carries part of the local Tau Core weighting, while the remaining TPG residual may carry missing observer- and environment-dependent weights. This is a candidate-search statement, not a proof. A defensible Paper 3 must therefore separate three layers:
\begin{{enumerate}}
\item an operational residual layer, based on frozen TPG/projection, MOND-simple, and RAR-like residual maps;
\item a systematics layer, based on distance, inclination, point count, H\,I kinematics, beam smearing, and non-circular motions;
\item a theory layer, where a Tau Core interpretation is allowed only if the residual pattern survives the systematics layer and is more specific than ordinary RAR/MOND behavior.
\end{{enumerate}}

The permitted claim is that this packet defines reproducible Tau Core signal candidates. The forbidden claim is that the candidates already prove Tau Core.

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

The relevant external baseline is the SPARC radial acceleration relation \cite{{McGaugh2016RAR,Lelli2016SPARC}} and the broader MOND/RAR literature \cite{{Milgrom1983MOND,Li2018RARFits}}. These works are not optional background; they define the strongest ordinary low-acceleration competitor. If Tau Core only reproduces the same residual ordering as RAR/MOND, it is not yet distinguished.

The systematics literature is equally central. Non-circular motions in H\,I velocity fields \cite{{Trachternach2008THINGS,Oman2019NonCircular}} and beam-smearing/rotation-curve quality effects \cite{{deBlok1997Beam}} can generate residual structure without new physics. The JWST/NIRCam Zone-of-Avoidance result \cite{{NiloCastellon2025ZoA}} is used only as an observer/line-of-sight motivation: foreground obscuration and hidden structure can materially affect what an observer can map. It is not evidence for Tau Core by itself.

\section{{Candidate construction}}

For each galaxy we record TPG/projection RMS, MOND-simple RMS, RAR-like RMS, TPG-specific excess, environment proxy, distance, reconstruction-risk proxy, inclination, and the first radial bin where the TPG/projection absolute residual exceeds 0.15 dex. Candidate classes are assigned by frozen screening rules:
\begin{{itemize}}
\item \textit{{clean tau candidate}}: high TPG residual, high TPG-specific excess, and non-high reconstruction risk;
\item \textit{{environment tau candidate}}: high residual burden with high environment cue;
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

\section{{RMOND and comparator status}}

The requested RMOND comparison is not yet a numeric endpoint in this seed packet. The repository contains theory notes relating TPG and RMOND, but it does not yet contain a frozen pointwise RMOND prediction/residual table on the same SPARC radii. Treating RMOND as tested without that table would weaken the paper. The next technical gate is therefore explicit: generate or import a pointwise RMOND residual map and add it to the same comparator table as TPG/projection, MOND-simple, and RAR-like baselines.

\section{{Interpretation}}

The most useful interpretation is modest. TPG success controls show where the local projection baseline is already adequate. TPG divergence follow-ups show where the residual has structure that might encode missing observer/environment weights, but these are exactly the cases where ordinary systematics can also enter. A candidate becomes interesting only when three facts hold together: the TPG residual diverges in a structured radial way, the divergence is not shared by all low-acceleration baselines, and the object has a plausible Tau Core weighting channel that is not reducible to distance, inclination, beam smearing, or non-circular motion.

\section{{Conclusion}}

This seed opens Paper 3 as a reproducible candidate-search project. It carries forward the discipline learned from Papers 1 and 2: freeze the endpoint, compare against MOND/RAR baselines, name the systematics, and avoid promoting a diagnostic pattern into a physical proof. The current evidence is enough to define targets for Tau Core follow-up, especially TPG-divergence galaxies and TPG-success controls, but it is not enough to claim a detected Tau Core field.

The next paper-grade step is a frozen RMOND residual table plus a held-out environment/line-of-sight validation rule. Only after that can the paper ask whether the residual candidates are truly Tau Core-specific rather than ordinary low-acceleration or H\,I-systematics behavior.

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
    tex = tex.replace("SHORTLIST_ROWS", latex_table_shortlist(shortlist))
    tex = tex.replace("STRESS_ROWS", latex_stress_rows(stress))
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
            "paper3_model_comparator_status.csv",
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
        PACKET / "paper3_model_comparator_status.csv",
        model_comparator_status(),
        ["Comparator", "ComputationStatus", "Role", "CurrentUse", "Blocker", "Guardrail"],
    )
    write_csv(
        PACKET / "paper3_related_literature_map.csv",
        literature_map(),
        ["Theme", "CitationKey", "UseInPaper3", "URL"],
    )
    write_csv(PACKET / "paper3_claim_boundary.csv", claim_boundary(), ["Status", "Claim", "Guardrail"])

    make_figures(candidates, stress)
    write_references()
    write_main_tex(shortlist, stress)
    build_arxiv_zip()
    pdf_status = compile_pdf()
    write_csv(PACKET / "paper3_readiness_table.csv", readiness_table(pdf_status), ["Item", "Status", "Detail", "Guardrail"])
    write_manifest(pdf_status)


if __name__ == "__main__":
    main()
