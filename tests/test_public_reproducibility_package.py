from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "studies/sparc_taucore_residual_signal_v01/packet_v01_seed"


def read_csv(name: str) -> list[dict[str, str]]:
    with (PACKET / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_regeneration_script_runs() -> None:
    subprocess.run(
        ["python", "studies/sparc_taucore_residual_signal_v01/make_paper3_submission_source_v01.py"],
        cwd=ROOT,
        check=True,
    )


def test_candidate_packet_has_tau_signal_boundaries() -> None:
    rows = read_csv("paper3_claim_boundary.csv")
    statuses = {row["Status"] for row in rows}
    assert "allowed" in statuses
    assert "forbidden" in statuses
    assert any("not_tau_core_validation" in row["Guardrail"] for row in rows)


def test_model_comparator_status_tracks_rmond_blocker() -> None:
    rows = read_csv("paper3_model_comparator_status.csv")
    rmond = [row for row in rows if row["Comparator"] == "RMOND"]
    assert rmond
    assert rmond[0]["ComputationStatus"] == "blocked_no_pointwise_residual_table"


def test_shortlist_is_reproducible_and_caveated() -> None:
    rows = read_csv("paper3_candidate_shortlist.csv")
    assert rows
    assert all(row["CandidateUse"] == "triage_not_detection" for row in rows)
    assert {"clean_tau_candidate", "tpg_success_control"} & {row["CandidateClass"] for row in rows}


def test_arxiv_zip_contains_tex_sources_only() -> None:
    archive = ROOT / "arxiv_submission_source.zip"
    assert archive.exists()
    with ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "main.tex" in names
    assert "references.bib" in names
    assert any(name.startswith("figures/") and name.endswith(".pdf") for name in names)
    assert "main.pdf" not in names
