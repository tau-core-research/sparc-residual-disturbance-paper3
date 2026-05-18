from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "studies/sparc_taucore_residual_signal_v01/packet_v01_seed"
PAPER = ROOT / "paper3_submission_source"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_regeneration_script_runs() -> None:
    subprocess.run(
        ["python", "studies/sparc_taucore_residual_signal_v01/make_paper3_submission_source_v01.py"],
        cwd=ROOT,
        check=True,
    )


def test_core_public_package_files_exist() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "DATA_NOTICE.md",
        ROOT / "requirements.txt",
        ROOT / "arxiv_submission_source.zip",
        PAPER / "main.tex",
        PAPER / "main.pdf",
        PAPER / "references.bib",
        ROOT / "figures/paper3_anchor_control_rotation_curves.svg",
        PAPER / "figures/paper3_anchor_control_rotation_curves.pdf",
        PACKET / "packet_manifest.json",
        PACKET / "paper3_signal_candidate_table.csv",
        PACKET / "paper3_s_tau_function_family_comparison.csv",
        PACKET / "paper3_tau_signal_ddo126_scoring_pilot_points_v01.csv",
        PACKET / "paper3_tau_signal_priority_ddo50_scoring_pilot_points_v01.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert missing == []


def test_core_packet_tables_have_rows_and_guardrails() -> None:
    tables = [
        "paper3_signal_candidate_table.csv",
        "paper3_candidate_shortlist.csv",
        "paper3_residual_onset_catalog.csv",
        "paper3_environment_observability_stress.csv",
        "paper3_s_tau_required_galaxy_summary.csv",
        "paper3_s_tau_function_family_comparison.csv",
        "paper3_model_comparator_status.csv",
        "paper3_claim_boundary.csv",
        "paper3_readiness_table.csv",
    ]
    for table in tables:
        rows = read_csv(PACKET / table)
        assert rows, table
        if "Guardrail" in rows[0]:
            assert any("validation" in row["Guardrail"] or "candidate" in row["Guardrail"] for row in rows)


def test_public_manuscript_uses_current_safe_framing() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "A reproducible candidate framework for projection-sensitive residual structure" in tex
    assert "Predictive validation gate" in tex
    assert "Heuristic residual-pattern decomposition" in tex
    assert "frozen operational projection baseline" in tex
    assert "rather than a validated gravity model" in tex
    forbidden_terms = [
        "Tau Core",
        "proof-gate",
        "Proof-gate",
        "projection-weight",
        "projection weight",
        "residual channel",
    ]
    for term in forbidden_terms:
        assert term not in tex


def test_arxiv_zip_contains_tex_bib_and_figures() -> None:
    zip_path = ROOT / "arxiv_submission_source.zip"
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "main.tex" in names
    assert "references.bib" in names
    figure_names = {name for name in names if name.startswith("figures/") and name.endswith(".pdf")}
    assert "figures/paper3_anchor_control_rotation_curves.pdf" in figure_names
    assert len(figure_names) >= 7
    assert "main.pdf" not in names


def test_no_raw_survey_data_are_bundled() -> None:
    forbidden_dirs = [
        ROOT / "local_data",
        ROOT / "data/sparc/Rotmod_LTG",
    ]
    assert [path for path in forbidden_dirs if path.exists()] == []
