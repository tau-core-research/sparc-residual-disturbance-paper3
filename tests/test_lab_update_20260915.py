import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_lab_scope_and_ownership():
    p=json.loads((ROOT/'data/derived/lab_update_2026_09_15.json').read_text())
    for k in ('physical_source_selected','physical_observer_selected','si_constants_derived','tau_specific_signal','empirical_scores_changed'):
        assert p[k] is False
    assert len(p['ledger'])==5
    for row in p['manuscripts']:
        text=(ROOT/row['source']).read_text()
        assert text.count('% BEGIN LAB UPDATE 20260915')==(1 if row['role'] else 0)
