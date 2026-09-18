import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_current_state_claim_boundary():
    p=json.loads((ROOT/'data/derived/state_update_20260918.json').read_text())
    assert not p['tau_specific_evidence']
    assert not p['empirical_scores_changed']
    assert len(p['ledger'])==5
    for m in p['manuscripts']:
        text=(ROOT/m['path']).read_text()
        assert text.count('% BEGIN STATE UPDATE 20260918')==1
        assert text.count('% END STATE UPDATE 20260918')==1
