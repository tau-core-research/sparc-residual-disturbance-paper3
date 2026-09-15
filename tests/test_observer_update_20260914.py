"""Claim-safe paper synchronization, not a physical validation test."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_observer_update_scope_and_manuscripts():
    p=json.loads((ROOT/'data/derived/observer_update_2026_09_14.json').read_text())
    for key in ('physical_observer_identified','physical_preparation_derived',
                'physical_stable_quantizer_derived','empirical_scores_changed'):
        assert p[key] is False
    for relative in p['sources']:
        text=(ROOT/relative).read_text()
        assert text.count('% BEGIN OBSERVER UPDATE 20260914')==1
        assert text.count('% END OBSERVER UPDATE 20260914')==1
    assert len(p['ledger'])==4
    result=p['source_results']['chdf_joint_ground_observer']
    assert abs(result['local_three_mode_purity']-0.627817314635553)<1e-12
