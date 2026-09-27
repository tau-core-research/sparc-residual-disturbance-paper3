import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_pair_readout_scope():
    p=json.loads((ROOT/'data/derived/pair_readout_update_20260927.json').read_text())
    assert not p['empirical_scores_changed'] and not p['tau_specific_evidence']
    assert len(p['pump_widths'])==4
    assert p['finite_window_control']['maximum_integral_identity_error']<1e-10
    for name in p['manuscripts']:
        text=(ROOT/name).read_text()
        assert text.count('% BEGIN PAIR READOUT UPDATE 20260927')==1
        assert text.count('% END PAIR READOUT UPDATE 20260927')==1
