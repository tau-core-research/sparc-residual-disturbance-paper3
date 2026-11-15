import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
def test_pair_readout_scope():
    p=json.loads((ROOT/'data/derived/pair_readout_update_20260927.json').read_text())
    assert not p['empirical_scores_changed'] and not p['tau_specific_evidence']
    assert p['canonical_result']['id']=='CPR1'
    assert p['canonical_result']['version']=='1.0'
    for name in p['manuscripts']:
        text=(ROOT/name).read_text()
        parts=re.findall(r'% BEGIN PAIR READOUT UPDATE 20260927(.*?)% END PAIR READOUT UPDATE 20260927',text,re.S)
        assert len(parts)==1
        part=parts[0]
        if name==p.get('proof_owner_manuscript'):
            assert 'label{thm:cpr1}' in part
            assert all('CPR1('+x+')' in part for x in 'abc')
        else:
            assert 'Theorem CPR1(' in part and 'v1.0 (2026-09-28)' in part
            assert 'full_derivation.pdf#nameddest=cpr1' in part
            assert len(part.split())<170
            assert 'section{' not in part and 'H_{' not in part
