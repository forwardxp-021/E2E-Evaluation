#!/usr/bin/env python3
"""Read-only frozen D3 delivery hash/identity verification; no simulation, inference or statistic recomputation."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.rbr_v2_d2_download import ROOT,read,sha


def main():
 m=read(ROOT/'docs/stageR/rbr_v2_d3_v1/Manifest_v1.json');path=ROOT/m['provenance_index']
 if sha(path)!=m['provenance_index_sha256']:raise ValueError('Provenance index changed')
 index=read(path);bad=[p for p,v in index['files'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=v['sha256']]
 if bad:raise ValueError('Frozen artifact mismatch '+repr(bad))
 if sha(ROOT/'outputs/rbr_v2_prod_v1/run/epoch30.pt')!=m['checkpoint_sha256'] or sha(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json')!=m['normalizer_sha256']:raise ValueError('Frozen model/normalizer mismatch')
 v=read(ROOT/'outputs/rbr_v2_d3_v1/SCIENCE_analysis_view.json');ids=[a['run_id'] for a in v['attempts']]
 if len(ids)!=80 or len(set(ids))!=80 or v['total_actual_complete_pairs']!=40 or v['conservative_scientific_CPU_budget_charge_seconds']>14400:raise ValueError('Full denominator/resource proof failure')
 for name in ['SCIENCE_ledger.json','SCIENCE_continuation_ledger.json','ENGINEERING_ledger.json']:
  if not read(ROOT/'outputs/rbr_v2_d3_v1'/name)['stopped']:raise ValueError('Scientific process not closed')
 print('D3 delivery verification PASS:',len(index['files']),'hashes;80 first identities;40 full pairs; model and stopped ledgers unchanged; no new execution')

if __name__=='__main__':main()
