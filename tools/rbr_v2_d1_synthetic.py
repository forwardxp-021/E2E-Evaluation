#!/usr/bin/env python3
"""Synthetic Gaussian A/A plumbing only. Never reads a real representation file."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
import numpy as np
from tools.rbr_v2_d1_statistics import fit_baseline,pool,conditional_diagnostic


def main():
 rng=np.random.RandomState(2026102601)
 fit={k:fit_baseline(rng.normal(size=(80,d)),['synthetic_fit_'+str(i) for i in range(80)],'D_fit') for k,d in [('RBR64',64),('H30',30)]}
 def make(name,n):
  return pool(['synthetic_'+name+'_'+str(i) for i in range(n)],{k:rng.normal(size=(n,s['width'])) for k,s in fit.items()},role='D_SYNTHETIC_'+name)
 cal,aa,ab=make('cal',80),make('aa',80),make('ab_is_also_null',160)
 result=conditional_diagnostic(cal,aa,ab,fit)
 result.update(scope='DETERMINISTIC_SYNTHETIC_GAUSSIAN_AA_ONLY_NOT_RBR_H30_SCIENTIFIC_PERFORMANCE',
               seed=2026102601,real_data_used=False,power_or_source_interval_validation=False,planted_drift=False,
               synthetic_widths={'RBR64':64,'H30':30},pool_SESSION_counts={'D_fit':80,'cal':80,'AA':80,'AB_null':160},
               warning='Representation names denote tensor widths only; independent Gaussian numbers are not actual learned or handcrafted data.')
 path=Path(__file__).resolve().parents[1]/'outputs/rbr_v2_d1_v1/synthetic_AA_diagnostic.json';path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
