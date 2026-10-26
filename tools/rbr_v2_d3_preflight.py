#!/usr/bin/env python3
"""D3 zero-propagation production assembly and lifecycle signature checks."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,inspect
from tools.rbr_v2_d3_execute import ROOT,OUT,contract
from tools.rbr_v2_d2_download import read,write,sha


def preflight(phase):
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.b1_production_preflight import zero_run_guard
 from tools.rbr_v2_d3_production import build,validate,lifecycle
 from tools.tsb_direct_coast_f2_production import lifecycle_f2
 c=read(contract(phase));rows=[];objects=[]
 with zero_run_guard() as guard:
  for spec in c['arms']:
   arm=build(spec,OUT/(phase+'_preflight')/spec['run_id']);objects.append(arm);proof=validate(arm)
   inspect.signature(lifecycle).bind(arm,spec,contract(phase));inspect.signature(lifecycle_f2).bind(arm,spec,contract(phase))
   rows.append(dict(run_id=spec['run_id'],status='PASS',config_sha256=sha(arm.run_root/'resolved_config.json'),clock=proof['clock']))
   print('D3 zero-run preflight',phase,len(rows),'/',len(c['arms']),spec['run_id'],flush=True)
  for name in ['runner','planner','tracker','motion_model','recorder']:
   if len({id(getattr(a,name)) for a in objects})!=len(objects):raise ValueError('Cross-arm mutable object reuse')
 if any(guard.values()):raise ValueError('Preflight invoked scientific execution')
 write(OUT/(phase+'_preflight.json'),dict(status='PASS',rows=rows,guard_calls=guard,contract_sha256=sha(contract(phase)),lifecycle_three_mandatory_arguments_verified=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['ENGINEERING','SCIENCE']);a=p.parse_args();preflight(a.phase)
