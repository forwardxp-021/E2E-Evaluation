#!/usr/bin/env python3
"""Read-only complete historical-D reference parity census, no model propagation or detector reads."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha


def main():
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.b1_production_preflight import zero_run_guard
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
 from tools.rbr_v2_d3_arc_binding import route_binding
 from tools.tsb_coast_p1r_planner import reference_profile
 from tools.r1_closed_loop_benchmark_v2_1 import _offset_preserving_route_xy,_headings_from_xy
 cfg=read(ROOT/'configs/rbr_v2_d2_v1/interface_contract.json');specs=read(ROOT/'docs/stageR/e2e_scoped_direct_coast_v1/QUALIFICATION_Arm_Specs_v2.json')['arms'];rows=[]
 with zero_run_guard() as guard:
  for spec,entry in zip(specs,cfg['runs'][:40]):
   assert spec['run_id']==entry['run_id'];path=ROOT/entry['root']/'trace/realized_current_ego.jsonl';states=[json.loads(s)['current_ego'] for s in path.read_text().splitlines()]
   api=get_maps_api(str(ROOT.parent/'nuplan/dataset/maps'),'nuplan-maps-v1.0',spec['map_name']);changed=[];max_shift=0.;different=0
   for i,current in enumerate(states):
    _,distance,_=reference_profile(current['speed_mps'],i*100000,spec['arm'],i<11);need=float(distance[-1])+1e-9
    old=build_native_route_reference_v1_2(api,spec['route_roadblock_ids'],current,need);new,audit=route_binding(api,spec['route_roadblock_ids'],current,need)
    x=_offset_preserving_route_xy(old['reference_xy'],old['current_route_arc_m']+distance,current);y=_offset_preserving_route_xy(new['reference_xy'],new['current_route_arc_m']+distance,current)
    if audit['correction_applied']:changed.append(i);max_shift=max(max_shift,audit['arc_shift_m'])
    if not np.array_equal(x,y) or not np.array_equal(_headings_from_xy(x),_headings_from_xy(y)):different+=1
   rows.append(dict(run_id=entry['run_id'],session_id=spec['session_id'],states=len(states),correction_frames=changed,max_arc_shift_m=max_shift,reference_different_frames=different,full_reference_bitwise_equal=different==0,trace_sha256=sha(path)))
   print('Historical D-only reference parity',len(rows),'/40',entry['run_id'],'changed frames',different,flush=True)
 if any(guard.values()):raise RuntimeError('Compatibility replay invoked science')
 write(ROOT/'outputs/rbr_v2_d3_v1/legacy_D20_reference_compatibility.json',dict(rows=rows,all40_arms_bitwise_reference_equal=all(r['full_reference_bitwise_equal'] for r in rows),zero_run_guard=guard,detector_outputs_read=False,model_propagations=0))

if __name__=='__main__':main()
