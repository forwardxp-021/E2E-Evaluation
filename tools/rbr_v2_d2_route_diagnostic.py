#!/usr/bin/env python3
"""Read-only map localization of the retained incomplete D arm; no propagation or safety requalification."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv,json,math
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha


def main():
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.b1_production_preflight import zero_run_guard
 from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
 from nuplan.common.actor_state.state_representation import Point2D
 from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 b=read(OUT/'D_execution_binding.json');spec=b['arms'][1];root=ROOT/'outputs/stageR/rbr_v2_d2_D'/spec['run_id']
 trace=root/'trace/realized_current_ego.jsonl';control=root/'telemetry/actual_lqr_controller_telemetry.jsonl'
 states=[json.loads(s) for s in trace.read_text().splitlines()];commands=[json.loads(s) for s in control.read_text().splitlines()]
 route_ids=set(map(str,spec['route_roadblock_ids']));rows=[];polygons={}
 with zero_run_guard() as guard:
  api=get_maps_api(str(ROOT.parent/'nuplan/dataset/maps'),'nuplan-maps-v1.0',spec['map_name'])
  for r in states:
   ego=r['current_ego'];pose=ego['rear_axle'];point=Point2D(pose['x'],pose['y']);edges=[]
   for layer in [Layer.LANE,Layer.LANE_CONNECTOR]:
    for edge in api.get_all_map_objects(point,layer):edges.append(dict(id=edge.id,roadblock_id=str(edge.get_roadblock_id()),in_frozen_route=str(edge.get_roadblock_id()) in route_ids))
   native_error=None
   try:build_native_route_reference_v1_2(api,spec['route_roadblock_ids'],ego,.01)
   except ValueError as error:native_error=str(error)
   i=r['iteration_index'];u=commands[i] if i<len(commands) else None
   rows.append(dict(iteration=i,episode_seconds=(ego['time_us']-states[0]['current_ego']['time_us'])*1e-6,x=pose['x'],y=pose['y'],heading_rad=pose['heading'],speed_mps=ego['speed_mps'],
       current_point_lane_count=len(edges),current_point_route_lane_count=sum(e['in_frozen_route'] for e in edges),lane_memberships=edges,
       diagnostic_native_route_error=native_error,executed_request_mps2=u['longitudinal_requested_acceleration'] if u else None,
       executed_steering_rate_request_rad_s=u['steering_request'] if u else None))
  origin=states[0]['current_ego']['rear_axle'];near=api.get_proximal_map_objects(Point2D(origin['x'],origin['y']),30,[Layer.LANE,Layer.LANE_CONNECTOR])
  for layer,edges in near.items():
   for edge in edges:polygons[(str(layer),edge.id)]=(edge.polygon,str(edge.get_roadblock_id()) in route_ids)
 if any(guard.values()):raise RuntimeError('Read-only diagnosis invoked science')
 dest=OUT/'route_diagnostic';dest.mkdir(exist_ok=True)
 cols=[k for k in rows[0] if k!='lane_memberships']
 with (dest/'events.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(11,4.5))
 for polygon,in_route in polygons.values():
  x,y=polygon.exterior.xy;ax[0].fill(np.asarray(x)-origin['x'],np.asarray(y)-origin['y'],alpha=.13,color='green' if in_route else 'gray')
 xx=[r['x']-origin['x'] for r in rows];yy=[r['y']-origin['y'] for r in rows];ax[0].plot(xx,yy,'o-',color='darkred',label='Recorded rear axle')
 for r,x,y in zip(rows,xx,yy):ax[0].annotate(f"{r['episode_seconds']:.1f}s",(x,y),fontsize=8)
 ax[0].set(xlabel='Relative world X (m)',ylabel='Relative world Y (m)',title='Incomplete arm: map lookup at recorded states',xlim=(-3,14),ylim=(-5,7));ax[0].set_aspect('equal');ax[0].legend(fontsize=8)
 ax[1].plot([r['episode_seconds'] for r in rows[:-1]],[r['executed_steering_rate_request_rad_s'] for r in rows[:-1]],'o-',label='Lateral steering-rate request (rad/s)')
 ax[1].plot([r['episode_seconds'] for r in rows[:-1]],[r['executed_request_mps2'] for r in rows[:-1]],'s-',label='Longitudinal request (m/s2)')
 ax[1].set(xlabel='Episode time (s)',title='Seven executed controls; stopped before braking');ax[1].grid(alpha=.25);ax[1].legend(fontsize=8)
 fig.tight_layout();fig.savefig(dest/'recorded_route_failure.png',dpi=160);plt.close(fig)
 write(dest/'diagnostic.json',dict(scope='POST_HOC_READ_ONLY_LOCALIZATION_NOT_OFFICIAL_SAFETY_OR_SCIENTIFIC_REQUALIFICATION',run_id=spec['run_id'],session_id=spec['session_id'],rows=rows,
       completed_model_transitions=len(commands),recorded_states=len(states),first_braking_scheduled_at_seconds=1.1,
       official_safety_result='NOT_PRODUCED_INCOMPLETE_RUN',zero_run_guard=guard,
       source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [trace,control,ROOT/'tools/b1_native_route_infrastructure.py',ROOT/'tools/r1_closed_loop_benchmark_v2_1.py',ROOT/'tools/tsb_direct_coast_f2_tracker.py',ROOT/'tools/rbr_v2_d2_route_diagnostic.py']},
       interpretation='The seven controls all have zero longitudinal request; loss of route membership occurs before either scheduled brake. Large recorded lateral request is a diagnostic fact, not a proved unique cause or permission to alter frozen SDC.'))
 print('Read-only route diagnostic:',len(states),'states,',len(commands),'transitions; final matching route lanes',rows[-1]['current_point_route_lane_count'],flush=True)

if __name__=='__main__':main()
