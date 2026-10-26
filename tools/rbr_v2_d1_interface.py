#!/usr/bin/env python3
"""Frozen historical B-development interface only; no scientific simulation or V/C reads."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_d1_v1';CFG=ROOT/'configs/rbr_v2_d1_v1/contract.json'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()


def write(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def config():
 c=json.loads(CFG.read_text())
 for path,key in [(c['checkpoint'],'checkpoint_sha256'),(c['normalizer'],'normalizer_sha256'),('tools/s1_protocol_schema.py','H30_source_sha256'),('tools/rbr_v2_canonical83.py','canonical83_source_sha256'),('tools/rbr_v2_model.py','model_source_sha256')]:
  if sha(ROOT/path)!=c[key]:raise ValueError('Frozen source/model changed: '+path)
 if c['V_access'] or c['C_access'] or c['new_simulation']:raise PermissionError('Engineering only binding required')
 return c


def extract():
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from nuplan.planning.simulation.simulation_log import SimulationLog
 from nuplan.common.actor_state.state_representation import Point2D
 from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
 from tools.rbr_v2_canonical83 import from_nuplan,nuplan_lane_infos
 from tools.s1_protocol_schema import handcrafted_h
 from tools.b1_production_preflight import zero_run_guard
 c=config();results=[];dest=OUT/'inputs';dest.mkdir(exist_ok=False)
 with zero_run_guard() as calls:
  for entry in c['runs']:
   root=ROOT/entry['root'];execution=json.loads((root/'execution_manifest.json').read_text());sources={}
   for p,h in execution['artifacts'].items():
    if sha(root/p)!=h:raise ValueError('Immutable historical arm changed '+p)
   spec=execution['spec']
   for k in ['session_id','log_id','scenario_token','arm','run_id']:
    if entry[k]!=spec[k]:raise ValueError('Bound source identity mismatch '+k)
   files=list((root/'raw/simulation_log').rglob('*.msgpack.xz'))
   if len(files)!=1:raise ValueError('Exactly one frozen serialized history required')
   log=SimulationLog.load_data(files[0]);samples=log.simulation_history.data
   if len(samples)!=80:raise ValueError('Exactly80 actual states/observations required')
   ego=[s.ego_state for s in samples];obs=[s.observation for s in samples]
   times=np.array([s.time_us for s in ego],np.int64)
   np.testing.assert_array_equal(times-times[0],np.arange(80,dtype=np.int64)*100000)
   clock=json.loads((root/'canonical_clock_mapping.json').read_text())['rows']
   trace=[json.loads(s)['current_ego'] for s in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()]
   states=[];lanes={};connectors={}
   for i,(s,t) in enumerate(zip(ego,trace)):
    if samples[i].iteration.index!=i or t['time_us']!=s.time_us or s.time_us!=clock[i]['simulation_time_us']:raise ValueError('Fair80frame physical identity mismatch')
    np.testing.assert_allclose([s.rear_axle.x,s.rear_axle.y,s.rear_axle.heading,s.dynamic_car_state.speed],[t['rear_axle']['x'],t['rear_axle']['y'],t['rear_axle']['heading'],t['speed_mps']],rtol=0,atol=1e-9)
    states.append(t)
    # Include all map lanes in the 100m assignment search support around each actual center.
    prox=log.scenario.map_api.get_proximal_map_objects(Point2D(s.center.x,s.center.y),120,[Layer.LANE,Layer.LANE_CONNECTOR])
    for layer,target in [(Layer.LANE,lanes),(Layer.LANE_CONNECTOR,connectors)]:
     for lane in prox[layer]:target[str(lane.id)]=lane
   infos=nuplan_lane_infos(lanes.values(),connectors.values())
   sample=from_nuplan(ego,obs,infos,'body');h=handcrafted_h(states)
   if h.shape!=(30,) or not np.isfinite(h).all():raise ValueError('Unmodified H30 finite30 required')
   if sample['x'].shape!=(80,83) or not sample['ego_valid'].all() or not np.isfinite(sample['x']).all():raise ValueError('Canonical83 validity failure')
   # Independent velocity oracle: body speed norm, world finite displacement; inspect center velocities separately from rear-axle H.
   from tools.rbr_v2_canonical83 import rotate
   body=np.array([[s.dynamic_car_state.center_velocity_2d.x,s.dynamic_car_state.center_velocity_2d.y] for s in ego]);head=np.array([s.center.heading for s in ego]);world=np.column_stack([body[:,0]*np.cos(head)-body[:,1]*np.sin(head),body[:,0]*np.sin(head)+body[:,1]*np.cos(head)])
   canonical_world=rotate(sample['x'][:,2:4],-head[0]);np.testing.assert_allclose(canonical_world,world,atol=2e-6,rtol=1e-6)
   p=dest/(entry['run_id']+'.npz');np.savez_compressed(p,**sample,H30=h,absolute_time_us=times)
   provenance={str(files[0].relative_to(ROOT)):sha(files[0]),str((root/'execution_manifest.json').relative_to(ROOT)):sha(root/'execution_manifest.json'),str((root/'canonical_clock_mapping.json').relative_to(ROOT)):sha(root/'canonical_clock_mapping.json'),str((root/'trace/realized_current_ego.jsonl').relative_to(ROOT)):sha(root/'trace/realized_current_ego.jsonl')}
   results.append(dict(**entry,input_path=str(p.relative_to(ROOT)),input_sha256=sha(p),time_origin_us=int(times[0]),frames=80,canonical_step_us=100000,H30_width=30,canonical83_version=sample['version'],frame_pairing='SAME_SERIALIZED_EGO_OBSERVATIONS_CANONICAL80',position_reference='RBR_center_H_frozen_rear_axle_same_vehicle',ego_velocity_frame='BODY_rotated_to_WORLD_then_initial_heading_frame',oracle_max_velocity_difference=float(np.abs(canonical_world-world).max()),valid_cells=int(sample['channel_valid'].sum()),presence_known_cells=int(sample['channel_valid'][:,[8,23,38,53,68]].sum()),neighbor_present_cells=int(sample['x'][:,[8,23,38,53,68]].sum()),map_lanes=len(lanes),map_connectors=len(connectors),source_sha256=provenance))
   print('historical interface',len(results),'/6',entry['run_id'],flush=True)
 if any(calls.values()):raise ValueError('Forbidden propagation/tracking call')
 write(OUT/'input_manifest.json',dict(status='PASS',rows=results,contract_sha256=sha(CFG),zero_run_guard=calls,source_risk=c['source_risk']))


def encode():
 import torch
 from tools.rbr_v2_model import RBRV2,normalize
 from tools.rbr_v2_d1_statistics import fit_baseline
 c=config();m=json.loads((OUT/'input_manifest.json').read_text());device='mps' if torch.backends.mps.is_available() else 'cpu'
 state=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False);model=RBRV2().to(device);model.load_state_dict(state['model']);model.eval()
 if state['committed_optimizer_steps']!=27390 or state['next_epoch']!=31:raise ValueError('Not the frozen completed epoch30 candidate')
 scaler=json.loads((ROOT/c['normalizer']).read_text());xs=[];ms=[];es=[];hs=[];raw=[]
 for row in m['rows']:
  p=ROOT/row['input_path']
  if sha(p)!=row['input_sha256']:raise ValueError('Materialized input changed')
  a=np.load(p);xs.append(normalize(a['x'],a['channel_valid'],scaler['input']));ms.append(a['channel_valid']);es.append(a['ego_valid']);hs.append(a['H30']);raw.append(a['x'])
 def infer():
  with torch.no_grad():return model(torch.as_tensor(np.array(xs),device=device),torch.as_tensor(np.array(ms),device=device),torch.as_tensor(np.array(es),device=device))[0].cpu().numpy()
 z=infer();again=infer();np.testing.assert_array_equal(z,again)
 if z.shape!=(6,64) or not np.isfinite(z).all():raise ValueError('z64 export invalid')
 np.save(OUT/'RBR64.npy',z);np.save(OUT/'H30.npy',np.array(hs));np.save(OUT/'normalized_canonical83.npy',np.array(xs))
 fitidx=[i for i,r in enumerate(m['rows']) if r['arm']=='BASELINE'];ids=[m['rows'][i]['session_id'] for i in fitidx]
 fitted={k:fit_baseline(v[fitidx],ids,'D_fit') for k,v in [('RBR64',z),('H30',np.array(hs))]}
 write(OUT/'D_fit_engineering_scalers.json',dict(status='ENGINEERING_ONLY_THREE_SESSION_NOT_FINAL_CALIBRATION',pipelines=fitted))
 centered=z.astype(float);centered-=centered.mean(0);sing=np.linalg.svd(centered,compute_uv=False);eig=sing**2;prob=eig[eig>1e-20]/eig.sum();erank=float(np.exp(-np.sum(prob*np.log(prob))))
 var=z.var(0);x=np.array(xs);mask=np.array(ms);valid=x[mask];pairdist=[float(np.linalg.norm(z[i]-z[i+1])) for i in range(0,6,2)]
 stats=dict(status='PASS_INTERFACE_NO_OBSERVED_CONSTANT_COLLAPSE',checkpoint_sha256=c['checkpoint_sha256'],normalizer_sha256=c['normalizer_sha256'],device=device,inference_runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),shape=list(z.shape),finite=True,repeated_eval_bitwise_equal=True,per_channel_variance=var.tolist(),variance_min=float(var.min()),variance_max=float(var.max()),centered_numerical_rank=int(np.linalg.matrix_rank(centered)),rank_upper_bound_from_six_rows=5,effective_rank=erank,pair_L2_distances=pairdist,channel_mask_valid_fraction=float(mask.mean()),channel_valid_fraction=np.mean(mask,axis=(0,1)).tolist(),normalized_valid_abs_p99=float(np.quantile(np.abs(valid),.99)),normalized_valid_abs_max=float(np.abs(valid).max()),normalized_fraction_abs_gt5=float((np.abs(valid)>5).mean()),normalized_fraction_abs_gt10=float((np.abs(valid)>10).mean()),no_outlier_removal=True,H30_shape=[6,30],D_fit_baseline_SESSION=len(ids),D_AA_real='NOT_EXECUTED_INSUFFICIENT_40_UNIQUE_SESSION',model_parameters_modified=False,independence_or_generalization_claim=False)
 write(OUT/'representation_diagnostics.json',stats);print(json.dumps({k:v for k,v in stats.items() if not isinstance(v,list)},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['extract','encode']);a=p.parse_args();extract() if a.mode=='extract' else encode()
