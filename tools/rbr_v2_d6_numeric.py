#!/usr/bin/env python3
"""D6 analytic scalar filter/feature error oracles and capacity checks; never propagates a vehicle."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy,hashlib,json,time
import numpy as np
from tools.rbr_v2_d6_static import ROOT,OUT,PLAN,read,write,check,table
from tools.rbr_v2_d1_interface import sha
from tools.rbr_v2_d1_statistics import fit_baseline,transform


def analytic_velocity(v0,a0,u,beta=2/3,dt=.1):
    """Closed finite sum of the published first-order command filter; no model object/state."""
    n=np.arange(80);result=v0+dt*beta*(1-beta**n)/(1-beta)*a0
    for i in range(1,80):result[i]+=dt*np.sum((1-beta**(i-np.arange(i)))*u[:i])
    return result


def mean_error_bound(epsilon,dt=.1):
    # median3 is1-Lipschitz, max absolute row sum of edge_order2 derivative is4/dt.
    return 4*epsilon/dt


def main():
    plan=check();start=time.monotonic();cpu=time.process_time();sources={}
    def bind(p):sources[str(p.relative_to(ROOT))]=sha(p)
    from tools.s1_protocol_schema import handcrafted_h
    from tools.r1_context_mechanism_core import median3
    modelsource=ROOT.parent/'nuplan-devkit/nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py'
    sources[str(modelsource)]=sha(modelsource)
    m=read(ROOT/'outputs/rbr_v2_d3_v1/interface/input_manifest.json')['rows'];h=np.load(ROOT/'outputs/rbr_v2_d3_v1/interface/H30.npy');z=np.load(ROOT/'outputs/rbr_v2_d3_v1/interface/RBR64.npy')
    base=[(i,r) for i,r in enumerate(m) if r['arm']=='BASELINE'];a0=[]
    for i,r in base:
        p=ROOT/r['root']/'telemetry/actual_lqr_controller_telemetry.jsonl';bind(p)
        with p.open() as f:first=json.loads(next(f))
        a0.append(first['model_acceleration_before'])
    beta=2/3;tail=[]
    for second in [5,6]:
        frames=np.arange(second*10,(second+1)*10);coefficient=float(np.mean((beta**frames+beta**(frames+1))/2));values=h[[i for i,r in base],14+second]
        predicted=coefficient*np.asarray(a0);residual=values-predicted
        tail.append(dict(signed_bin_seconds=second,initial_acceleration_coefficient=coefficient,observed_baseline_SD=float(values.std()),predicted_SD_from_a0=float(predicted.std()),residual_SD=float(residual.std()),initial_acceleration_SD=float(np.std(a0)),feature_level_mean=float(values.mean())))
    probes=[];selected=sorted(base,key=lambda ir:hashlib.sha256((plan['numerical']['trace_order_salt']+'|'+ir[1]['session_id']).encode()).hexdigest())[:8]
    for i,r in selected:
        root=ROOT/r['root'];tp=root/'trace/realized_current_ego.jsonl';cp=root/'telemetry/actual_lqr_controller_telemetry.jsonl';bind(tp)
        states=[json.loads(line)['current_ego'] for line in tp.read_text().splitlines()];control=[json.loads(line) for line in cp.read_text().splitlines()]
        original=handcrafted_h(states);np.testing.assert_allclose(original,h[i],atol=1e-13,rtol=0)
        v=np.array([s['speed_mps'] for s in states]);u=np.array([c['longitudinal_requested_acceleration'] for c in control])
        # Official state0 speed is a norm; the model evolves longitudinal vx and zeros vy.
        vx0=control[0]['speed_before'];pred=analytic_velocity(vx0,control[0]['model_acceleration_before'],u)
        err=float(np.max(np.abs(pred[1:]-v[1:])))
        if err>1e-11:raise ValueError('Analytic scalar filter does not match saved speed; no vehicle rerun allowed')
        time_s=(np.array([s['time_us'] for s in states])-states[0]['time_us'])*1e-6
        for label,e in [('one_ULP',float(np.spacing(v).max())),('arithmetic_stress_1e-6',1e-6)]:
            vv=v+e*np.where(np.arange(80)%2,1.,-1.);proxy=np.gradient(median3(v),time_s,edge_order=2);proxy2=np.gradient(median3(vv),time_s,edge_order=2);bound=mean_error_bound(e,dt=float(np.diff(time_s).min()))
            perturb=copy.deepcopy(states)
            for s,value in zip(perturb,vv):s['speed_mps']=float(value)
            altered=handcrafted_h(perturb);actual=float(np.max(np.abs(proxy2-proxy)))
            # Floating roundoff of derivative itself is separate from input perturbation.
            allowance=64*np.finfo(float).eps*max(float(np.abs(v).max()),1)/.1
            if actual>bound+allowance:raise ValueError('Independent derivative error bound violated')
            probes.append(dict(session_id=r['session_id'],probe=label,epsilon_speed=e,initial_norm_minus_longitudinal_speed=float(v[0]-vx0),analytic_speed_max_error_n1to79=err,proxy_max_difference=actual,analytic_proxy_bound=bound,roundoff_allowance=allowance,signed_bin5_delta=float(altered[19]-original[19]),signed_bin6_delta=float(altered[20]-original[20]),measurement_uncertainty_certified=False))
    # RBR numerical stability: all predetermined leave-one-out fits, no alarms or winner.
    fits=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines'];fm=read(ROOT/'outputs/rbr_v2_d1_v1/input_manifest.json')['rows'];fz=np.load(ROOT/'outputs/rbr_v2_d1_v1/RBR64.npy');fi=[i for i,r in enumerate(fm) if r['arm']=='BASELINE'];sensitivity=[]
    for omit in [-1,0,1,2]:
        idx=[i for j,i in enumerate(fi) if j!=omit];fit=fits['RBR64'] if omit==-1 else fit_baseline(fz[idx],[fm[i]['session_id'] for i in idx],'D_fit');b=transform(z[[i for i,r in base]],fit)
        sensitivity.append(dict(variant='frozen3' if omit==-1 else 'LOO_'+str(omit),sigma=fit['sigma'],minimum_scale=min(fit['scale']),per_channel_baseline_SD_over_fit_scale_quantiles=np.quantile(b.std(0),[0,.05,.5,.95,1]).tolist()))
    census=read(OUT/'static_census.json');counts=census['counts_by_prior_role'];cells=list(census['strict_future_cells']);rng=np.random.RandomState(plan['numerical']['synthetic_seed']);boot=[]
    groups={name:[counts[name][c] for c in cells] for name in ['D_new_science','final_calibration_candidates','V_AA_candidates','V_AB_candidates']}
    groups['strict_future_optimistic_combined']=[census['strict_future_cells'][c] for c in cells]
    groups['hypothetical_original80_pool_equal_cells']=[20]*4
    groups['hypothetical120_pool_equal_cells']=[30]*4
    for name,ns in groups.items():
        supported=0;minimum=[]
        for _ in range(2000):
            unique=[int(np.count_nonzero(rng.multinomial(n,np.full(n,1/n)))) if n else 0 for n in ns];minimum.append(min(unique));supported+=min(unique)>=10
        boot.append(dict(pool=name,cell_counts=ns,replicates=2000,bootstrap_replicates_with_all_cells_at_least10_unique=supported,minimum_unique_across_replicates=min(minimum),coverage_certified=False))
    # Current D3 exact factory/clock version only. Historical compatibility does not create more sources.
    actual={r['session_id'] for r in m if r['arm']=='BASELINE' and r['new_role']!='D_SWAP_CAL_EXISTING_B_REUSE'}
    Crows=list(__import__('csv').DictReader((OUT/'SESSION_Task_And_Role_Census.csv').open()));actual_cells={c:sum(r['session_id'] in actual and r['status']=='ELIGIBLE_Q' and r['cell']==c for r in Crows) for c in cells}
    write(OUT/'numerical_and_capacity.json',dict(physical_tail_diagnostics=tail,derivative_probes=probes,RBR_scaler_sensitivity=sensitivity,actual_same_version_D3_baseline_q_cells=actual_cells,bootstrap_support=boot,
        single_numeric_pipeline_certified=False,reason='Floating arithmetic/analytic filter can be verified, but sensor/measurement resolution and all30-feature error floors not certified; no floor adopted. RBR three-fit source stability insufficient.',
        delta010_power_at_design_gain020='NOT_ESTABLISHED_REAL_AA_AB_SUPPORT_AND_COVERAGE_MISSING',protected_read_sources=sources,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,new_vehicle_propagation=0,new_model_forward=0,new_training=0,downloads=0,Primary_V_C=0))
    table(OUT/'Numeric_Oracle_Probes.csv',probes);check();print('D6 analytic/numeric/capacity evidence complete; mandatory certification BLOCKED')

if __name__=='__main__':main()
