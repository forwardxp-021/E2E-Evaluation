#!/usr/bin/env python3
"""Export existing F2 actual records only; never launches scientific execution."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
import numpy as np
from tools.s2r_production_executor import sha256_file
from tools.tsb_direct_coast_f2_executor import OUT,SCIENCE_ROOT,LEDGER


def export():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not font.exists():raise FileNotFoundError('F2 Chinese plot font absent')
    matplotlib.font_manager.fontManager.addfont(str(font));plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'svg.hashsalt':'F2_FIXED_v1','svg.fonttype':'path'})
    read=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    data={};rows=[];source={};reviews={}
    for arm in ('BASELINE','TREATMENT'):
        root=SCIENCE_ROOT/('F2-DEV-01-'+arm)
        if not (root/'trace/realized_current_ego.jsonl').is_file():continue
        trace=read(root/'trace/realized_current_ego.jsonl');control=read(root/'telemetry/actual_lqr_controller_telemetry.jsonl');planner=read(root/'telemetry/planner_transfer.jsonl')
        if len(trace)!=80 or len(control)!=79:raise ValueError('F2 incomplete actual arm; report without fabricating complete curves')
        manifest=json.loads((root/'execution_manifest.json').read_text())
        for name,digest in manifest['artifacts'].items():
            if sha256_file(root/name)!=digest:raise ValueError('F2 actual artifact changed: '+name)
        ts=[r['current_ego']['time_us'] for r in trace];t=np.asarray([x-ts[0] for x in ts])/1e6
        if ts!=[ts[0]+i*100000 for i in range(80)]:raise ValueError('F2 plotted clock mismatch')
        v=[r['current_ego']['speed_mps'] for r in trace];a=[control[0]['model_acceleration_before']]+[r['model_realized_acceleration'] for r in control]
        for i,c in enumerate(control):
            if c['time_us']!=ts[i] or c['next_time_us']!=ts[i+1] or c['model_acceleration_before']!=a[i]:raise ValueError('F2 plotted command/model state alignment mismatch')
        u=[r['longitudinal_requested_acceleration'] for r in control];ref=[r['reference_acceleration_mps2'][0] for r in planner[:79]]
        data[arm]=dict(t=t,v=v,a=a,u=u,ref=ref)
        for i in range(80):rows.append(dict(arm=arm,iteration=i,episode_time_s=float(t[i]),simulation_time_us=ts[i],actual_speed_mps=v[i],actual_model_acceleration_state_mps2=a[i],
            planner_current_interval_reference_mps2=ref[i] if i<79 else '',final_request_mps2=u[i] if i<79 else '',tracking_mode=control[i]['tracking_mode'] if i<79 else 'NO_NEXT_TRANSITION'))
        for name in ('trace/realized_current_ego.jsonl','telemetry/actual_lqr_controller_telemetry.jsonl','telemetry/planner_transfer.jsonl','execution_manifest.json','arm_science_review.json'):
            source[str(root/name)]=sha256_file(root/name)
        reviews[arm]=json.loads((root/'arm_science_review.json').read_text())
    if not data:raise ValueError('F2 has no actual arm to plot')
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)
        path=OUT/(name+'.svg');path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(len(data),1,figsize=(12,4*len(data)),sharex=True,squeeze=False,layout='constrained')
    for ax,(arm,d) in zip(axes[:,0],data.items()):
        for s,e,c in [(1.1,2.1,'#f0d8be'),(2.1,3.9,'#d4e9df'),(3.9,4.9,'#f0d8be')]:ax.axvspan(s,e,color=c,alpha=.45)
        ax.step(d['t'],d['ref']+[d['ref'][-1]],where='post',color='#888888',ls=':',lw=3,label='planner当前参考请求')
        ax.step(d['t'],d['u']+[d['u'][-1]],where='post',color='#825ca5',ls='--',lw=1.8,label='最终直接请求（区间起点）')
        ax.plot(d['t'],d['a'],color='#ce5834',lw=2,label='官方模型实际加速度（状态时刻）')
        ax.axhline(0,color='#555555',lw=.8);ax.set_title('F2 '+('基线：一次直接制动' if arm=='BASELINE' else '处理：两次直接制动，中间零纵向请求'));ax.set_ylabel('加速度（m/s²）');ax.grid(alpha=.2);ax.legend(fontsize=9,loc='lower right')
    axes[-1,0].set(xlim=(0,7.9),xlabel='规范物理时间（秒）');save(fig,'F2_Reference_Final_Request_Model_v1')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    ax.axvspan(2.1,3.9,color='#d4e9df',alpha=.6,label='Treatment零纵向请求窗口')
    for arm,d in data.items():ax.plot(d['t'],d['v'],lw=2,label='Baseline实际速度' if arm=='BASELINE' else 'Treatment实际速度')
    ax.set(xlim=(0,7.9),xlabel='规范物理时间（秒）',ylabel='实际速度（m/s）',title='F2 两臂真实速度：仅使用已执行闭环记录');ax.grid(alpha=.2);ax.legend();save(fig,'F2_Actual_Speed_Comparison_v1')
    pair_path=SCIENCE_ROOT/'F2-DEV-01-pair_review.json';pair=json.loads(pair_path.read_text()) if pair_path.exists() else None
    result=dict(arms=reviews,pair=pair,ledger=json.loads(LEDGER.read_text()),plot_sources_sha256=source,
        plot_source_sha256=sha256_file(Path(__file__)),model_alignment='TRANSITION_OUTPUT_AT_NEXT_TIME_COMMAND_AT_CURRENT_TIME',
        capability='ABSTRACT_COAST_PROXY_ONLY',scientific_reexecution=0)
    (OUT/'F2_Actual_Results_v1.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if pair:
        f=pair['F_match']
        with (OUT/'F2_FMatch_v1.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=['metric','baseline','treatment','absolute_delta','caliper','pass','scope'],lineterminator='\n');w.writeheader()
            for key in f['calipers']:w.writerow(dict(metric=key,baseline=f['baseline'][key],treatment=f['treatment'][key],absolute_delta=f['absolute_delta'][key],caliper=f['calipers'][key],scope='FORMAL_F2_DEVELOPMENT_PAIR_FMATCH',**{'pass':f['absolute_delta'][key]<=f['calipers'][key]+1e-12}))
    return dict(status=pair['status'] if pair else 'ARM_GATE_OR_INFRASTRUCTURE_STOP',actual_arms=list(data))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',action='store_true',required=True);p.parse_args();print(json.dumps(export(),indent=2))
