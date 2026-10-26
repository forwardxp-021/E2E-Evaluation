#!/usr/bin/env python3
"""Package D5 decision artifacts and verify directly relevant frozen inputs only."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from pathlib import Path
from tools.rbr_v2_d5_evidence import ROOT,OUT,PLAN,read,write,check,sha
DOC=ROOT/'docs/stageR/rbr_v2_d5_v1'


def package():
    check();r=read(OUT/'evidence.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(4)
    ax.bar(x-.18,[v['raw_alarm_fraction'] for v in r['synthetic']],.36,label='Original task composition')
    ax.bar(x+.18,[v['oracle_task_standardized_alarm_fraction'] for v in r['synthetic']],.36,label='Common-task sampling*')
    ax.set_xticks(x);ax.set_xticklabels(['Matched-task\nnull','Task-mixture\nchange only','Known behavior\nchange','Residual task\nconfounding'])
    ax.axhline(.05,color='red',ls='--',label='Nominal alpha=0.05');ax.set(ylabel='Alarm fraction (500 synthetic datasets/design)',ylim=(0,1.08),title='Marginal change, conditional null, and insufficient conditioning')
    ax.legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3);ax.grid(alpha=.2)
    fig.text(.5,.015,'* Last case balances coarse S only; the full continuous context C is not matched.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(DOC/'D5_Null_And_Task_Composition.png',dpi=160);plt.close(fig)
    write(DOC/'Status_v1.json',dict(status='D5_SCIENTIFIC_ROUTE_RECOMMENDED_PENDING_OWNER_APPROVAL',recommended_route='TASK_STANDARDIZED_UNPAIRED_VERSION_MONITORING_WITH_EXTERNAL_NULL_CALIBRATION',current_double_brake_positive_RBR_Level2_investment='NOT_JUSTIFIED_BY_CURRENT_D_EVIDENCE_NOT_A_FORMAL_PRIMARY_REJECTION',D4_Primary_approved=False,new_240_D_sources_recommended=False,SAP_status='PROPOSED_ONLY',detector_ready=False,model_H30_SAP_and_old_results_unchanged=True,new_simulation=0,new_model_inference=0,new_training=0,downloads=0,Primary_V_C=0))
    before=(OUT/'user_Quick_before.txt').read_bytes();section=(OUT/'quick_section.md').read_bytes()
    if (ROOT/'QUICK_REFERENCE.md').read_bytes()!=before+section:raise ValueError('User Quick prefix changed')
    csvpath=ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'
    if sha(csvpath)!='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8':raise ValueError('Protected CSV changed')
    write(OUT/'protection.json',dict(user_Quick_prefix_unchanged=True,protected_CSV_sha256=sha(csvpath),historical_direct_sources_checked=len(read(PLAN)['protected']),unrelated_historical_scan=False))
    print('D5 decision package created; no method activated')


def manifest():
    check();target=DOC/'Manifest_v1.json';paths=list(OUT.rglob('*'))+list(DOC.rglob('*'))+list((ROOT/'configs/rbr_v2_d5_v1').rglob('*'))+list((ROOT/'tools').glob('rbr_v2_d5_*.py'))+list((ROOT/'tests').glob('test_rbr_v2_d5_*.py'));files={}
    for p in paths:
        if not p.is_file() or p==target or '__pycache__' in p.parts or p.name=='user_Quick_before.txt' or p.name.startswith('manifest_packaging'):continue
        files[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    write(target,dict(status='D5_SCIENTIFIC_ROUTE_RECOMMENDED_PENDING_OWNER_APPROVAL',source_commit=read(PLAN)['start_commit'],files=files,protected=read(PLAN)['protected'],SAP_activated=False,new_scientific_execution=0))
    print('D5 manifest bound',len(files),'own files')


def verify():
    check();m=read(DOC/'Manifest_v1.json')
    for p,h in m['files'].items():
        if sha(ROOT/p)!=h['sha256']:raise ValueError('D5 artifact changed '+p)
    print('D5 verification PASS:',len(m['files']),'own hashes; frozen model/statistics and directly related historical evidence unchanged; no Primary activation')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['package','manifest','verify']);a=p.parse_args();{'package':package,'manifest':manifest,'verify':verify}[a.mode]()
