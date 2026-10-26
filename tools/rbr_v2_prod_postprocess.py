#!/usr/bin/env python3
"""Single-task dependency waiter; never launches training or downstream evaluation."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_prod_v1'
TRAIN_PY=ROOT/'outputs/rbr_v2_smoke_v1/runtime/bin/python'
REPORT_PY=Path('/Users/liuqing/miniconda3/envs/nuplan/bin/python')


def main():
    started=time.monotonic()
    while time.monotonic()-started<90000:
        progress=json.loads((OUT/'run/progress.json').read_text())
        state=progress['state']
        if state in ('NUMERICAL_FAILURE','BUDGET_STOP','INFRASTRUCTURE_INTERRUPTED'):
            raise RuntimeError('Training stopped; retain state without starting any new identity: '+state)
        attempts=json.loads((OUT/'run/attempts.json').read_text())
        if state=='TRAINING_COMPLETED_FIXED_EPOCH30' and 'wall_seconds' in attempts[-1]:
            if attempts[-1]['returncode']!=0:raise RuntimeError('Training process did not finish normally')
            break
        time.sleep(10)
    else:raise TimeoutError('Postprocess dependency wait exceeded bounded window')
    for label in ['first','second']:
        with (OUT/('export_'+label+'.log')).open('w') as log:
            subprocess.run([str(TRAIN_PY),str(ROOT/'tools/rbr_v2_prod_train.py'),'export'],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (OUT/'report_packaging.log').open('w') as log:
        subprocess.run([str(REPORT_PY),str(ROOT/'tools/rbr_v2_prod_report.py')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    (OUT/'postprocess_complete.flag').write_text('FORMAL_EPOCH30_RELOAD_EXPORT_REPORT_COMPLETE\n')
    print('Postprocess completed; no new training or downstream execution',flush=True)


if __name__=='__main__':main()
