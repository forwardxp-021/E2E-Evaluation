#!/usr/bin/env python3
"""Wait within this Work for40 legal new D sources, then zero-run prepare only; never launches science."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess,time
from tools.rbr_v2_d2_download import ROOT,OUT,read,write
PYTHON='/Users/liuqing/miniconda3/envs/nuplan/bin/python'


def main():
 start=time.monotonic()
 while time.monotonic()-start<18000:
  p=OUT/'acquisition_progress.json';n=read(p)['counts'].get('ELIGIBLE',0) if p.exists() else 0
  if n>=40:break
  if (OUT/'acquisition_status.json').exists():raise RuntimeError('First frame ended with fewer40 eligible D sources; do not invent or duplicate source support')
  time.sleep(5)
 else:raise TimeoutError('Bounded D preparation dependency wait exceeded')
 for label,args in [('prepare',['tools/rbr_v2_d2_prepare.py']),('preflight',['tools/rbr_v2_d2_execute.py','--preflight'])]:
  with (OUT/('D_'+label+'.log')).open('x') as f:subprocess.run([PYTHON,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(OUT/'D_zero_run_ready.json',dict(status='ZERO_RUN_READY_PENDING_AGENT_COMMIT_VERIFICATION',science_started=False))
 print('D80 zero-run ready; no automatic scientific run or user approval request',flush=True)

if __name__=='__main__':main()
