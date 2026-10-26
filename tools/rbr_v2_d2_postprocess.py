#!/usr/bin/env python3
"""D-only post-run inference/statistics with persistent CPU accounting, never launches simulation or V/C."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math,signal,subprocess,time
from pathlib import Path
import psutil
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha
PYTHON='/Users/liuqing/miniconda3/envs/nuplan/bin/python';TORCH=ROOT/'outputs/rbr_v2_smoke_v1/runtime/bin/python'


def run_child(label,cmd,remaining):
 start=time.monotonic();limit=max(1,int(math.floor(remaining)));observed=0.;reason=None
 def cpu_limit():
  import resource
  resource.setrlimit(resource.RLIMIT_CPU,(limit,limit))
 with (OUT/('D_'+label+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=cpu_limit)
  while True:
   try:c=psutil.Process(p.pid).cpu_times();observed=max(observed,c.user+c.system)
   except psutil.NoSuchProcess:pass
   pid,status,usage=os.wait4(p.pid,os.WNOHANG)
   if pid:p.returncode=os.waitstatus_to_exitcode(status);break
   if observed>=remaining or time.monotonic()-start>1800:
    reason='D_NUMERICAL_CPU_OR_WALL_LIMIT';os.killpg(p.pid,signal.SIGKILL);_,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status);break
   time.sleep(.1)
 return dict(stage=label,returncode=p.returncode,wall_seconds=time.monotonic()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,rss_peak_native=usage.ru_maxrss,reason=reason)


def main():
 started=time.monotonic()
 while time.monotonic()-started<18000:
  if (OUT/'science_ledger.json').exists():
   state=read(OUT/'science_ledger.json')
   if state['stopped']:break
  time.sleep(5)
 else:raise TimeoutError('D science completion dependency exceeded; no autonomous rerun')
 if state['stop_reason']!='FROZEN_D80_COMPLETE' or len(state['attempts'])!=80 or any(r['status']=='TECHNICAL_FAIL' for r in state['attempts']):
  write(OUT/'D_numerical_blocked.json',dict(reason='INCOMPLETE_TECHNICAL_FULL_DENOMINATOR',actual_attempts=len(state['attempts']),source_stop_reason=state['stop_reason'],statistics_executed=False));return
 b=read(OUT/'D_execution_binding.json')
 for name,h in b['source_sha256'].items():
  if sha(ROOT/name)!=h:raise PermissionError('D numerical source differs from pre-science freeze '+name)
 if sha(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')!=b['D_fit_scalers_sha256']:raise PermissionError('Frozen D baseline scaler changed')
 if (OUT/'D_postprocess_resources.json').exists():raise FileExistsError('One D numerical identity only; retain old progress before any repair')
 stages=[('features',[PYTHON,'-u','tools/rbr_v2_d2_interface.py','extract']),('encoding',[str(TORCH),'-u','tools/rbr_v2_d2_interface.py','encode']),('statistics',[PYTHON,'-u','tools/rbr_v2_d2_statistics.py'])];records=[];driver=time.process_time()
 for label,cmd in stages:
  used=state['cpu_seconds']+state.get('driver_cpu_seconds',0)+sum(r['cpu_seconds'] for r in records)+time.process_time()-driver
  if used>=14400:raise RuntimeError('No scientific CPU budget remains for numerical stage')
  r=run_child(label,cmd,14400-used);records.append(r);write(OUT/'D_postprocess_resources.json',dict(stages=records,scientific_CPU_budget=14400,rollout_and_driver_CPU=state['cpu_seconds']+state.get('driver_cpu_seconds',0),numerical_CPU=sum(x['cpu_seconds'] for x in records),numerical_driver_CPU=time.process_time()-driver))
  if r['returncode']!=0:raise RuntimeError('D numerical stage failed; preserve inputs/logs: '+label)
 write(OUT/'D_numerical_complete.json',dict(status='D_ONLY_NUMERICAL_COMPLETE_NO_PRIMARY',wall_seconds=time.monotonic()-started,all_new_attempts=80,new_optimizer_steps=0,V_C_execution=0))
 print('Real D-only frozen model/strong H30 numerical evidence complete',flush=True)

if __name__=='__main__':main()
