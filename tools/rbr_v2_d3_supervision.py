#!/usr/bin/env python3
"""Process-only supervisor: bounded post-parent cleanup, full hardware-wall CPU upper charging."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal,subprocess,time
from pathlib import Path


def supervise(command,log_path,remaining_CPU_seconds,wall_limit=1200):
 import psutil
 started=time.monotonic();cores=os.cpu_count() or 1;deadline=min(wall_limit,remaining_CPU_seconds/cores);observed={};known={};reason=None;cleanup=[];forced=False
 with Path(log_path).open('x') as f:
  p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  def observe():
   try:members=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
   except psutil.NoSuchProcess:members=[]
   for m in members:
    try:
     key=(m.pid,m.create_time());c=m.cpu_times();observed[key]=max(observed.get(key,0),c.user+c.system);known[key]=m
    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
  while True:
   observe();pid,status,usage=os.wait4(p.pid,os.WNOHANG)
   if pid:p.returncode=os.waitstatus_to_exitcode(status);break
   if time.monotonic()-started>=deadline:
    reason='HARD_WALL_OR_CONSERVATIVE_CPU_BUDGET_STOP';os.killpg(p.pid,signal.SIGKILL);_,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status);break
   time.sleep(.02)
  grace=time.monotonic()
  while True:
   live=[]
   for key,m in known.items():
    if m.pid==p.pid:continue
    try:
     if m.is_running() and m.status()!=psutil.STATUS_ZOMBIE:
      c=m.cpu_times();observed[key]=max(observed.get(key,0),c.user+c.system);live.append(m)
      if not any(x['pid']==m.pid for x in cleanup):cleanup.append(dict(pid=m.pid,creation_time=key[1],cmdline=m.cmdline(),pgid=os.getpgid(m.pid)))
    except (psutil.NoSuchProcess,psutil.ZombieProcess,ProcessLookupError):pass
   if not live:break
   if time.monotonic()-grace>=2 or time.monotonic()-started>=deadline:
    try:os.killpg(p.pid,signal.SIGKILL);forced=True
    except ProcessLookupError:pass
    break
   time.sleep(.02)
 elapsed=time.monotonic()-started;measured=max(sum(observed.values()),usage.ru_utime+usage.ru_stime)
 return dict(returncode=p.returncode,reason=reason,wall_seconds=elapsed,cpu_seconds_measured=measured,cpu_seconds_budget_charged=max(measured,elapsed*cores),hardware_CPU_upper_seconds=elapsed*cores,
      cpu_bound_semantics='Every arm charged full elapsed wall times hardware CPU count, including post-parent cleanup; conservative upper bound, not actual measured consumption',
      post_parent_helpers_observed=cleanup,post_parent_cleanup_forced=forced,wait4_CPU_seconds=usage.ru_utime+usage.ru_stime)
