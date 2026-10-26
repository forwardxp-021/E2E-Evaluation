#!/usr/bin/env python3
"""Prospective all-SESSION second source frame; lossless cache packing, unchanged SDC, one global byte cap."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,hashlib,json,shutil,time,zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from tools.rbr_v2_d2_download import ROOT,OUT,LOCAL,LIMIT,RESERVE,read,write,sha,accounting


def plan():
 source=read(OUT/'acquisition_plan.json');catalog=read(OUT/'official_db_catalog.json')['entries'];groups={}
 for e in catalog:groups.setdefault(e['session_id'],[]).append(e)
 rows=[]
 for first in source['rows']:
  candidates=[e for e in groups[first['session_id']] if e['filename']!=first['filename'] and int(Path(e['filename']).stem.rsplit('_',2)[-1])-int(Path(e['filename']).stem.rsplit('_',2)[-2])>=120]
  if not candidates:continue
  item=min(candidates,key=lambda x:(x['uncompressed_bytes'],x['compressed_bytes'],hashlib.sha256(('D2_SECOND_SOURCE_COMPLETENESS_V1|'+x['archive']+'|'+x['member']).encode()).hexdigest()))
  rows.append(dict(item,SESSION_rank=first['SESSION_rank']))
 rows.sort(key=lambda r:(r['SESSION_rank'],r['session_id']))
 path=OUT/'round2_source_plan.json'
 if path.exists():raise FileExistsError('Second source frame already frozen')
 write(path,dict(version='D2_SOURCE_COMPLETENESS_FRAME_V2_PROSPECTIVE_BEFORE_ANY_D_SCIENCE',rows=rows,
                 rule='One additional minimum-size >=120s filename-duration LOG for EVERY SESSION with such a LOG, including first-round eligible SESSION; no failed-only lookup, no per-SESSION alternatives, same frozen16 anchor/physical SDC predicates',
                 basis='Official runnable query requires >=5 scene rows. Very short cost-minimum chunks lack runnable support;120s is a conservative source completeness margin, not a relaxed physics or scientific threshold.',
                 prior_frame_preserved=True,combined_session_rule='First eligible source in fixed round1 then round2 order; one SESSION once; all static attempts and denominators retained',
                 max_total_download_bytes=LIMIT,disk_reserve_bytes=RESERVE,target_total_static_eligible=400,science_attempts_so_far=0,
                 forbidden_dependency='No RBR/H30, MMD, mechanism/safety rollout or D outcomes used for selection',source_plan_sha256=sha(OUT/'acquisition_plan.json')))
 print('Prospective second LOG frame',len(rows),'SESSION; all selected irrespective of first-round eligibility',flush=True)


def pack_excluded():
 """Losslessly retain owned downloaded excluded DBs in compressed local cache. No historical files touched."""
 entries=[];dest=LOCAL/'lossless_excluded_cache';dest.mkdir(exist_ok=True)
 for result_path in sorted((OUT/'static').glob('*.json')):
  r=read(result_path)
  if r['status']=='ELIGIBLE':continue
  p=Path(r['log']['source_file']);archive=dest/(p.name+'.zlib');receipt=OUT/'cache_transforms'/(p.name+'.json')
  if receipt.exists():entries.append(read(receipt));continue
  st=p.stat();h=hashlib.sha256();co=zlib.compressobj(1);partial=archive.with_suffix('.zlib.writing')
  with p.open('rb') as f,partial.open('wb') as g:
   for b in iter(lambda:f.read(8388608),b''):h.update(b);g.write(co.compress(b))
   g.write(co.flush())
  if h.hexdigest()!=r['download_sha256']:raise ValueError('Raw source changed before lossless cache pack')
  os.replace(partial,archive)
  # Independently stream decompress and verify full raw SHA before removing only our owned unpacked copy.
  de=zlib.decompressobj();hh=hashlib.sha256();n=0
  with archive.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):
    raw=de.decompress(b);hh.update(raw);n+=len(raw)
  tail=de.flush();hh.update(tail);n+=len(tail)
  if hh.hexdigest()!=h.hexdigest() or n!=st.st_size or not de.eof:raise ValueError('Lossless cache restoration check failed')
  item=dict(raw_path=str(p),raw_sha256=h.hexdigest(),raw_bytes=n,raw_mtime_ns=st.st_mtime_ns,archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
            transform='zlib_level1_lossless_verified_full_stream',SESSION=r['session_id'],scope='ONLY_THIS_D2_NEW_DOWNLOAD_EXCLUDED_STATIC_CACHE',scientific_data_mutated=False)
  write(receipt,item);p.unlink();entries.append(item)
  if len(entries)%40==0:print('lossless source-cache packing',len(entries),'excluded DBs',flush=True)
 write(OUT/'lossless_cache_index.json',dict(entries=entries,saved_bytes=sum(e['raw_bytes']-e['archive_bytes'] for e in entries),no_network_download=True,restore='zlib decompression to original path; verify raw SHA; restore original mtime_ns; no scientific input/qualification changes'))


def restore(raw_path):
 p=Path(raw_path);item=read(OUT/'cache_transforms'/(p.name+'.json'))
 if p.exists():
  if sha(p)!=item['raw_sha256']:raise ValueError('Existing raw cache changed')
  return p
 if shutil.disk_usage(LOCAL).free-item['raw_bytes']<RESERVE:raise RuntimeError('DISK_RESERVE_STOP')
 if sha(item['archive_path'])!=item['archive_sha256']:raise ValueError('Lossless source archive changed')
 de=zlib.decompressobj();partial=p.with_suffix('.db.restore');h=hashlib.sha256()
 with Path(item['archive_path']).open('rb') as f,partial.open('wb') as g:
  for b in iter(lambda:f.read(8388608),b''):
   raw=de.decompress(b);h.update(raw);g.write(raw)
  tail=de.flush();h.update(tail);g.write(tail)
 if h.hexdigest()!=item['raw_sha256'] or not de.eof:raise ValueError('Raw source restoration hash failure')
 os.replace(partial,p);os.utime(p,ns=(item['raw_mtime_ns'],item['raw_mtime_ns']));return p


def combined():
 results={r['session_id']:r for r in map(read,(OUT/'static').glob('*.json'))}
 second=OUT/'round2/static'
 if second.exists():
  for r in map(read,second.glob('*.json')):
   if r['status']=='ELIGIBLE' and results.get(r['session_id'],{}).get('status')!='ELIGIBLE':results[r['session_id']]=r
 return results


def run():
 started=time.monotonic()
 while not (OUT/'acquisition_status.json').exists():
  if time.monotonic()-started>18000:raise TimeoutError('First acquisition dependency did not complete; no automatic science')
  time.sleep(5)
 if read(OUT/'acquisition_status.json')['stop_reason']=='BOUNDED_ENGINEERING_BATCH_CHECKPOINT':raise RuntimeError('Full first frame not completed')
 plan2=read(OUT/'round2_source_plan.json')
 if sum(r['status']=='ELIGIBLE' for r in combined().values())>=400:
  write(OUT/'round2_status.json',dict(stop_reason='TARGET_ALREADY_MET',processed=0));return
 pack_excluded()
 import tools.rbr_v2_d2_acquire as a
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env();a.OUT=OUT/'round2';a.OUT.mkdir(exist_ok=True)
 results=combined();stop='SECOND_SOURCE_FRAME_EXHAUSTED';i=0
 while i<len(plan2['rows']):
  if sum(r['status']=='ELIGIBLE' for r in results.values())>=400:stop='PREREGISTERED_STATIC_CAPACITY_TARGET';break
  batch=[];nr=0;nb=0;b=read(OUT/'download_budget.json')
  for e in plan2['rows'][i:i+20]:
   if (OUT/'downloads'/(e['filename']+'.json')).exists():batch.append(e);continue
   if shutil.disk_usage(LOCAL).free-nr-e['uncompressed_bytes']<RESERVE:stop='DISK_RESERVE_STOP';break
   if b['received_bytes']+b['reserved_bytes']+nb+e['compressed_bytes']+16384>LIMIT:stop='DOWNLOAD_50GB_BUDGET_STOP';break
   batch.append(e);nr+=e['uncompressed_bytes'];nb+=e['compressed_bytes']+16384
  if not batch:break
  with ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(a.get,e) for e in batch]
   for e,f in zip(batch,futures):
    receipt=f.result()
    try:r=a.scan(receipt)
    except Exception as error:write(a.OUT/'scan_errors'/(e['session_id']+'.json'),dict(error=repr(error),source_sha256=receipt['sha256']));raise
    if r['status']=='ELIGIBLE' and results.get(e['session_id'],{}).get('status')!='ELIGIBLE':results[e['session_id']]=r
    i+=1;ne=sum(v['status']=='ELIGIBLE' for v in results.values())
    write(OUT/'round2_progress.json',dict(processed=i,total_planned=len(plan2['rows']),combined_eligible_SESSION=ne,received_bytes=read(OUT/'download_budget.json')['received_bytes'],disk_free_bytes=shutil.disk_usage(LOCAL).free))
    print('second source',i,'/',len(plan2['rows']),e['session_id'],r['status'],'combined distinct eligible',ne,flush=True)
  if len(batch)<20:break
 write(OUT/'round2_status.json',dict(stop_reason=stop,processed=i,total_planned=len(plan2['rows']),combined_eligible_SESSION=sum(v['status']=='ELIGIBLE' for v in results.values()),download_bytes=read(OUT/'download_budget.json')['received_bytes'],disk_free_bytes=shutil.disk_usage(LOCAL).free))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run','restore']);p.add_argument('--path');a=p.parse_args()
 if a.mode=='plan':plan()
 elif a.mode=='run':run()
 else:print(restore(a.path))
