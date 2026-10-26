#!/usr/bin/env python3
"""Official public v1.1 ZIP range reader, incremental DB-only acquisition and hard byte accounting."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,contextlib,csv,fcntl,hashlib,io,json,shutil,struct,time,zipfile,zlib,threading
from pathlib import Path
import xml.etree.ElementTree as ET
import requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_d2_v1';LOCAL=ROOT.parent/'nuplan/dataset/data/cache/d2_trainval_v1'
BASE='https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/'
THREAD_LOCK=threading.RLock()
LIMIT=50_000_000_000;RESERVE=20*1024**3;NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.writing');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(t,p)
@contextlib.contextmanager
def accounting():
 with THREAD_LOCK, (OUT/'download_budget.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)


def init_budget():
 if not (OUT/'download_budget.json').exists():write(OUT/'download_budget.json',dict(limit_bytes=LIMIT,received_bytes=0,reserved_bytes=0,requests=[],start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def fetch(key,start,length,size,etag):
 if length<=0:return b''
 init_budget();end=start+length-1
 if start<0 or end>=size:raise ValueError('Range beyond official object boundary')
 with accounting():
  b=read(OUT/'download_budget.json')
  if b['received_bytes']+b['reserved_bytes']+length>LIMIT:raise RuntimeError('DOWNLOAD_50GB_BUDGET_STOP')
  b['reserved_bytes']+=length;rid=len(b['requests']);b['requests'].append(dict(key=key,start=start,end=end,expected=length,received=0,status='RESERVED'));write(OUT/'download_budget.json',b)
 data=bytearray();status='INTERRUPTED';started=time.monotonic()
 try:
  r=requests.get(BASE+key,headers={'Range':f'bytes={start}-{end}','If-Match':etag,'Accept-Encoding':'identity'},timeout=(15,60),stream=True)
  if r.status_code!=206:raise RuntimeError('Expected public HTTP206, got '+str(r.status_code)+'; no whole-archive fallback')
  if r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Remote Content-Range mismatch')
  if r.headers.get('ETag')!=etag:raise ValueError('Official object ETag changed')
  for part in r.iter_content(1024*1024):
   data.extend(part)
   if len(data)>length:raise ValueError('Remote sent beyond approved byte range')
  if len(data)!=length:raise ValueError('Incomplete compressed byte range')
  status='COMPLETE';return bytes(data)
 finally:
  if 'r' in locals():r.close()
  with accounting():
   b=read(OUT/'download_budget.json');b['received_bytes']+=len(data);b['reserved_bytes']-=length
   b['requests'][rid].update(received=len(data),status=status,wall_seconds=time.monotonic()-started);write(OUT/'download_budget.json',b)


class RemoteZip(io.RawIOBase):
 def __init__(self,key,size,etag):self.key=key;self.size=size;self.etag=etag;self.pos=0;self.cache=[]
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('Negative remote offset')
  return self.pos
 def read(self,n=-1):
  n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
  if n<=0:return b''
  for start,data in self.cache:
   if start<=self.pos and self.pos+n<=start+len(data):v=data[self.pos-start:self.pos-start+n];self.pos+=n;return v
  if n>20_000_000:raise RuntimeError('Unexpected >20MB metadata read; refuse archive body')
  start=self.pos;data=fetch(self.key,start,n,self.size,self.etag);self.cache.append((start,data));self.pos+=len(data);return data


def catalog():
 xml=ET.fromstring((OUT/'aws_v11.xml').read_bytes());archives=[];entries=[]
 for item in xml.findall('s:Contents',NS):
  key=item.findtext('s:Key',namespaces=NS);name=Path(key).name
  if not (name.startswith('nuplan-v1.1_train_') or name=='nuplan-v1.1_val.zip'):continue
  archives.append(dict(key=key,size=int(item.findtext('s:Size',namespaces=NS)),etag=item.findtext('s:ETag',namespaces=NS),official_split='val' if name.endswith('_val.zip') else 'train'))
 for i,a in enumerate(archives):
  f=RemoteZip(a['key'],a['size'],a['etag']);z=zipfile.ZipFile(f)
  for x in z.infolist():
   if x.is_dir() or not x.filename.endswith('.db'):continue
   name=Path(x.filename).name;sid=name.rsplit('_',2)[0]
   entries.append(dict(session_id=sid,filename=name,archive=a['key'],archive_size=a['size'],etag=a['etag'],official_split=a['official_split'],member=x.filename,
        compressed_bytes=x.compress_size,uncompressed_bytes=x.file_size,crc32=x.CRC,compression=x.compress_type,header_offset=x.header_offset,flags=x.flag_bits))
  print('official catalog',i+1,'/',len(archives),a['key'],len(z.infolist()),'members',flush=True)
 write(OUT/'official_db_catalog.json',dict(archives=archives,entries=entries,license_sha256=sha(OUT/'official_LICENSE.txt'),source='AWS registry official public v1.1; no test or sensor objects queried',zip_metadata_bytes_in_budget=True))


def plan():
 c=read(OUT/'official_db_catalog.json');old=read(ROOT/'outputs/rbr_v2_d1_v1/local_source_inventory.json');known={r['session_id'] for r in old['paths']};groups={}
 for x in c['entries']:
  if x['session_id'] in known:continue
  # Avoid invalid filename/missing complete8s support by metadata duration only, not outcome.
  parts=Path(x['filename']).stem.rsplit('_',2)
  if len(parts)!=3 or int(parts[-1])-int(parts[-2])<15:continue
  groups.setdefault(x['session_id'],[]).append(x)
 salt='RBR_V2_D2_OFFICIAL_NEW_SESSION_SOURCE_V1_20261026'
 rows=[]
 for sid,items in groups.items():
  item=min(items,key=lambda x:(x['uncompressed_bytes'],x['compressed_bytes'],hashlib.sha256((salt+'|LOG|'+sid+'|'+x['archive']+'|'+x['member']).encode()).hexdigest()))
  rows.append(dict(item,SESSION_rank=hashlib.sha256((salt+'|SESSION|'+sid).encode()).hexdigest()))
 rows.sort(key=lambda r:(r['SESSION_rank'],r['session_id']))
 # New C is metadata-reserved before raw access. Original C remains protected independently.
 protect=rows[:20];acquire=rows[20:]
 payload=dict(version='RBR_V2_D2_PROSPECTIVE_ACQUISITION_V1',salt=salt,source_rule='One minimum-uncompressed-size >=15s filename-duration SQLite chunk per new SESSION, compressed-size then fixed SHA ties; file-cost metadata only before any map/trajectory/outcome; original S2R representative preselection inside chosen LOG then first16 SDC-supported anchors; no outcome-driven second LOG',
              old_SESSION_excluded=sorted(known),new_C_METADATA_ONLY=[r['session_id'] for r in protect],C_reservations=protect,rows=acquire,
              download_limit_bytes=LIMIT,disk_reserve_bytes=RESERVE,batch_SESSION=40,science_max_new_SESSION=40,science_first_arms_max=80,
              target_static_capacity=400,target_reason='320 smallest unpaired final-cal+V envelope +40 new D + capacity margin; not a power-approved sample size',
              old_SDC_rule_sha256=sha(ROOT/'docs/stageR/e2e_scoped_direct_coast_v1/Frozen_Qualification_Protocol_v1.json'),no_test=True,no_sensor=True)
 target=OUT/'acquisition_plan.json'
 if target.exists():raise FileExistsError('Acquisition frame already frozen')
 write(target,payload);print('new independent source candidates',len(rows),'reserved C metadata',len(protect),'planned active',len(acquire),flush=True)


def download_entry(e):
 LOCAL.mkdir(parents=True,exist_ok=True);dest=LOCAL/e['filename'];receipt=OUT/'downloads'/(e['filename']+'.json')
 if receipt.exists():
  r=read(receipt)
  if sha(dest)!=r['sha256']:raise ValueError('Certified cache changed')
  return r
 if dest.exists():raise FileExistsError('Unreceipted data not overwritten')
 if shutil.disk_usage(LOCAL).free-e['uncompressed_bytes']<RESERVE:raise RuntimeError('DISK_RESERVE_STOP')
 start=time.monotonic();header=fetch(e['archive'],e['header_offset'],30,e['archive_size'],e['etag'])
 sig,version,flags,method,mt,md,crc,cs,us,nl,el=struct.unpack('<IHHHHHIIIHH',header)
 if sig!=0x04034b50 or method!=e['compression'] or flags&1:raise ValueError('Unsupported/encrypted ZIP member')
 names=fetch(e['archive'],e['header_offset']+30,nl+el,e['archive_size'],e['etag'])
 if names[:nl].decode('utf-8')!=e['member']:raise ValueError('ZIP local header name mismatch')
 data=fetch(e['archive'],e['header_offset']+30+nl+el,e['compressed_bytes'],e['archive_size'],e['etag'])
 raw=zlib.decompress(data,-15) if method==8 else data if method==0 else None
 if raw is None or len(raw)!=e['uncompressed_bytes'] or zlib.crc32(raw)&0xffffffff!=e['crc32']:raise ValueError('Official ZIP CRC/size verification failed')
 if not raw.startswith(b'SQLite format 3\0'):raise ValueError('Not a real SQLite database')
 partial=dest.with_suffix('.db.part');partial.write_bytes(raw);os.replace(partial,dest)
 result=dict(**e,path=str(dest),sha256=sha(dest),crc_verified=True,official_object_etag_verified=True,wall_seconds=time.monotonic()-start,
             license='CC-BY-NC-SA-4.0 plus Motional Dataset Terms; noncommercial PhD research',license_sha256=sha(OUT/'official_LICENSE.txt'))
 write(receipt,result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['catalog','plan']);a=p.parse_args();catalog() if a.mode=='catalog' else plan()
