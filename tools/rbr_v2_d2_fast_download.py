#!/usr/bin/env python3
"""Bounded-memory resumable member transfer. Exact official bytes/CRC/SHA and same global50GB ledger."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fcntl,hashlib,shutil,struct,time,zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tools.rbr_v2_d2_download import ROOT,OUT,LOCAL,RESERVE,read,write,sha,fetch
SEGMENT=16*1024*1024


def download_entry(e):
 LOCAL.mkdir(parents=True,exist_ok=True);dest=LOCAL/e['filename'];receipt=OUT/'downloads'/(e['filename']+'.json');parts=LOCAL/'download_parts'/e['filename'];parts.mkdir(parents=True,exist_ok=True)
 with (parts/'member.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if receipt.exists():
   r=read(receipt)
   if sha(dest)!=r['sha256']:raise ValueError('Certified SQLite cache changed')
   return r
  if dest.exists():raise FileExistsError('Unreceipted raw DB is never overwritten')
  if shutil.disk_usage(LOCAL).free-e['uncompressed_bytes']-e['compressed_bytes']<RESERVE:raise RuntimeError('DISK_RESERVE_STOP')
  started=time.monotonic();head=fetch(e['archive'],e['header_offset'],30,e['archive_size'],e['etag']);sig,version,flags,method,mt,md,crc,cs,us,nl,el=struct.unpack('<IHHHHHIIIHH',head)
  if sig!=0x04034b50 or flags&1 or method not in [0,8] or method!=e['compression']:raise ValueError('ZIP local format unsupported')
  names=fetch(e['archive'],e['header_offset']+30,nl+el,e['archive_size'],e['etag'])
  if names[:nl].decode('utf-8')!=e['member']:raise ValueError('ZIP member name mismatch')
  origin=e['header_offset']+30+nl+el;chunks=[(i,offset,min(SEGMENT,e['compressed_bytes']-offset)) for i,offset in enumerate(range(0,e['compressed_bytes'],SEGMENT))]
  def one(c):
   i,offset,n=c;p=parts/f'{i:06d}.bin';proof=parts/f'{i:06d}.json'
   if proof.exists():
    q=read(proof)
    if q['etag']!=e['etag'] or q['offset']!=origin+offset or q['bytes']!=n or sha(p)!=q['sha256']:raise ValueError('Certified ranged cache segment changed')
    return p
   data=fetch(e['archive'],origin+offset,n,e['archive_size'],e['etag']);partial=p.with_suffix('.bin.writing');partial.write_bytes(data);os.replace(partial,p)
   write(proof,dict(etag=e['etag'],offset=origin+offset,bytes=n,sha256=sha(p)));return p
  with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(one,chunks))
  partial=dest.with_suffix('.db.part');de=zlib.decompressobj(-15) if method==8 else None;h=hashlib.sha256();check=0;count=0;prefix=b''
  with partial.open('wb') as g:
   for f in files:
    with f.open('rb') as datafile:
     for block in iter(lambda:datafile.read(1024*1024),b''):
      raw=de.decompress(block) if de else block
      if len(prefix)<16:prefix=(prefix+raw)[:16]
      g.write(raw);h.update(raw);check=zlib.crc32(raw,check);count+=len(raw)
   if de:
    tail=de.flush();g.write(tail);h.update(tail);check=zlib.crc32(tail,check);count+=len(tail)
  if count!=e['uncompressed_bytes'] or check&0xffffffff!=e['crc32'] or prefix!=b'SQLite format 3\0' or de is not None and not de.eof:raise ValueError('Official member CRC/size/SQLite verification failed')
  os.replace(partial,dest);result=dict(**e,path=str(dest),sha256=h.hexdigest(),crc_verified=True,official_object_etag_verified=True,wall_seconds=time.monotonic()-started,transport='resumable16MiB member ranges;4 parallel segments; bounded streaming decompression',
       license='CC-BY-NC-SA-4.0 plus Motional Dataset Terms; noncommercial PhD research',license_sha256=sha(OUT/'official_LICENSE.txt'))
  write(receipt,result)
  # Only transport temporaries are removed after a complete verified raw DB is certified.
  for f in files:f.unlink();f.with_suffix('.json').unlink()
  return result
