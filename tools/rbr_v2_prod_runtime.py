#!/usr/bin/env python3
"""Read-only shard loader, exact masks, streaming train-only normalization."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import hashlib
import json
import resource
from pathlib import Path
import numpy as np
import torch
from tools.rbr_v2_model import Normalizer,normalize
from tools.rbr_v2_prod_features import mask_plan_fast

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_prod_v1';CFG=ROOT/'configs/rbr_v2_prod_v1/protocol.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.writing');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(temp,path)


class Corpus:
    def __init__(self,verify=True):
        self.manifest=json.loads((OUT/'data_manifest.json').read_text());self.n=self.manifest['rows']
        self.hash=sha(OUT/'data_manifest.json');self.shards=self.manifest['shards'];self.maps=[]
        soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft<1024:resource.setrlimit(resource.RLIMIT_NOFILE,(min(1024,hard) if hard!=-1 else 1024,hard))
        self.metadata=[];self.fingerprints={}
        for s in self.shards:
            directory=OUT/'cache'/s['shard_id']
            for file,h in s['sha256'].items():
                path=directory/file
                if verify and sha(path)!=h:raise ValueError('Certified cache hash mismatch: '+str(path))
                st=path.stat();self.fingerprints[path]=(st.st_ino,st.st_size,st.st_mtime_ns)
            arrays={k:np.load(directory/(k+'.npy'),mmap_mode='r') for k in ['x','channel_valid','ego_valid','semantic','semantic_valid']}
            n=s['rows']
            for k,shape in [('x',(n,80,83)),('channel_valid',(n,80,83)),('ego_valid',(n,80)),('semantic',(n,13)),('semantic_valid',(n,13))]:
                if arrays[k].shape!=shape:raise ValueError('Frozen shard shape error')
            np.testing.assert_array_equal(np.load(directory/'time_us.npy'),np.arange(80,dtype=np.int64)*100000)
            rows=list(csv.DictReader((directory/'rows.csv').open()))
            if len(rows)!=n:raise ValueError('Metadata denominator mismatch')
            self.metadata.extend(rows);self.maps.append(arrays)
        if len(self.metadata)!=self.n or [int(r['global_index']) for r in self.metadata]!=list(range(self.n)):
            raise ValueError('Global ordered sample identity mismatch')
        self.starts=np.array([s['global_start'] for s in self.shards],np.int64)
        self.identities=[':'.join([r['scenario_id'],r['actor_id'],r['start'],r['length']]) for r in self.metadata]
        diag=set(self.manifest['diagnostic_identities']);self.diagnostic=np.array([i for i,sid in enumerate(self.identities) if sid in diag],np.int64)
        if len(self.diagnostic)!=256:raise ValueError('Fixed diagnostic identities missing')

    def unchanged(self):
        for p,state in self.fingerprints.items():
            s=p.stat()
            if (s.st_ino,s.st_size,s.st_mtime_ns)!=state:raise ValueError('Frozen cache changed during training: '+str(p))

    def gather(self,indices):
        ix=np.asarray(indices,np.int64)
        if np.any(ix<0) or np.any(ix>=self.n):raise ValueError('Invalid frozen dataset index')
        result={k:np.empty((len(ix),)+a.shape[1:],dtype=a.dtype) for k,a in self.maps[0].items()}
        sid=np.searchsorted(self.starts,ix,side='right')-1
        for s in np.unique(sid):
            pos=np.flatnonzero(sid==s);local=ix[pos]-self.starts[s]
            for k,a in self.maps[s].items():result[k][pos]=a[local]
        return result

    def batch(self,indices,scaler,mask_seed,masked=True):
        a=self.gather(indices);x=normalize(a['x'],a['channel_valid'],scaler['input'])
        st=normalize(a['semantic'],a['semantic_valid'],scaler['semantic'])
        if masked:
            hm=np.stack([mask_plan_fast(v,json.dumps([self.hash,self.metadata[int(i)]['scenario_id'],
                self.metadata[int(i)]['actor_id'],self.metadata[int(i)]['start']]),mask_seed)
                for i,v in zip(indices,a['channel_valid'])])
        else:hm=np.zeros_like(a['channel_valid'])
        visible=a['channel_valid']&~hm;corrupted=np.where(visible,x,0)
        return tuple(torch.from_numpy(v) for v in [corrupted,visible,a['ego_valid'],st,x,a['semantic_valid'],hm])


def fit_normalizer(corpus):
    nx=Normalizer(83);ny=Normalizer(13)
    for si,arrays in enumerate(corpus.maps):
        for offset in range(0,len(arrays['x']),256):
            sl=slice(offset,offset+256);x=arrays['x'][sl];m=arrays['channel_valid'][sl]
            y=arrays['semantic'][sl];ym=arrays['semantic_valid'][sl]
            if not np.isfinite(x).all() or not np.isfinite(y[ym]).all():raise ValueError('Nonfinite frozen inputs')
            if not np.all(arrays['ego_valid'][sl].sum(1)>=64) or not np.all(m[:,:,6].sum(1)>=2):
                raise ValueError('Input eligibility changed in full corpus')
            nx.update(x,m,'train');ny.update(y,ym,'train')
        print('Train-only normalizer shard %d/%d'%(si+1,len(corpus.maps)),flush=True)
    inp,sem=nx.state(True),ny.state()
    if sem['zero_support']:raise ValueError('Unsupported semantic target in full training set')
    return dict(input=inp,semantic=sem,fit_rows=corpus.n,fit_role='TRAIN_ONLY',
        diagnostic_is_in_training=True,data_manifest_sha256=corpus.hash,
        source_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED')
