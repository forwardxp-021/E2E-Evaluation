#!/usr/bin/env python3
"""Lossless engineering accelerators. Canonical83/model definitions remain read-only."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hashlib
import math
import numpy as np
from tools.rbr_v2_canonical83 import canonical,resample,waymo_tracks,SLOTS
from tools.waymo_lane_utils import extract_lane_polylines,project_point_to_lane,find_best_lane_for_agent
from tools.lane_aware_assignment import assign_neighbors_lane_aware


def mask_plan_fast(valid,identity,seed):
    """Same ordered first-visit span algorithm as mask_plan, vectorized (no new RNG)."""
    from tools.rbr_v2_canonical83 import PRESENCE
    valid=np.asarray(valid,bool).copy();valid[:,PRESENCE]=False
    rng=np.random.RandomState(int.from_bytes(hashlib.sha256((str(seed)+':'+identity).encode()).digest()[:4],'little'))
    spans=np.argwhere(valid[:-1]|valid[1:])
    spans=spans[rng.permutation(len(spans))]
    cells=spans[:,0]*83+spans[:,1]
    visits=np.stack([cells,cells+83],axis=1).reshape(-1)
    visits=visits[valid.reshape(-1)[visits]]
    unique,first=np.unique(visits,return_index=True)
    target=int(math.ceil(.15*valid.sum()))
    if target==0 or len(unique)<target:raise ValueError('Insufficient reconstruction cells')
    selected=unique[np.argsort(first,kind='stable')[:target]]
    out=np.zeros(80*83,bool);out[selected]=True
    return out.reshape(80,83)


def projections_batch(xyh,lanes):
    """Batch lane screening, exact original final projections and boundary fallback.

    Stable lane order/top32 and the original abs(l) cost are preserved. Near any
    threshold/segment/lane tie, rerun the complete original scalar finder.
    """
    if not lanes:return [None]*len(xyh)
    xyh=np.asarray(xyh,np.float64);xy=xyh[:,:2];keys=list(lanes);ls=list(lanes.values())
    q=len(xy);k=min(32,len(ls));rank=np.arange(k)[None,:]
    centers=np.array([l.bbox_center_xy for l in ls]);lo=np.array([l.bbox_min_xy for l in ls])-20
    hi=np.array([l.bbox_max_xy for l in ls])+20
    hits=((xy[:,None,:]>=lo)&(xy[:,None,:]<=hi)).all(2);has=hits.any(1)
    dc=np.hypot(xy[:,None,0]-centers[:,0],xy[:,None,1]-centers[:,1])
    dc=np.where(has[:,None]&~hits,np.inf,dc)
    cand=np.argsort(dc,axis=1,kind='stable')[:,:k]
    counts=np.where(has,hits.sum(1),len(ls));active=rank<counts[:,None]
    dist=np.full((q,k),np.inf);hd=np.full((q,k),np.inf);ambiguous=np.zeros(q,bool)
    eps=1e-8+64*np.finfo(np.float64).eps*(1+max(float(np.abs(xy).max()),float(np.abs(centers).max())))
    for li,lane in enumerate(ls):
        qi,ki=np.where((cand==li)&active)
        if not len(qi):continue
        for offset in range(0,len(qi),128):
            ids=qi[offset:offset+128];cols=ki[offset:offset+128];p=xy[ids,None,:]
            den=lane.seg_den;ok=den>1e-9
            if not ok.any():continue
            u=np.sum((p-lane.seg_start_xy)*lane.seg_vec_xy,axis=2)/np.where(ok,den,1.)
            u=np.clip(u,0.,1.);foot=lane.seg_start_xy+u[:,:,None]*lane.seg_vec_xy
            d=p-foot;eu=np.linalg.norm(d,axis=2);eu[:,~ok]=np.inf
            bi=np.argmin(eu,axis=1);rr=np.arange(len(ids))
            heading=np.asarray(lane.seg_heading,np.float64)[bi]
            left=np.stack([-np.sin(heading),np.cos(heading)],1)
            lat=np.abs(np.sum(d[rr,bi]*left,axis=1))
            dh=np.abs((xyh[ids,2]-heading+np.pi)%(2*np.pi)-np.pi)
            good=(lat<=3)&(dh<=np.pi/4)&np.isfinite(eu[rr,bi])
            dist[ids,cols]=np.where(good,lat,np.inf);hd[ids,cols]=np.where(good,dh,np.inf)
            ambiguous[ids]|=(np.abs(lat-3)<=eps)|(np.abs(dh-np.pi/4)<=eps)
    order=np.lexsort((np.broadcast_to(rank,(q,k)),hd,dist),axis=1);best=order[:,0]
    result=[]
    for i in range(q):
        if ambiguous[i]:
            proj=find_best_lane_for_agent(xy[i],xyh[i,2],lanes,3.,np.pi/4,20.,32,False)[0]
        elif np.isfinite(dist[i,best[i]]):
            # Recompute ALL numerical contenders with the original scalar kernel,
            # in the original candidate order, rather than approximating ties.
            choices=[];unexpected=False
            for ci in np.flatnonzero(dist[i]<=dist[i,best[i]]+eps):
                p=project_point_to_lane(xy[i],lanes[keys[cand[i,ci]]])
                dh0=abs((xyh[i,2]-p['heading']+np.pi)%(2*np.pi)-np.pi)
                unexpected|=abs(p['distance_to_lane']-dist[i,ci])>eps
                if p['distance_to_lane']<=3 and dh0<=np.pi/4:
                    choices.append((p['distance_to_lane'],dh0,int(ci),p))
            proj=min(choices,key=lambda x:x[:3])[3] if choices else None
            if unexpected or proj is None:
                proj=find_best_lane_for_agent(xy[i],xyh[i,2],lanes,3.,np.pi/4,20.,32,False)[0]
        else:proj=None
        result.append(proj)
    return result


class PreparedScenario:
    """One decode/resampling/projection per Scenario/start shared by all ego windows."""
    def __init__(self,scenario,start=0):
        times=np.rint(np.array(scenario.timestamps_seconds,np.float64)*1e6).astype(np.int64)
        self.tracks={k:resample(a,times,times[start]) for k,a in waymo_tracks(scenario).items()}
        self.lanes=extract_lane_polylines(scenario);self.states=[];self.projections=[]
        for t in range(80):
            st={k:dict(x=a[t,0],y=a[t,1],velocity_x=a[t,2],velocity_y=a[t,3],heading=a[t,4],
                       speed=float(np.hypot(a[t,2],a[t,3])),valid=True)
                for k,a in self.tracks.items() if a[t,5]>.5}
            self.states.append(st)
        refs=[(t,aid) for t,st in enumerate(self.states) for aid in st]
        if refs:
            queries=np.array([[self.states[t][aid][key] for key in ['x','y','heading']] for t,aid in refs])
            # Bounding geometry and matrix allocations bounded to 512 queries.
            found=[]
            for offset in range(0,len(queries),512):found.extend(projections_batch(queries[offset:offset+512],self.lanes))
        else:found=[]
        self.projections=[{} for _ in range(80)]
        for (t,aid),proj in zip(refs,found):
            if proj is not None:self.projections[t][aid]=proj

    def sample(self,ego_id):
        ego_id=str(ego_id);ego=self.tracks[ego_id]
        n=np.zeros((80,5,6));ids=np.full((80,5),'',dtype='U64');known=np.zeros(80,bool)
        cfg=dict(allow_geometric_adjacent_lane_inference=False,ego_projection_precomputed=True,candidate_projections_complete=True)
        for t in range(80):
            if ego[t,5]<=.5:continue
            st=self.states[t];candidates={k:v for k,v in st.items() if k!=ego_id}
            pr={k:v for k,v in self.projections[t].items() if k!=ego_id}
            result=assign_neighbors_lane_aware(st[ego_id],candidates,self.lanes,'lane_aware_only',cfg,
                ego_projection=self.projections[t].get(ego_id),candidate_projections=pr)
            known[t]=result.lane_assignment_available
            for j,slot in enumerate(SLOTS):
                aid=result.slot_to_agent.get(slot)
                if aid is not None:ids[t,j]=aid;n[t,j]=self.tracks[aid][t]
        return canonical(ego,n,ids,known)
