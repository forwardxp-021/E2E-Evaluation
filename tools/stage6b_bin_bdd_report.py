#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd
from tools.stage6b_compare_baselines import load_manifest_arrays, mmd_with_stats


def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x
        return tqdm


def iter_progress(iterable, enabled=True, **kwargs):
    if not enabled:
        return iterable
    return get_tqdm()(iterable, **kwargs)


def balance_by_bins(a_idx,b_idx,df,keys,min_bin_size,rng):
    ta=df[df.global_row.isin(a_idx)].copy(); tb=df[df.global_row.isin(b_idx)].copy()
    ba=[]; bb=[]; rows=[]; gb=tb.groupby(keys)
    for key,ga in ta.groupby(keys):
        g2 = gb.get_group(key) if key in gb.groups else pd.DataFrame(columns=tb.columns)
        n=min(len(ga),len(g2)); used=n>=min_bin_size
        if used:
            ba.append(rng.choice(ga.global_row.values,n,replace=False)); bb.append(rng.choice(g2.global_row.values,n,replace=False))
        rows.append({'bin_key':str(key),'n_A':len(ga),'n_B':len(g2),'n_used':int(n if used else 0),'used':used})
    return (np.concatenate(ba) if ba else np.array([],dtype=int)), (np.concatenate(bb) if bb else np.array([],dtype=int)), pd.DataFrame(rows)


def main(a):
    t0=time.time(); progress_enabled = not a.no_progress
    out=Path(a.output_dir)
    if out.exists() and not a.overwrite: raise FileExistsError
    if out.exists() and a.overwrite: shutil.rmtree(out)
    (out/'plots').mkdir(parents=True, exist_ok=True)
    feat,emb=load_manifest_arrays(a.shard_manifest,a.embedding_manifest)
    a_idx=np.load(a.a_indices_path); b_idx=np.load(a.b_indices_path)
    rng=np.random.default_rng(a.seed)
    rows=[]; bal_rows=[]
    behavior=pd.read_csv(a.behavior_bins_path) if a.behavior_bins_path else None
    odd=pd.read_csv(a.odd_bins_path) if a.odd_bins_path else None

    if behavior is not None:
        cols=[x.strip() for x in a.report_bins.split(',') if x.strip() and x in behavior.columns]
        for col in iter_progress(cols, enabled=progress_enabled, desc='behavior-event BDD bins', unit='bin'):
            for val,sub in iter_progress(list(behavior.groupby(col)), enabled=progress_enabled, desc=f'{col} values', unit='value', leave=False):
                ai=np.intersect1d(a_idx, sub.global_row.values); bi=np.intersect1d(b_idx, sub.global_row.values)
                if len(ai)<a.min_bin_size or len(bi)<a.min_bin_size: continue
                st=mmd_with_stats(emb[ai],emb[bi],rng,a.num_bootstrap,a.num_permutation,a.max_mmd_samples)
                rows.append({'bin_type':'behavior_event','bin_name':col,'bin_value':val,'n_A':len(ai),'n_B':len(bi),**st,'control_bins':'','balanced':False,'n_A_balanced':len(ai),'n_B_balanced':len(bi),'interpretation':''})
                if odd is not None and a.control_bins:
                    for _ in iter_progress([0], enabled=progress_enabled, desc='ODD-controlled event BDD', leave=False):
                        merged=sub[['global_row']].merge(odd,on='global_row',how='inner')
                        bai,bbi,tab=balance_by_bins(ai,bi,merged,a.control_bins.split(','),a.min_bin_size,rng)
                    if len(bai)>0 and len(bbi)>0:
                        bst=mmd_with_stats(emb[bai],emb[bbi],rng,a.num_bootstrap,a.num_permutation,a.max_mmd_samples)
                        rows.append({'bin_type':'behavior_event','bin_name':col,'bin_value':val,'n_A':len(ai),'n_B':len(bi),**bst,'control_bins':a.control_bins,'balanced':True,'n_A_balanced':len(bai),'n_B_balanced':len(bbi),'interpretation':'odd_controlled'})
                        bal_rows.extend(tab.to_dict('records'))

    if odd is not None and a.control_bins:
        for _ in iter_progress([0], enabled=progress_enabled, desc='overall ODD-balanced BDD', unit='step'):
            bai,bbi,tab=balance_by_bins(a_idx,b_idx,odd,a.control_bins.split(','),a.min_bin_size,rng)
        if len(bai)>0 and len(bbi)>0:
            raw=mmd_with_stats(emb[a_idx],emb[b_idx],rng,a.num_bootstrap,a.num_permutation,a.max_mmd_samples)
            bst=mmd_with_stats(emb[bai],emb[bbi],rng,a.num_bootstrap,a.num_permutation,a.max_mmd_samples)
            rows.append({'bin_type':'odd_control','bin_name':'overall','bin_value':'raw', 'n_A':len(a_idx),'n_B':len(b_idx),**raw,'control_bins':'','balanced':False,'n_A_balanced':len(a_idx),'n_B_balanced':len(b_idx),'interpretation':'raw'})
            rows.append({'bin_type':'odd_control','bin_name':'overall','bin_value':'odd_balanced', 'n_A':len(a_idx),'n_B':len(b_idx),**bst,'control_bins':a.control_bins,'balanced':True,'n_A_balanced':len(bai),'n_B_balanced':len(bbi),'interpretation':'odd_balanced'})
            bal_rows.extend(tab.to_dict('records'))

    pd.DataFrame(rows).to_csv(out/'bin_bdd_summary.csv', index=False)
    pd.DataFrame(bal_rows).to_csv(out/'odd_balance_summary.csv', index=False)
    (out/'warnings.json').write_text(json.dumps({'note':'plots omitted in headless minimal implementation'},indent=2,ensure_ascii=False), encoding='utf-8')
    (out/'bin_bdd_report.md').write_text(f'# Stage6B bin BDD report\n\n- total runtime seconds: {time.time()-t0:.3f}\n', encoding='utf-8')

if __name__=='__main__':
 p=argparse.ArgumentParser();
 p.add_argument('--embedding_manifest',required=True); p.add_argument('--shard_manifest',required=True); p.add_argument('--feature_schema_path',required=True)
 p.add_argument('--a_indices_path',required=True); p.add_argument('--b_indices_path',required=True); p.add_argument('--odd_bins_path'); p.add_argument('--behavior_bins_path'); p.add_argument('--output_dir',required=True)
 p.add_argument('--report_bins',default=''); p.add_argument('--control_bins',default=''); p.add_argument('--num_bootstrap',type=int,default=50); p.add_argument('--num_permutation',type=int,default=100); p.add_argument('--max_mmd_samples',type=int,default=2000); p.add_argument('--min_bin_size',type=int,default=100); p.add_argument('--seed',type=int,default=42); p.add_argument('--no_progress',action='store_true'); p.add_argument('--overwrite',action='store_true');
 main(p.parse_args())
