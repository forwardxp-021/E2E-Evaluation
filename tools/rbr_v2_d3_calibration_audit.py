#!/usr/bin/env python3
"""Read-only score algebra/identity audit of frozen D arrays; no refit, threshold change or new release operating point."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha
from tools.rbr_v2_d1_statistics import transform,mmd2
from tools.rbr_v2_d2_statistics import kernels
OUT=ROOT/'outputs/rbr_v2_d3_v1';DOC=ROOT/'docs/stageR/rbr_v2_d3_v1'


def main():
 m=read(OUT/'interface/input_manifest.json');r=read(OUT/'D_statistics_results.json');fits=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines'];arrays={k:np.load(OUT/'interface'/(k+'.npy')) for k in ['RBR64','H30']};indices={}
 for i,row in enumerate(m['rows']):indices.setdefault(row['session_id'],{})[row['arm']]=i
 rows=[];max_error=0.
 for g,ids in r['groups'].items():
  bi=[indices[s]['BASELINE'] for s in ids];ti=[indices[s]['TREATMENT'] for s in ids]
  for name,a in arrays.items():
   fit=fits[name];b=transform(a[bi],fit);t=transform(a[ti],fit);K=kernels(a[bi],a[ti],fit);off=K[:20,:20][np.triu_indices(20,1)].mean();identical=mmd2(b,b,fit['sigma']);identity=2/20*(off-1)
   np.testing.assert_allclose(identical,identity,rtol=0,atol=2e-14)
   filename='D_SWAP_CAL_'+name+'_statistics.npy' if g=='cal' else 'D_AA_CONTROLLED_SWAP_'+name+'_statistics.npy' if g=='aa' else 'D_AB_PAIRED_PI50_'+name+'_statistics.npy'
   stats=np.load(OUT/filename);rng=np.random.RandomState({'cal':2026091101,'aa':2026091102,'ab':2026091103}[g]);errors=[]
   for j in range(20):
    selected=rng.choice(20,20,replace=False);drift=np.zeros(20,bool);drift[rng.choice(20,10,replace=False)]=True
    left=b[selected].copy();right=np.where(drift[:,None],t[selected],b[selected])
    if g!='ab':
     swap=rng.randint(0,2,20).astype(bool);old=left.copy();left[swap]=right[swap];right[swap]=old[swap]
    direct=mmd2(left,right,fit['sigma']);errors.append(abs(direct-stats[j]));np.testing.assert_allclose(direct,stats[j],rtol=0,atol=2e-14)
   max_error=max(max_error,max(errors));diag=np.diag(K[:20,20:]);mask=[m['rows'][i]['valid_cells']/(80*83) for i in bi]
   baseline_distances=np.linalg.norm(b[:,None]-b[None,:],axis=-1)[np.triu_indices(20,1)]
   rows.append(dict(role=g,representation=name,independent_SESSION=20,baseline_off_diagonal_kernel_mean=float(off),baseline_identical_shared_log_score=float(identical),shared_log_formula_value=float(identity),
       shared_log_diagnostic_only_not_independent_AA=True,BT_same_SESSION_kernel_mean=float(diag.mean()),BT_same_SESSION_kernel_min=float(diag.min()),score_quantiles=np.quantile(stats,[0,.05,.5,.95,1]).tolist(),
       frozen_threshold=r['thresholds'][name]['value'],baseline_mask_valid_fraction_mean=float(np.mean(mask)),baseline_pairwise_standardized_distance_median=float(np.median(baseline_distances)),
       direct_original_formula_vs_cached_20_draws_max_abs_error=max(errors)))
 result=dict(status='NUMERICAL_AND_IDENTITY_AUDIT_PASS_CALIBRATION_TRANSFER_DIAGNOSTIC_FAILURE',rows=rows,direct_vs_cached_max_abs_error=max_error,source_groups_unchanged=True,
      shared_log_MMD_identity='For m identical rows on both sides with the fixed unbiased two-sample formula, MMD2(X,X)=2/m*(mean_offdiag_K-1), generally negative. This is a structural shared-source term, not numerical error.',
      inference='RBR score ranges are disjoint across fixed calibration/held-out AA/AB pools. Source-dependent background kernel geometry and small paired-calibration pools are an evidenced calibration-transfer problem. Direct formula parity rules out cached-kernel/index-order arithmetic error; it does not prove a unique population cause.',
      illegal_rescues_not_performed=['negative score clamp','new bandwidth','refit scaler','recalibrate on held-out AA/AB','select another checkpoint','choose successful subset','use shared-log diagnostic as independent operational AA'],
      baseline_identity_scope='Algebra check only, never a replacement AA experiment',new_model_forward_or_gradient_calls=0,Primary_V_C=False,
      source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'D_statistics_results.json',OUT/'interface/input_manifest.json',OUT/'interface/RBR64.npy',OUT/'interface/H30.npy',ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json']})
 write(OUT/'paired_calibration_transfer_audit.json',result)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.4))
 for ax,name in zip(axes,['RBR64','H30']):
  for role,label,color,file in [('cal','Frozen swap calibration','gray','D_SWAP_CAL_'),('aa','Held-out swap AA','tab:blue','D_AA_CONTROLLED_SWAP_'),('ab','Paired AB, pi=0.5','tab:orange','D_AB_PAIRED_PI50_')]:
   s=np.load(OUT/(file+name+'_statistics.npy'));ax.hist(s,bins=45,density=True,alpha=.45,color=color,label=label)
  ax.axvline(r['thresholds'][name]['value'],color='red',ls='--',label='Frozen threshold');ax.set(title=name+' frozen score distributions',xlabel='Unbiased-form MMD2 (negative values retained)',ylabel='Conditional finite-pool density');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(DOC/'D3_Frozen_Calibration_Transfer_v1.png',dpi=170);plt.close(fig)
 print('Exact formula audit PASS, max error',max_error,'; source calibration transfer NOT VALIDATED; no parameter changes',flush=True)

if __name__=='__main__':main()
