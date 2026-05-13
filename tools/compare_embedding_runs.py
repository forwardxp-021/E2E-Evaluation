#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOCUS=['centroid_accuracy_overall','hit_at_1','mean_same_label_fraction_topk','spearman_rms_jerk_delta','spearman_rms_yaw_rate_delta','spearman_rms_curvature_delta','spearman_mean_speed_delta']

def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for item in args.runs:
        name, path = item.split('=',1)
        bdf=pd.read_csv(Path(path)/'baseline_comparison_summary.csv')
        lr=bdf[bdf['method']=='learned'].iloc[0]
        row={'run':name}
        for m in FOCUS: row[m]=lr.get(m, np.nan)
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(out/'comparison_summary.csv', index=False)
    (out/'comparison_summary.md').write_text(df.to_markdown(index=False)+'\n')

    n_runs = max(1, len(df))
    fig_w = max(6, 1.4 * n_runs)
    label_rot = 30 if n_runs >= 5 else 0
    plt.figure(figsize=(fig_w,4)); plt.bar(df['run'], df['centroid_accuracy_overall']); plt.xticks(rotation=label_rot, ha='right' if label_rot else 'center'); plt.tight_layout(); plt.savefig(out/'comparison_classification_bar.png'); plt.close()
    plt.figure(figsize=(fig_w,4)); plt.bar(df['run'], df['hit_at_1']); plt.xticks(rotation=label_rot, ha='right' if label_rot else 'center'); plt.tight_layout(); plt.savefig(out/'comparison_retrieval_bar.png'); plt.close()
    corr_cols=['spearman_rms_jerk_delta','spearman_rms_yaw_rate_delta','spearman_rms_curvature_delta','spearman_mean_speed_delta']
    x=np.arange(n_runs); w=0.18 if n_runs >= 5 else 0.2
    plt.figure(figsize=(max(10, 1.6 * n_runs),4))
    for i,c in enumerate(corr_cols): plt.bar(x+i*w, df[c], w, label=c.replace('spearman_',''))
    plt.xticks(x+1.5*w, df['run'], rotation=label_rot, ha='right' if label_rot else 'center'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/'comparison_style_correlation_bar.png'); plt.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    run(p.parse_args())
