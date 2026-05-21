#!/usr/bin/env python3
import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
from pathlib import Path
import pandas as pd

def main(a):
    d=Path(a.input_dir)
    bdd=json.loads((d/'bdd_summary.json').read_text(encoding='utf-8'))
    c=pd.read_csv(d/'category_delta.csv') if (d/'category_delta.csv').exists() else pd.DataFrame()
    f=pd.read_csv(d/'feature_delta.csv') if (d/'feature_delta.csv').exists() else pd.DataFrame()
    s=pd.read_csv(d/'scenario_slice_delta.csv') if (d/'scenario_slice_delta.csv').exists() else pd.DataFrame()
    t=pd.read_csv(d/'top_drift_cases.csv') if (d/'top_drift_cases.csv').exists() else pd.DataFrame()
    topf=f.reindex(f.delta_normalized.abs().sort_values(ascending=False).head(10).index) if not f.empty else pd.DataFrame()
    concl=f"A/B 存在{'明显' if bdd.get('mmd2',0)>0.01 else '轻微'}行为分布漂移（BDD-MMD={bdd.get('mmd2',0):.4f}）。"
    lines=['# Style Report Card','',f'**结论**: {concl}','','## Overall BDD',f"- metric: {bdd.get('metric')}",f"- mmd2: {bdd.get('mmd2'):.6f}",f"- CI95: [{bdd.get('ci95_low'):.6f}, {bdd.get('ci95_high'):.6f}]",f"- p-value: {bdd.get('p_value'):.6f}", '', '## Category-wise style delta']
    lines.append(c.to_markdown(index=False) if not c.empty else '_missing_')
    lines += ['', '## Top feature deltas', topf.to_markdown(index=False) if not topf.empty else '_missing_','', '## Scenario/proxy slice analysis', s.to_markdown(index=False) if not s.empty else '_missing_','', '## Top drift cases', t.head(20).to_markdown(index=False) if not t.empty else '_missing_','', '## Warnings and limitations','- BDD measures overall distribution drift magnitude only.','- Directional interpretation comes from category/feature/case analysis.','- In unpaired mode, results may be affected by scenario distribution mismatch.','- 建议先看 scenario/proxy slice 再下工程结论。']
    (d/'style_report_card.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); main(p.parse_args())
