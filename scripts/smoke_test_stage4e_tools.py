#!/usr/bin/env python3
import tempfile, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd

def run():
  td=Path(tempfile.mkdtemp()); base=td/'outputs/waymo_human_v1_full51'; (base/'eval_with_learned').mkdir(parents=True)
  bdf=pd.DataFrame([{'method':'learned','centroid_accuracy_overall':0.6,'hit_at_1':0.7,'mean_same_label_fraction_topk':0.68,'hit_at_1_lift_over_chance':0.3,'spearman_mean_speed_delta':0.5,'spearman_rms_jerk_delta':0.1,'spearman_rms_yaw_rate_delta':0.3,'spearman_rms_curvature_delta':0.4,'spearman_mean_thw_delta':np.nan,'valid_pairs_mean_speed_delta':1000,'valid_pairs_rms_jerk_delta':1000,'valid_pairs_rms_yaw_rate_delta':1000,'valid_pairs_rms_curvature_delta':1000,'valid_pairs_mean_thw_delta':10}])
  bdf.to_csv(base/'eval_with_learned/baseline_comparison_summary.csv',index=False)
  (base/'human_embedding_model').mkdir(); (base/'pseudo_labels').mkdir()
  (base/'build_summary.json').write_text(json.dumps({'n_files_processed':1,'n_scenarios_processed':2,'n_windows_kept':3,'split_counts':{'train':1,'val':1,'test':1},'front_found_rate':1.0}))
  (base/'pseudo_labels/pseudo_label_summary.json').write_text(json.dumps({'conservative_like':1,'aggressive_like':1,'lateral_stable_like':1,'n_unlabeled':0,'split_labeled_counts':{'train':1,'val':1,'test':1}}))
  (base/'human_embedding_model/train_summary.json').write_text(json.dumps({'n_total':3,'n_retained':3,'n_dropped':0,'traj_nan_count_raw':0,'traj_nan_count_after_sanitize':0,'traj_repaired_count':0,'feature_clipped_values':0,'best_val_loss':1.0}))
  (base/'embedding_export_summary.json').write_text(json.dumps({'shape':[3,64],'row_aligned':True}))
  subprocess.check_call(['python','tools/generate_paper_tables.py','--eval_dir',str(base/'eval_with_learned'),'--train_summary',str(base/'human_embedding_model/train_summary.json'),'--export_summary',str(base/'embedding_export_summary.json'),'--pseudo_label_summary',str(base/'pseudo_labels/pseudo_label_summary.json'),'--build_summary',str(base/'build_summary.json'),'--out_dir',str(base/'paper_tables')])
  subprocess.check_call(['python','tools/compare_embedding_runs.py','--runs',f'v1={base}/eval_with_learned',f'v2={base}/eval_with_learned','--out_dir',str(base/'compare')])
  subprocess.check_call(['python','tools/train_human_behavior_embedding.py','--out_dir',str(td/'train_smoke'),'--smoke_test','--epochs','1','--batch_size','16','--feature_weight_mode','jerk_comfort'])
  subprocess.check_call(['python','tools/evaluate_vehicledata_validation.py','--smoke_test'])
  print('smoke_test_stage4e_tools_pass')
if __name__=='__main__': run()
