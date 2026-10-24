"""Pure accounting fixtures and consistency of already-computed F1 evidence.

Never invokes the F1 analysis again, original qualification, or a simulator.
"""
import ast
import csv
from io import StringIO
from decimal import Decimal
import json
from pathlib import Path
import unittest
import numpy as np
from tools.tsb_coast_f1_readonly_analysis import (
    ROOT,OLD,OUT,LABEL,filter_area_identity,motion_moments,decompose_motion,
)


class F1ReadOnlyAccountingTests(unittest.TestCase):
    def test_filter_area_endpoint_term_on_static_analytic_arrays(self):
        r=filter_area_identity([-1,0,0],[0,-1/3,-2/9],[-1/3,-2/9,-4/27])
        self.assertAlmostEqual(r['identity_residual_mps'],0,places=14)
        self.assertNotEqual(r['endpoint_filter_term_mps'],0)

    def test_discrete_distance_moment_and_symmetric_product_decomposition(self):
        for baseline,treatment in [([10,9,9,8],[10,10,9,8]),([10,9,8,7],[10,10,9,9])]:
            b=motion_moments(baseline);t=motion_moments(treatment);d=decompose_motion(b,t)
            self.assertAlmostEqual(b['distance_identity_residual_m'],0,places=12)
            self.assertAlmostEqual(t['distance_identity_residual_m'],0,places=12)
            self.assertAlmostEqual(d['sum_m'],t['forward_euler_distance_m']-b['forward_euler_distance_m'],places=12)
        with self.assertRaises(ValueError):motion_moments([10,10,10])

    def test_published_decimal_differences_are_exploratory_and_old_fields_blank(self):
        old=list(csv.DictReader(StringIO((OLD/'P1R_D_FMatch_And_Safety_v1.csv').read_text())))
        new=list(csv.DictReader(StringIO((OUT/'F1_PostHoc_Motion_Comparison_v1.csv').read_text())))
        self.assertTrue(all(r['analysis_label']==LABEL and r['formal_pair_result']=='NOT_EVALUATED_UNCHANGED' for r in new))
        for r in old:
            if r['category']!='F_SINGLE_ARM_DESCRIPTIVE_ONLY':continue
            self.assertEqual(r['absolute_pair_delta'],'')
            current=next(x for x in new if x['metric']==r['metric'])
            self.assertEqual(Decimal(current['signed_T_minus_B']),Decimal(r['treatment_actual'])-Decimal(r['baseline_actual']))

    def test_actual_stage_conservation_and_model_area_identity(self):
        summary=json.loads((OUT/'evidence/F1_Numeric_Analysis_v1.json').read_text())
        for arm,data in summary['arms'].items():
            self.assertAlmostEqual(sum(x['signed_net_speed_loss_mps'] for x in data['stages']),data['motion_moments']['net_speed_loss_mps'],places=12)
            self.assertAlmostEqual(data['actual_model_negative_impulse_mps']-data['actual_model_positive_impulse_mps'],data['motion_moments']['net_speed_loss_mps'],places=12)
            self.assertLess(abs(data['filter_area_identity']['identity_residual_mps']),1e-12)
            self.assertLess(abs(data['observed_position_minus_euler_speed_integral_m']),1e-8)
        account=summary['path_difference_accounting']
        self.assertLess(abs(account['actual_path_minus_decomposed_delta_residual_m']),1e-8)
        ledger=json.loads((ROOT/'outputs/stageR/tsb_coast_p1r_c_execution_state_v1/budget.json').read_text())
        self.assertTrue(ledger['stopped']);self.assertEqual(len(ledger['attempts']),2)

    def test_actual_zero_requests_and_pair_speed_time_coverage(self):
        path=ROOT/'outputs/stageR/tsb_coast_p1r_d_first_pair_v1/P1R-C2-DEV-01-TREATMENT/telemetry/actual_lqr_controller_telemetry.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual(len(rows[21:39]),18)
        self.assertTrue(all(r['longitudinal_requested_acceleration']==0 and r['coast_command_reached_model'] and r['state_propagation_continues'] for r in rows[21:39]))
        data=list(csv.DictReader(StringIO((OUT/'F1_Paired_Speed_Difference_v1.csv').read_text())))
        self.assertEqual(len(data),80);self.assertEqual(data[-1]['baseline_request_mps2'],'')
        for i,r in enumerate(data):
            self.assertAlmostEqual(float(r['episode_time_s']),i/10,places=12)
            self.assertAlmostEqual(float(r['speed_difference_T_minus_B_mps']),float(r['treatment_speed_mps'])-float(r['baseline_speed_mps']),places=12)

    def test_no_production_or_qualification_imports_or_calls(self):
        tree=ast.parse((ROOT/'tools/tsb_coast_f1_readonly_analysis.py').read_text())
        forbidden={'analyze_completed_arm','analyze_completed_pair','measure','fmatch_pair','audit_proxy_response','propagate_state','track_trajectory','launch_first_pair_arm','compute_trajectory'}
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
                self.assertTrue(all(not n.startswith(('nuplan','torch','tools.')) for n in names))
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                self.assertNotIn(name,forbidden)


if __name__=='__main__':unittest.main()
