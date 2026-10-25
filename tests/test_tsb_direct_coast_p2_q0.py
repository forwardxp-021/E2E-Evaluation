"""P2 zero-run selection and fail-closed authorization tests; no model/run calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from tools.tsb_direct_coast_p2_q0 import ROOT,OUT,ranked_frame,reject_q0_execution,read_csv


class P2Q0Tests(unittest.TestCase):
    def fixture(self):
        rows=[];roles={};logs=[]
        for sid,exposure,speed in [('s1','E1_UNRELATED_HISTORICAL_USE',6),('s2','E2_BENCHMARK_ENGINEERING',5),('s3','E3_PRIMARY_METHOD_DEVELOPMENT',8),('s4','E5_UNTOUCHED_CONFIRMATORY',9),('s5','E1_UNRELATED_HISTORICAL_USE',4.999)]:
            rows.append(dict(session_id=sid,representative_log_id=sid+'log',representative_scenario_token=sid+'token',static_eligibility_status='STATIC_ELIGIBLE',execution_initial_speed_mps=str(speed)))
            roles[sid]=dict(exposure_class=exposure,source_role_provenance_status='COMPLETE',source_fingerprint_complete='true',role_reservation_status='CANDIDATE_POOL_ONLY_NO_FINAL_V_OR_C_RESERVATION')
            logs.append(dict(log_id=sid+'log',session_id=sid,frozen_fingerprint_match=True,source_stat_unchanged_during_read=True,observed_db_fingerprint_sha256=sid+'hash'))
        return rows,roles,logs

    def test_hash_ranking_permutation_and_forbidden_outcome_independence(self):
        rows,roles,logs=self.fixture();a=ranked_frame(rows,roles,set(),{'s4'},{'s4hash'},logs,'fixed')[1]
        changed=copy.deepcopy(list(reversed(rows)))
        for row in changed:row.update(mechanism_pass=False,F_match='FAIL',collision_count=999,RBR_win=True)
        b=ranked_frame(changed,roles,set(),{'s4'},{'s4hash'},logs,'fixed')[1]
        self.assertEqual(a,b);self.assertEqual({x['session_id'] for x in a},{'s1','s2'})

    def test_session_exclusion_unknown_and_C_source_alias_fail_closed(self):
        rows,roles,logs=self.fixture();roles['s1']['exposure_class']='UNKNOWN'
        eligible=ranked_frame(rows,roles,{'s2'},{'s4'},{'s4hash'},logs,'fixed')[1];self.assertEqual(eligible,[])
        rows,roles,logs=self.fixture();logs[0]['observed_db_fingerprint_sha256']='s4hash'
        eligible=ranked_frame(rows,roles,set(),{'s4'},{'s4hash'},logs,'fixed')[1];self.assertEqual([x['session_id'] for x in eligible],['s2'])

    def test_duplicate_SESSION_rejected(self):
        rows,roles,logs=self.fixture()
        with self.assertRaises(ValueError):ranked_frame(rows+[rows[0]],roles,set(),set(),set(),logs,'fixed')

    def test_old_and_proposed_and_manual_authorized_credentials_cannot_run_Q0(self):
        for stage,status in [('B1_TSB_QUALIFICATION','AUTHORIZED'),('TSB_COAST_P1R_C_FIRST_PAIR','AUTHORIZED'),('TSB_DIRECT_COAST_F2','AUTHORIZED'),('TSB_DIRECT_COAST_P2','PROPOSED_ONLY'),('TSB_DIRECT_COAST_P2','AUTHORIZED')]:
            with self.assertRaises(PermissionError):reject_q0_execution(dict(stage=stage,authorization_status=status),'DIRECTCOAST-P2-01-BASELINE')

    def test_frozen_actual_twenty_ranked_SESSION_and_raw_LOG_identities(self):
        cap=json.loads((OUT/'evidence/P2_Q0_Capacity_And_Selection_v1.json').read_text());rows=read_csv(OUT/'P2_Q0_Independent_Session_Roster_v1.csv');rule=json.loads((OUT/'P2_Q0_Prospective_Selection_Rule_v1.json').read_text())
        self.assertEqual(len(rows),20)
        for key in ['session_id','log_id','database_log_token','source_database_sha256']:self.assertEqual(len({r[key] for r in rows}),20)
        excluded=set(cap['all_old_B1_sessions'])|set(cap['all_Coast_DEV_sessions'])|set(cap['protected_C_E5_sessions']);self.assertFalse(excluded&{r['session_id'] for r in rows})
        ranks=read_csv(OUT/'evidence/P2_Q0_Full_Eligible_Hash_Ranking_v1.csv');self.assertEqual([r['session_id'] for r in rows],[r['session_id'] for r in ranks[:20]])
        for row in rows:self.assertEqual(row['rank_sha256'],hashlib.sha256((rule['session_rank_salt']+'|'+row['session_id']).encode()).hexdigest())
        self.assertEqual(hashlib.sha256((OUT/'P2_Q0_Independent_Session_Roster_v1.csv').read_bytes()).hexdigest(),cap['roster_sha256'])

    def test_real_historical_metadata_fingerprint_is_not_full_file_digest(self):
        from tools.tsb_direct_coast_p2_q0 import historical_db_fingerprint,sha256_file
        row=read_csv(OUT/'P2_Q0_Independent_Session_Roster_v1.csv')[0]
        self.assertEqual(historical_db_fingerprint(Path(row['db_path'])),row['source_database_sha256'])
        self.assertNotEqual(sha256_file(Path(row['db_path'])),row['source_database_sha256'])

    def test_zero_run_tool_has_no_launch_model_or_science_analysis_call_sites(self):
        tree=ast.parse((ROOT/'tools/tsb_direct_coast_p2_q0.py').read_text());forbidden={'run','propagate_state','track_trajectory','launch','launch_first_pair_arm','analyze_completed_arm','analyze_completed_pair','measure','fmatch_pair','claim_attempt'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                self.assertNotIn(name,forbidden)


if __name__=='__main__':unittest.main()
