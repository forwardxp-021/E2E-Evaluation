#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from pathlib import Path
from tools.interaction_context_features import write_feature_schema_json

p = argparse.ArgumentParser()
p.add_argument('--out_dir', required=True)
args = p.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
write_feature_schema_json(out / 'feature_schema.json')
print(out / 'feature_schema.json')
