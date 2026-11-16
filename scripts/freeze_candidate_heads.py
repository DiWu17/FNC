"""Record external head hashes before independent calibration/evaluation."""
import argparse
from pathlib import Path

from clickagent.action_benchmark import sha256,write_json
from clickagent.public_benchmark import read_jsonl,cache_key

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=Path("runs/candidate_v2"))
args = parser.parse_args()
output = args.output
cases = read_jsonl(output/"manifest.jsonl")
present = sum((output/"features"/f"{cache_key(r['sample_id'])}.npz").exists() for r in cases if r["split"]=="test")
write_json(output/"training_weights_manifest.json",{
    "models_sha256":{p.name:sha256(p) for p in (output/"models").glob("*.pt")},
    "test_features_collected_at_freeze":present,
    "training_records_sha256":sha256(output/"training_records.json")})
print("External heads frozen",flush=True)
