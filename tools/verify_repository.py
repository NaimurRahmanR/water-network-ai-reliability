from pathlib import Path
import hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
KNOWN={
"results/net3/baseline_v0_5/baseline_mlp_v0_5.pt":"3e83192de687c42db771964aa82715b6221ad8dc89423484115a9b6aed23a040",
"results/net3/graph_v0_6/graph_gcn_gru_v0_6.pt":"c8d104c5512556dbbcb459eb63ccef8ba8e71cb04fc9b22a5b16c23eee51e0af",
"results/net3/clean_test_v0_7b/heldout_test_predictions.csv":"9e75780befaf5502bd38c3c02a1a1b62a3ec8c972eb8eab6d1be4b552fb040ce",
"results/net3/final_v0_12/heldout_degradation_predictions.csv":"ace62010843176a1ee94f877810c0d889b04dcf713981e48011589dcc8f832c5",
"results/net3/final_v0_12/heldout_controller_outcomes.csv":"806bcee09b11105ab9ee7d88282dc9eec90fbe24ce63a105e97947c545ce7104",
}
REQUIRED=["README.md","DATA.md","REPRODUCIBILITY.md","report/method.md","report/results.md","report/failure_analysis.md","report/limitations.md","src/net3/v0_12/01_final_degraded_test_v0_12.py","src/battledim/v0_17/01_run_final_external_2019.py","results/battledim/v0_17/BattLeDIM_final_external_2019_summary_v0_17.txt","results/battledim/v0_18/transfer_failure_diagnostic_summary_v0_18.txt"]
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
errors=[]
for rel in REQUIRED:
    if not (ROOT/rel).exists(): errors.append('missing required file: '+rel)
for rel,exp in KNOWN.items():
    p=ROOT/rel
    if not p.exists(): errors.append('missing frozen artifact: '+rel); continue
    got=sha256(p)
    if got!=exp: errors.append(f'hash mismatch: {rel} expected={exp} actual={got}')
if errors:
    print('INTEGRITY: FAIL')
    [print('-',e) for e in errors]
    sys.exit(1)
print('INTEGRITY: PASS')
print(f'Verified {len(REQUIRED)} required files and {len(KNOWN)} frozen scientific artifact hashes.')
