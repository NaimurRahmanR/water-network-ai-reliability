from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def test_mlp(): assert sha256(ROOT/'results/net3/baseline_v0_5/baseline_mlp_v0_5.pt')=='3e83192de687c42db771964aa82715b6221ad8dc89423484115a9b6aed23a040'
def test_gcn(): assert sha256(ROOT/'results/net3/graph_v0_6/graph_gcn_gru_v0_6.pt')=='c8d104c5512556dbbcb459eb63ccef8ba8e71cb04fc9b22a5b16c23eee51e0af'
def test_predictions(): assert sha256(ROOT/'results/net3/final_v0_12/heldout_degradation_predictions.csv')=='ace62010843176a1ee94f877810c0d889b04dcf713981e48011589dcc8f832c5'
def test_controller(): assert sha256(ROOT/'results/net3/final_v0_12/heldout_controller_outcomes.csv')=='806bcee09b11105ab9ee7d88282dc9eec90fbe24ce63a105e97947c545ce7104'
