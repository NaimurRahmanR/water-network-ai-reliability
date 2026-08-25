
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

H18S="00f56ce42642307cb4c267bd48166c1c1272a96cb8e8f24d3f17dcda09599b6e"
H19S="a021ade90c3cfa17362520f0fd7e9794a6c816e4787d7e49a3e52fdd6aa0d15d"
H18P="3d972603e72f194104a5f2a32df8bc80052b452267b69b0650ce45cddd1ef6c8"
H19P="bd688f47767be93ffbab771b71773d48db7341523b0469c422a8df3bce027efc"
H19W="078b1fa238469047b7f4dc199f2fb11e871735d25ccd52e2af56c8ecb09652eb"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def find(root,name):
    root=Path(root); d=root/name
    if d.exists(): return d
    m=list(root.rglob(name))
    if len(m)!=1: raise SystemExit(f"Expected one {name} under {root}; found {len(m)}")
    return m[0]

def verify(p,h,label):
    if not Path(p).exists(): raise SystemExit(f"Missing {label}: {p}")
    g=sha(p)
    if g!=h: raise SystemExit(f"{label} hash mismatch\nexpected={h}\nactual={g}")
    print(f"[OK] {label}")

def ks(a,b):
    a=np.sort(np.asarray(a,float)); b=np.sort(np.asarray(b,float))
    v=np.sort(np.unique(np.r_[a,b]))
    return float(np.max(np.abs(np.searchsorted(a,v,"right")/len(a)-np.searchsorted(b,v,"right")/len(b))))

def psi(a,b,n=10):
    a=np.asarray(a,float); b=np.asarray(b,float)
    cuts=np.unique(np.quantile(a,np.linspace(0,1,n+1)))
    if len(cuts)<3: return 0.0
    cuts[0],cuts[-1]=-np.inf,np.inf
    ar,_=np.histogram(a,bins=cuts); br,_=np.histogram(b,bins=cuts)
    ap=np.clip(ar/ar.sum(),1e-6,None); bp=np.clip(br/br.sum(),1e-6,None)
    return float(np.sum((bp-ap)*np.log(bp/ap)))

def ece(y,p,n=15):
    y=np.asarray(y,int); p=np.asarray(p,float); edges=np.linspace(0,1,n+1); out=0.
    for i in range(n):
        m=(p>=edges[i]) & ((p<=edges[i+1]) if i==n-1 else (p<edges[i+1]))
        if m.any(): out += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-ready",required=True)
    ap.add_argument("--train-freeze",required=True)
    ap.add_argument("--external-2019",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    mr,tf,ex,out=map(Path,[a.model_ready,a.train_freeze,a.external_2019,a.out]); out.mkdir(parents=True,exist_ok=True)

    print("AQUA-VERA EXTERNAL TRANSFER FAILURE DIAGNOSTIC v0.18")
    print("POST-HOC READ-ONLY. NO RETRAINING / RETUNING.\n")

    p18s=find(mr,"2018_scada_model_ready.csv"); p19s=find(mr,"2019_scada_model_ready.csv")
    p18p=tf/"2018_clean_predictions_v0_16b.csv"; p19p=ex/"2019_external_predictions.csv"; p19w=ex/"2019_external_window_manifest.csv"
    for p,h,l in [(p18s,H18S,"2018 SCADA"),(p19s,H19S,"2019 SCADA"),(p18p,H18P,"2018 predictions"),(p19p,H19P,"2019 predictions"),(p19w,H19W,"2019 windows")]: verify(p,h,l)

    s18=pd.read_csv(p18s); s19=pd.read_csv(p19s); q18=pd.read_csv(p18p); q19=pd.read_csv(p19p); w19=pd.read_csv(p19w)
    feats=[c for c in s18.columns if c!="Timestamp"]
    if feats != [c for c in s19.columns if c!="Timestamp"]: raise SystemExit("Feature order mismatch")

    dr=[]
    for c in feats:
        x=s18[c].to_numpy(float); y=s19[c].to_numpy(float)
        dr.append({"feature":c,"psi":psi(x,y),"ks":ks(x,y),"std_mean_shift":abs(y.mean()-x.mean())/max(x.std(),1e-12)})
    drift=pd.DataFrame(dr).sort_values(["psi","ks"],ascending=False)
    drift.to_csv(out/"feature_drift_2018_vs_2019.csv",index=False)

    clean=q19[q19["family"]=="CLEAN"].copy()
    if "target_recent_leak_onset_24h" not in clean.columns:
        t19=next(c for c in clean.columns if c.startswith("target") and "onset" in c)
    else: t19="target_recent_leak_onset_24h"
    t18="target_recent_leak_onset_24h"

    rows=[]
    for model,p18col in [("MLP","mlp_probability"),("GCN_GRU","gcn_probability")]:
        d=clean[clean["model"]==model]
        y18=q18[t18].to_numpy(int); p18=q18[p18col].to_numpy(float)
        y19=d[t19].to_numpy(int); p19=d["probability"].to_numpy(float)
        rows.append({
            "model":model,
            "prevalence_2018":y18.mean(),"prevalence_2019":y19.mean(),
            "auprc_2018_dev":average_precision_score(y18,p18),"auprc_2019_ext":average_precision_score(y19,p19),
            "auroc_2018_dev":roc_auc_score(y18,p18),"auroc_2019_ext":roc_auc_score(y19,p19),
            "brier_2018":brier_score_loss(y18,p18),"brier_2019":brier_score_loss(y19,p19),
            "ece15_2018":ece(y18,p18),"ece15_2019":ece(y19,p19),
            "probability_ks":ks(p18,p19),"probability_psi":psi(p18,p19),
            "mean_p_2018":p18.mean(),"mean_p_2019":p19.mean(),
            "predpos_2018":(p18>=0.5).mean(),"predpos_2019":(p19>=0.5).mean(),
        })
    transfer=pd.DataFrame(rows)
    transfer.to_csv(out/"clean_transfer_diagnostics.csv",index=False)

    clean["final_timestamp"]=pd.to_datetime(clean["final_timestamp"])
    clean["month"]=clean["final_timestamp"].dt.to_period("M").astype(str)
    mon=[]
    for (m,mo),g in clean.groupby(["model","month"]):
        y=g[t19].to_numpy(int); p=g["probability"].to_numpy(float)
        r={"model":m,"month":mo,"n":len(g),"positives":int(y.sum()),"prevalence":y.mean(),"mean_probability":p.mean(),"brier":brier_score_loss(y,p)}
        if len(np.unique(y))==2:
            r["auroc"]=roc_auc_score(y,p); r["auprc"]=average_precision_score(y,p)
        mon.append(r)
    pd.DataFrame(mon).to_csv(out/"monthly_clean_2019_performance.csv",index=False)

    # Event-window view: positive-window groups are separated by gaps > 30 min.
    ew=[]
    for model,g in clean.groupby("model"):
        d=g.sort_values("final_timestamp").copy()
        pos=d[d[t19]==1].copy()
        if len(pos):
            gap=pos["final_timestamp"].diff().gt(pd.Timedelta(minutes=30)).fillna(True)
            pos["event_group"]=gap.cumsum()
            for eg,h in pos.groupby("event_group"):
                det=h[h["prediction"]==1]
                ew.append({
                    "model":model,"event_group":int(eg),
                    "start":str(h["final_timestamp"].min()),"end":str(h["final_timestamp"].max()),
                    "positive_windows":len(h),"detected_windows":int((h["prediction"]==1).sum()),
                    "detected_fraction":(h["prediction"]==1).mean(),
                    "max_probability":h["probability"].max(),"mean_probability":h["probability"].mean(),
                    "any_threshold_detection":int(len(det)>0)
                })
    events=pd.DataFrame(ew)
    events.to_csv(out/"event_window_detection_2019.csv",index=False)

    lines=[
        "AQUA-VERA EXTERNAL TRANSFER FAILURE DIAGNOSTIC v0.18",
        "="*78,
        "STATUS: COMPLETE / POST-HOC READ-ONLY DIAGNOSTIC",
        "",
        "BOUNDARY",
        "  models retrained: NO",
        "  thresholds changed: NO",
        "  external result changed: NO",
        "",
        "FEATURE DRIFT 2018 -> 2019",
        f"  PSI >= 0.25: {(drift.psi>=0.25).sum()}/{len(drift)} features",
        f"  PSI median/max: {drift.psi.median():.6f} / {drift.psi.max():.6f}",
        f"  KS median/max: {drift.ks.median():.6f} / {drift.ks.max():.6f}",
        "",
        "CLEAN TRANSFER"
    ]
    for _,r in transfer.iterrows():
        lines += [
            f"  {r.model}: AUPRC {r.auprc_2018_dev:.6f} -> {r.auprc_2019_ext:.6f}; "
            f"AUROC {r.auroc_2018_dev:.6f} -> {r.auroc_2019_ext:.6f}",
            f"    ECE15 {r.ece15_2018:.6f} -> {r.ece15_2019:.6f}; "
            f"score KS/PSI {r.probability_ks:.6f}/{r.probability_psi:.6f}"
        ]
    lines += ["","EVENT-WINDOW DETECTION"]
    if len(events):
        for model,g in events.groupby("model"):
            lines.append(
                f"  {model}: groups={len(g)}, any detection={int(g.any_threshold_detection.sum())}/{len(g)}, "
                f"median detected fraction={g.detected_fraction.median():.6f}, "
                f"median max probability={g.max_probability.median():.6f}"
            )
    lines += ["","TOP 10 DRIFTED FEATURES"]
    for _,r in drift.head(10).iterrows():
        lines.append(f"  {r.feature}: PSI={r.psi:.6f}, KS={r.ks:.6f}, |mean shift|/SD18={r.std_mean_shift:.6f}")
    lines += ["","INTERPRETATION","  Diagnostic only; may characterize failure but must not be used to retune the frozen 2019 test.","","NEXT","  Retain transfer_failure_diagnostic_summary_v0_18.txt with the project record."]
    summary=out/"transfer_failure_diagnostic_summary_v0_18.txt"; summary.write_text("\n".join(lines),encoding="utf-8")

    rec={"status":"POST_HOC_DIAGNOSTIC_COMPLETE","version":"0.18","retraining":False,"threshold_changes":False,"external_result_changed":False}
    (out/"POST_HOC_DIAGNOSTIC_RECORD_v0_18.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__": main()
