
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

# ----------------------------------------------------------------------
# Frozen hashes
# ----------------------------------------------------------------------

WINDOW_HASHES = {
    "window_manifest.csv": "ee67fa17f169b9ac3d8b48efa49a6e634dc5ca405dca657ea5f82043c05a301d",
    "normalization_train_only.json": "a6a4bf24c805e9e02a635cd363b006aab0830ab925fe70b7c9dbe0b0f5e3e1b8",
    "window_config.json": "d6ff1aa93bf5d607b648d1e6c62c6a6a2d2728204161b372e3ea5d56974cb1a1",
}
GENERATED_HASHES = {
    "generation_index.csv": "bc33af176da705ed87850e6032dc313761e2c839faf04aa03c59bd12deb5848a",
    "tensor_layout.json": "33220279d7e1ff6c92caacd54785040764168c576d5eb218401515be206a9261",
    "graph_nodes.csv": "dfb90c5fa1b7c80935edcca5294ec0b87b22cb3788732f4432d58135c8e6f295",
    "graph_links.csv": "d92ab147d4678c8aff999a86f761f15deeb843a738b3021acc3062b824860c1e",
}
BASELINE_HASH = "3e83192de687c42db771964aa82715b6221ad8dc89423484115a9b6aed23a040"
GRAPH_HASH = "c8d104c5512556dbbcb459eb63ccef8ba8e71cb04fc9b22a5b16c23eee51e0af"

CAL_PRED_HASH = "36edd2accc9c990d5806a48079348f18319554bc86fa08717597f29df3a2429e"
CAL_CORRUPTION_HASH = "df467191e0acc4b257ad6ae8002468d10943352e5b741e9f1d8d34519d0117a8"

ATTR_PROFILES_HASH = "3b5d1e10b8c5c0bab327e74e8d47a0c21466bb9f24751efa9a823702f709b9d1"
ATTR_INDEX_HASH = "3ee003c17dcd5a655b28abfbe86b510907cdd040717e7b52b6e632205dc7ff79"
ATTR_DETECTION_HASH = "248aec709ea357fe52cf68b62b07d3ba8d4e49ce864a55f6a80baa174069750f"
ATTR_CENTROIDS_HASH = "98c54574ef8a706fffe6d073017347745f2f6c07b4633bcb806e8cb94d162927"
ATTR_EXECUTION_HASH = "c7d6933f782dbf8b266f29b93ba9bd76e76d00ca71981116e5e8553ac66b1786"

BASELINE_THRESHOLD = 0.464201702923
GRAPH_THRESHOLD = 0.376129878873
RIDGE_ALPHA = 1.0
CLEAN_QUANTILE = 0.95
PROFILE_EPS = 1e-12

MLP_H1, MLP_H2 = 256, 64
NODE_FEATURES, GCN_HIDDEN, GRU_HIDDEN, HEAD_HIDDEN = 10, 32, 64, 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected, label=None):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing artifact: {path}")
    got = sha256(path)
    if got != expected:
        raise SystemExit(
            f"HASH MISMATCH {label or path.name}\nexpected={expected}\ngot={got}"
        )
    print(f"[OK] {label or path.name}")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(792, MLP_H1),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(MLP_H1, MLP_H2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(MLP_H2, 1),
        )
    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Linear(i, o)
    def forward(self, x, A):
        return self.linear(torch.einsum("ij,btjf->btif", A, x))


class GraphTemporalModel(nn.Module):
    def __init__(self, A_norm):
        super().__init__()
        self.register_buffer("A_norm", torch.tensor(A_norm, dtype=torch.float32))
        self.gcn1 = GCNLayer(NODE_FEATURES, GCN_HIDDEN)
        self.gcn2 = GCNLayer(GCN_HIDDEN, GCN_HIDDEN)
        self.dropout = nn.Dropout(0.20)
        self.gru = nn.GRU(GCN_HIDDEN*2, GRU_HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(HEAD_HIDDEN, 1),
        )
    def forward(self, x):
        h = self.dropout(torch.relu(self.gcn1(x, self.A_norm)))
        h = self.dropout(torch.relu(self.gcn2(h, self.A_norm)))
        z = torch.cat([h.mean(2), h.max(2).values], dim=-1)
        o, _ = self.gru(z)
        return self.head(o[:, -1]).squeeze(-1)


def build_graph(gen, layout):
    nodes = pd.read_csv(gen/"graph_nodes.csv", dtype={"node_id":str})
    links = pd.read_csv(
        gen/"graph_links.csv",
        dtype={"link_id":str,"start_node":str,"end_node":str},
    )
    ids = [str(x) for x in layout["original_node_ids"]]
    ix = {n:i for i,n in enumerate(ids)}
    A = np.zeros((len(ids),len(ids)),dtype=np.float32)
    endpoints = {}
    for _,r in links.iterrows():
        u,v=ix[str(r["start_node"])],ix[str(r["end_node"])]
        A[u,v]=A[v,u]=1.0
        endpoints[str(r["link_id"])]=(u,v)
    Ah=A+np.eye(len(ids),dtype=np.float32)
    d=Ah.sum(1); q=np.zeros_like(d); m=d>0
    q[m]=np.power(d[m],-.5).astype(np.float32)
    An=q[:,None]*Ah*q[None,:]

    degree=A.sum(1)
    node_type={str(r["node_id"]):str(r["node_type"]).lower() for _,r in nodes.iterrows()}
    static=np.zeros((len(ids),4),dtype=np.float32)
    static[:,0]=degree/max(float(degree.max()),1.0)
    for n,i in ix.items():
        t=node_type.get(n,"")
        static[i,1]=1.0 if "junction" in t else 0.0
        static[i,2]=1.0 if "tank" in t else 0.0
        static[i,3]=1.0 if "reservoir" in t else 0.0

    pmap=[];fmap=[];tmap=[]
    for fi,name in enumerate(layout["sensor_feature_order"]):
        if name.startswith("pressure::"):
            nid=name.split("::",1)[1];pmap.append((fi,nid,ix[nid]))
        elif name.startswith("flow::"):
            lid=name.split("::",1)[1];fmap.append((fi,lid,*endpoints[lid]))
        elif name.startswith("tank_level::"):
            nid=name.split("::",1)[1];tmap.append((fi,nid,ix[nid]))
    return ids,An,static,pmap,fmap,tmap


class GraphRawWrapper(nn.Module):
    def __init__(self, graph_model, n_nodes, static, pmap, fmap, tmap, missing_fis=None):
        super().__init__()
        missing=set(missing_fis or [])
        P=np.zeros((33,n_nodes),np.float32)
        T=np.zeros((33,n_nodes),np.float32)
        F=np.zeros((33,n_nodes),np.float32)
        pmask=np.zeros(n_nodes,np.float32)
        tmask=np.zeros(n_nodes,np.float32)
        fcount=np.zeros(n_nodes,np.float32)

        for fi,sid,ni in pmap:
            P[fi,ni]=1.0;pmask[ni]=0.0 if fi in missing else 1.0
        for fi,sid,ni in tmap:
            T[fi,ni]=1.0;tmask[ni]=1.0
        for fi,lid,u,v in fmap:
            F[fi,u]-=1.0;F[fi,v]+=1.0;fcount[u]+=1.0;fcount[v]+=1.0

        self.graph_model=graph_model
        self.register_buffer("P",torch.tensor(P))
        self.register_buffer("T",torch.tensor(T))
        self.register_buffer("F",torch.tensor(F))
        self.register_buffer("pmask",torch.tensor(pmask))
        self.register_buffer("tmask",torch.tensor(tmask))
        self.register_buffer("fcount",torch.tensor(fcount))
        self.register_buffer("static",torch.tensor(static,dtype=torch.float32))

    def forward(self,x):
        pv=torch.einsum("btf,fn->btn",x,self.P)
        tv=torch.einsum("btf,fn->btn",x,self.T)
        fv=torch.einsum("btf,fn->btn",x,self.F)
        B,T,_=pv.shape
        dyn=torch.stack([
            pv,
            self.pmask[None,None,:].expand(B,T,-1),
            tv,
            self.tmask[None,None,:].expand(B,T,-1),
            fv,
            self.fcount[None,None,:].expand(B,T,-1),
        ],dim=-1)
        st=self.static[None,None,:,:].expand(B,T,-1,-1)
        return self.graph_model(torch.cat([dyn,st],dim=-1))


def sigmoid(z):
    z=np.asarray(z,dtype=np.float64)
    return 1/(1+np.exp(-np.clip(z,-60,60)))


def entropy01(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-12,1-1e-12)
    return -(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2.0)


def js_divergence(p,q):
    p=np.asarray(p,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    p=np.clip(p,PROFILE_EPS,None);q=np.clip(q,PROFILE_EPS,None)
    p=p/p.sum(axis=-1,keepdims=True);q=q/q.sum(axis=-1,keepdims=True)
    m=.5*(p+q)
    return .5*np.sum(p*np.log(p/m),axis=-1)+.5*np.sum(q*np.log(q/m),axis=-1)


def input_x_gradient(wrapper,X,batch):
    wrapper.eval()
    for p in wrapper.parameters():p.requires_grad_(False)
    profiles=[];logits=[]
    for i in range(0,len(X),batch):
        xb=torch.tensor(X[i:i+batch],dtype=torch.float32,requires_grad=True)
        z=wrapper(xb)
        g=torch.autograd.grad(z.sum(),xb)[0]
        a=torch.abs(xb*g).sum(1)+PROFILE_EPS
        a=a/a.sum(1,keepdim=True)
        profiles.append(a.detach().numpy())
        logits.append(z.detach().numpy())
    return np.concatenate(profiles),np.concatenate(logits)


def parse_ids(v):
    if pd.isna(v) or str(v).strip()=="":return []
    return str(v).split(",")


def fit_reconstructors(train_context_ids, gen, mean, scale, pressure_feature_indices):
    rows=[]
    for cid in sorted(train_context_ids):
        with np.load(gen/"contexts"/f"{cid}.npz",allow_pickle=False) as z:
            for key in ["clean_sensor","leak_sensor"]:
                x=(z[key].astype(np.float32)-mean)/scale
                rows.append(x)
    X=np.concatenate(rows,axis=0)
    if X.shape!=(86700,33):
        raise RuntimeError(f"Expected training reconstruction matrix (86700,33), got {X.shape}")

    models={}
    quality=[]
    all_idx=np.arange(33)
    for fi in pressure_feature_indices:
        keep=all_idx[all_idx!=fi]
        model=Ridge(alpha=RIDGE_ALPHA,fit_intercept=True)
        model.fit(X[:,keep],X[:,fi])
        pred=model.predict(X[:,keep])
        models[int(fi)]={
            "coef":model.coef_.astype(np.float64),
            "intercept":float(model.intercept_),
            "keep":keep.astype(int),
        }
        quality.append({
            "feature_index":int(fi),
            "train_rmse_normalized":float(mean_squared_error(X[:,fi],pred)**0.5),
            "train_r2":float(r2_score(X[:,fi],pred)),
        })
    return models,pd.DataFrame(quality)


def reconstruct_normalized_sequence(x_missing, missing_fis, recon):
    # Simultaneous one-pass reconstruction from the same pre-reconstruction input.
    base=x_missing.copy()
    out=x_missing.copy()
    for fi in missing_fis:
        r=recon[int(fi)]
        out[:,fi]=base[:,r["keep"]]@r["coef"]+r["intercept"]
    return out


def action_from_signals(ent,attr,ent_thr,attr_thr):
    eh=ent>ent_thr; ah=attr>attr_thr
    if not eh and not ah:return "PROCEED"
    if eh and not ah:return "VERIFY"
    if not eh and ah:return "REQUEST_SENSOR"
    return "ESCALATE"


def main():
    ap=argparse.ArgumentParser()
    for arg in ["--windows","--generated","--baseline","--graph",
                "--calibration-degradations","--calibration-attribution","--out"]:
        ap.add_argument(arg,required=True)
    a=ap.parse_args()

    win=Path(a.windows);gen=Path(a.generated);bd=Path(a.baseline);gd=Path(a.graph)
    cd=Path(a.calibration_degradations);ca=Path(a.calibration_attribution);out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)

    print("AQUA-VERA RELIABILITY CONTROLLER FREEZE v0.11")
    print("="*78)

    for n,h in WINDOW_HASHES.items():verify(win/n,h)
    for n,h in GENERATED_HASHES.items():verify(gen/n,h)
    verify(bd/"baseline_mlp_v0_5.pt",BASELINE_HASH,"frozen MLP")
    verify(gd/"graph_gcn_gru_v0_6.pt",GRAPH_HASH,"frozen GCN-GRU")
    verify(cd/"calibration_degradation_predictions.csv",CAL_PRED_HASH)
    verify(cd/"calibration_corruption_manifest.csv",CAL_CORRUPTION_HASH)
    verify(ca/"calibration_attribution_profiles.npz",ATTR_PROFILES_HASH)
    verify(ca/"calibration_attribution_index.csv",ATTR_INDEX_HASH)
    verify(ca/"calibration_attribution_detection.csv",ATTR_DETECTION_HASH)
    verify(ca/"calibration_attribution_centroids.json",ATTR_CENTROIDS_HASH)
    verify(ca/"calibration_attribution_execution.json",ATTR_EXECUTION_HASH)

    manifest=pd.read_csv(win/"window_manifest.csv")
    train=manifest[manifest.split=="train"]
    cal=manifest[manifest.split=="calibration"]
    if train.context_id.nunique()!=150 or cal.context_id.nunique()!=30:
        raise SystemExit("split context count mismatch")
    if (manifest.split=="test").sum()!=2700:
        raise SystemExit("frozen test manifest mismatch")
    print("[BOUNDARY] degraded test not loaded")

    norm=json.loads((win/"normalization_train_only.json").read_text())
    mean=np.array(norm["mean"],np.float32);scale=np.array(norm["scale_used"],np.float32)
    layout=json.loads((gen/"tensor_layout.json").read_text())
    node_ids,An,static,pmap,fmap,tmap=build_graph(gen,layout)
    pressure_fis=[fi for fi,_,_ in pmap]
    p_id_to_fi={sid:fi for fi,sid,_ in pmap}

    # ------------------------------------------------------------------
    # Freeze thresholds from CLEAN calibration only.
    # ------------------------------------------------------------------
    pred=pd.read_csv(cd/"calibration_degradation_predictions.csv")
    attr=pd.read_csv(ca/"calibration_attribution_index.csv")
    centroids=json.loads((ca/"calibration_attribution_centroids.json").read_text())

    thresholds={}
    for model in ["MLP","GCN_GRU"]:
        clean_full=pred[(pred.model==model)&(pred.family=="CLEAN")]
        clean_attr=attr[(attr.model==model)&(attr.family=="CLEAN")]
        thresholds[model]={
            "entropy_full_q95":float(np.quantile(clean_full.predictive_entropy,CLEAN_QUANTILE)),
            "entropy_attr_subset_q95":float(np.quantile(clean_attr.predictive_entropy,CLEAN_QUANTILE)),
            "attribution_divergence_q95":float(np.quantile(clean_attr.attribution_profile_divergence,CLEAN_QUANTILE)),
            "clean_quantile":CLEAN_QUANTILE,
            "source":"uncorrupted calibration only",
        }

    # ------------------------------------------------------------------
    # Train pressure reconstruction from TRAIN unique rows only.
    # ------------------------------------------------------------------
    print("[RECONSTRUCTION] fitting 24 training-only ridge models")
    recon,recon_quality=fit_reconstructors(
        sorted(train.context_id.unique()),gen,mean,scale,pressure_fis
    )

    recon_json={}
    for fi,r in recon.items():
        recon_json[str(fi)]={
            "keep_indices":[int(x) for x in r["keep"]],
            "coef":[float(x) for x in r["coef"]],
            "intercept":float(r["intercept"]),
        }
    recon_path=out/"pressure_reconstruction_ridge.json"
    recon_path.write_text(json.dumps({
        "alpha":RIDGE_ALPHA,
        "training_rows":86700,
        "training_split":"train only",
        "simultaneous_multi_missing_rule":"one pass from mean-imputed pre-reconstruction vector",
        "gcn_missing_mask_after_reconstruction":0,
        "models":recon_json,
    },indent=2))
    recon_quality_path=out/"pressure_reconstruction_train_quality.csv"
    recon_quality.to_csv(recon_quality_path,index=False)

    # Models
    bc=torch.load(bd/"baseline_mlp_v0_5.pt",map_location="cpu",weights_only=False)
    mlp=MLP();mlp.load_state_dict(bc["model_state_dict"],strict=True);mlp.eval()
    gc=torch.load(gd/"graph_gcn_gru_v0_6.pt",map_location="cpu",weights_only=False)
    graph=GraphTemporalModel(An);graph.load_state_dict(gc["model_state_dict"],strict=True);graph.eval()

    # ------------------------------------------------------------------
    # Missing-sensor reconstruction calibration evaluation on the exact
    # attribution subset only. Other conditions require no recovery model.
    # ------------------------------------------------------------------
    corr=pd.read_csv(cd/"calibration_corruption_manifest.csv")
    attr_missing=attr[attr.family=="MISSING_SENSOR"].copy()
    selected_ids=set(attr_missing.window_id)

    # Obtain frozen subset rows from manifest, exactly same IDs.
    sub=cal[cal.window_id.isin(selected_ids)].copy()
    raw={}
    for cid in sorted(sub.context_id.unique()):
        with np.load(gen/"contexts"/f"{cid}.npz",allow_pickle=False) as z:
            raw[(cid,"CLEAN")]=z["clean_sensor"].astype(np.float32)
            raw[(cid,"LEAK")]=z["leak_sensor"].astype(np.float32)

    recovery_rows=[]
    max_model_prob_check=0.0

    for sev in ["low","medium","high"]:
        for cid,grp in sub.groupby("context_id",sort=True):
            cm=corr[(corr.context_id==cid)&(corr.family=="MISSING_SENSOR")&(corr.severity==sev)]
            if len(cm)!=1:raise RuntimeError("missing corruption row")
            selected=parse_ids(cm.iloc[0].selected_pressure_ids)
            missing_fis=[p_id_to_fi[s] for s in selected]

            windows=[];rows=[]
            for _,r in grp.sort_values(["variant","final_idx"]).iterrows():
                arr=raw[(cid,str(r.variant))]
                x=(arr-mean)/scale
                x_missing=x.copy()
                x_missing[:,missing_fis]=0.0
                x_rec=reconstruct_normalized_sequence(x_missing,missing_fis,recon)
                s,f=int(r.start_idx),int(r.final_idx)
                windows.append(x_rec[s:f+1].astype(np.float32));rows.append(r)
            X=np.stack(windows)
            rows=pd.DataFrame(rows).reset_index(drop=True)

            # MLP recovered attribution/probability
            prof,z=input_x_gradient(mlp,X,128);p=sigmoid(z)
            for j,r in rows.iterrows():
                pred_class=int(p[j]>=BASELINE_THRESHOLD)
                c=np.array(centroids["MLP"][str(pred_class)]["profile"],dtype=float)
                div=float(js_divergence(prof[j:j+1],c[None,:])[0])
                ent=float(entropy01([p[j]])[0])
                action=action_from_signals(
                    ent,div,
                    thresholds["MLP"]["entropy_attr_subset_q95"],
                    thresholds["MLP"]["attribution_divergence_q95"],
                )
                recovery_rows.append({
                    "model":"MLP","severity":sev,"window_id":r.window_id,
                    "context_id":cid,"variant":r.variant,
                    "target_active_leak":int(r.target_active_leak),
                    "reconstructed_probability":float(p[j]),
                    "reconstructed_prediction":pred_class,
                    "reconstructed_correct":int(pred_class==int(r.target_active_leak)),
                    "reconstructed_entropy":ent,
                    "reconstructed_attribution_divergence":div,
                    "post_reconstruction_action":action,
                })

            # GCN recovered attribution/probability; missing mask remains 0.
            wrapper=GraphRawWrapper(
                graph,len(node_ids),static,pmap,fmap,tmap,missing_fis=missing_fis
            )
            prof,z=input_x_gradient(wrapper,X,32);p=sigmoid(z)
            for j,r in rows.iterrows():
                pred_class=int(p[j]>=GRAPH_THRESHOLD)
                c=np.array(centroids["GCN_GRU"][str(pred_class)]["profile"],dtype=float)
                div=float(js_divergence(prof[j:j+1],c[None,:])[0])
                ent=float(entropy01([p[j]])[0])
                action=action_from_signals(
                    ent,div,
                    thresholds["GCN_GRU"]["entropy_attr_subset_q95"],
                    thresholds["GCN_GRU"]["attribution_divergence_q95"],
                )
                recovery_rows.append({
                    "model":"GCN_GRU","severity":sev,"window_id":r.window_id,
                    "context_id":cid,"variant":r.variant,
                    "target_active_leak":int(r.target_active_leak),
                    "reconstructed_probability":float(p[j]),
                    "reconstructed_prediction":pred_class,
                    "reconstructed_correct":int(pred_class==int(r.target_active_leak)),
                    "reconstructed_entropy":ent,
                    "reconstructed_attribution_divergence":div,
                    "post_reconstruction_action":action,
                })

    recovery=pd.DataFrame(recovery_rows)
    recovery_path=out/"calibration_missing_reconstruction_outcomes.csv"
    recovery.to_csv(recovery_path,index=False)

    # ------------------------------------------------------------------
    # Freeze controller policy and evaluate descriptively on calibration
    # attribution subset.
    # ------------------------------------------------------------------
    policy_rows=[]
    recovery_lookup={
        (r.model,r.severity,r.window_id):r
        for _,r in recovery.iterrows()
    }

    for _,r in attr.iterrows():
        model=r.model
        ent_thr=thresholds[model]["entropy_attr_subset_q95"]
        attr_thr=thresholds[model]["attribution_divergence_q95"]

        if r.family=="MISSING_SENSOR":
            rr=recovery_lookup[(model,r.severity,r.window_id)]
            initial_action="RECONSTRUCT"
            final_action=rr.post_reconstruction_action
            final_prediction=int(rr.reconstructed_prediction)
            final_correct=int(rr.reconstructed_correct)
            final_entropy=float(rr.reconstructed_entropy)
            final_attr=float(rr.reconstructed_attribution_divergence)
        else:
            initial_action=action_from_signals(
                float(r.predictive_entropy),
                float(r.attribution_profile_divergence),
                ent_thr,attr_thr,
            )
            final_action=initial_action
            final_prediction=int(r.prediction)
            final_correct=int(r.correct)
            final_entropy=float(r.predictive_entropy)
            final_attr=float(r.attribution_profile_divergence)

        final_proceed = int(final_action=="PROCEED")
        unsafe_proceed = int(final_proceed==1 and final_correct==0)

        policy_rows.append({
            "model":model,"family":r.family,"severity":r.severity,
            "window_id":r.window_id,"context_id":r.context_id,
            "target_active_leak":int(r.target_active_leak),
            "initial_action":initial_action,
            "final_action":final_action,
            "final_prediction":final_prediction,
            "final_correct":final_correct,
            "final_proceed":final_proceed,
            "unsafe_proceed":unsafe_proceed,
            "final_entropy":final_entropy,
            "final_attribution_divergence":final_attr,
        })

    policy=pd.DataFrame(policy_rows)
    policy_path=out/"calibration_controller_outcomes.csv"
    policy.to_csv(policy_path,index=False)

    summary_rows=[]
    for (model,fam,sev),d in policy.groupby(["model","family","severity"],sort=True):
        summary_rows.append({
            "model":model,"family":fam,"severity":sev,"n":len(d),
            "proceed_rate":float(d.final_proceed.mean()),
            "unsafe_proceed_rate":float(d.unsafe_proceed.mean()),
            "accuracy_if_forced_prediction":float(d.final_correct.mean()),
            "reconstruct_rate":float((d.initial_action=="RECONSTRUCT").mean()),
            "verify_rate":float((d.final_action=="VERIFY").mean()),
            "request_sensor_rate":float((d.final_action=="REQUEST_SENSOR").mean()),
            "escalate_rate":float((d.final_action=="ESCALATE").mean()),
        })
    controller_metrics=pd.DataFrame(summary_rows)
    metrics_path=out/"calibration_controller_metrics.csv"
    controller_metrics.to_csv(metrics_path,index=False)

    # Overall by model/evidence state
    overall=[]
    for model in ["MLP","GCN_GRU"]:
        for state,mask in [
            ("clean",policy.family=="CLEAN"),
            ("degraded",policy.family!="CLEAN"),
        ]:
            d=policy[(policy.model==model)&mask]
            overall.append({
                "model":model,"state":state,"n":len(d),
                "proceed_rate":float(d.final_proceed.mean()),
                "unsafe_proceed_rate":float(d.unsafe_proceed.mean()),
                "forced_prediction_accuracy":float(d.final_correct.mean()),
            })
    overall_df=pd.DataFrame(overall)
    overall_path=out/"calibration_controller_overall.csv"
    overall_df.to_csv(overall_path,index=False)

    thresholds_path=out/"FROZEN_CONTROLLER_THRESHOLDS.json"
    thresholds_path.write_text(json.dumps({
        "status":"FROZEN",
        "selection_rule":"95th percentile of uncorrupted calibration signal",
        "clean_quantile":CLEAN_QUANTILE,
        "thresholds":thresholds,
        "action_mapping":{
            "entropy_low_attribution_low":"PROCEED",
            "entropy_high_attribution_low":"VERIFY",
            "entropy_low_attribution_high":"REQUEST_SENSOR",
            "entropy_high_attribution_high":"ESCALATE",
            "known_missing_sensor_mask":"RECONSTRUCT first, then apply same dual-signal mapping",
        },
        "missing_reconstruction":{
            "method":"per-pressure-sensor Ridge regression",
            "alpha":RIDGE_ALPHA,
            "training_source":"86,700 unique TRAIN rows only",
            "multi_missing":"simultaneous one-pass",
            "GCN_observation_mask_after_reconstruction":0,
        },
        "test_or_degraded_test_used_for_threshold_selection":False,
    },indent=2))

    artifacts=[
        recon_path,recon_quality_path,recovery_path,policy_path,
        metrics_path,overall_path,thresholds_path
    ]
    hashes={p.name:sha256(p) for p in artifacts}
    record={
        "status":"RELIABILITY_CONTROLLER_FROZEN_PRE_DEGRADED_TEST",
        "artifact_sha256":hashes,
        "degraded_test_loaded":False,
        "models_retrained":False,
        "clean_model_thresholds_changed":False,
        "controller_thresholds_source":"clean calibration only",
    }
    (out/"FROZEN_CONTROLLER_RECORD.json").write_text(json.dumps(record,indent=2))

    # compact useful results
    rec_stats=recovery.groupby(["model","severity"]).agg(
        n=("window_id","size"),
        reconstruction_accuracy=("reconstructed_correct","mean"),
        post_reconstruction_proceed=("post_reconstruction_action",lambda x:float((x=="PROCEED").mean())),
    ).reset_index()

    lines=[
        "AQUA-VERA RELIABILITY CONTROLLER FREEZE v0.11",
        "="*78,
        "STATUS: PASS / CONTROLLER FROZEN PRE-DEGRADED-TEST",
        "",
        "BOUNDARY",
        "  reconstruction training source: TRAIN ONLY",
        "  controller threshold source: CLEAN CALIBRATION ONLY",
        "  degraded TEST loaded: NO",
        "  clean models retrained: NO",
        "  clean model thresholds changed: NO",
        "",
        "THRESHOLD RULE",
        "  95th percentile of clean calibration signal",
    ]
    for model in ["MLP","GCN_GRU"]:
        t=thresholds[model]
        lines += [
            f"  {model} entropy q95 (attribution subset): {t['entropy_attr_subset_q95']:.9f}",
            f"  {model} attribution-divergence q95: {t['attribution_divergence_q95']:.9f}",
        ]
    lines += [
        "",
        "ACTION MAPPING",
        "  low entropy + low attribution divergence -> PROCEED",
        "  high entropy only -> VERIFY",
        "  high attribution divergence only -> REQUEST_SENSOR",
        "  both high -> ESCALATE",
        "  explicit missing-sensor mask -> RECONSTRUCT, then re-evaluate",
        "",
        "RECONSTRUCTION",
        "  24 pressure-sensor Ridge models",
        "  alpha: 1.0",
        "  unique training rows: 86,700",
        "  simultaneous one-pass for multiple missing sensors",
        "  GCN missing observation mask remains 0 after reconstruction",
        f"  train R2 min/median/max: "
        f"{recon_quality.train_r2.min():.6f} / "
        f"{recon_quality.train_r2.median():.6f} / "
        f"{recon_quality.train_r2.max():.6f}",
        "",
        "CALIBRATION MISSING-SENSOR RECOVERY (ATTRIBUTION SUBSET)",
    ]
    for _,r in rec_stats.iterrows():
        lines.append(
            f"  {r.model} {r.severity}: "
            f"forced accuracy={r.reconstruction_accuracy:.6f}, "
            f"post-reconstruction proceed={r.post_reconstruction_proceed:.6f}"
        )
    lines += [
        "",
        "CALIBRATION CONTROLLER OVERALL (ATTRIBUTION SUBSET)",
    ]
    for _,r in overall_df.iterrows():
        lines.append(
            f"  {r.model} {r.state}: n={int(r.n)}, "
            f"proceed={r.proceed_rate:.6f}, "
            f"unsafe_proceed={r.unsafe_proceed_rate:.6f}, "
            f"forced_accuracy={r.forced_prediction_accuracy:.6f}"
        )
    lines += ["","ARTIFACT HASHES"]
    for n,h in hashes.items():lines.append(f"  {n}: {h}")
    lines += [
        "",
        "NEXT",
        "  Retain controller_freeze_summary.txt with the project record.",
        "  If PASS, reliability rules are frozen and degraded held-out test",
        "  may be generated/evaluated exactly once.",
    ]
    summary=out/"controller_freeze_summary.txt"
    summary.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__=="__main__":
    main()
