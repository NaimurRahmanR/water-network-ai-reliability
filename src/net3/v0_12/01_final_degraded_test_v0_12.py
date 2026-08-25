import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss

# Frozen source-script hashes from the exact packages already executed.
SRC09 = '719d8d88301ff8084e046711f2b05129ffa64dc962de3299efcab47c438a7f7b'
SRC10 = 'dac46065335e10d521db91d490c4c60c0a437b00c4ef6a7406171768eafc4403'
SRC11 = '19444c5f29f9541957d575c597ad55903d0a811fe5e377a097a07f543fb4f73c'

WINDOW = {
 'window_manifest.csv':'ee67fa17f169b9ac3d8b48efa49a6e634dc5ca405dca657ea5f82043c05a301d',
 'normalization_train_only.json':'a6a4bf24c805e9e02a635cd363b006aab0830ab925fe70b7c9dbe0b0f5e3e1b8',
 'window_config.json':'d6ff1aa93bf5d607b648d1e6c62c6a6a2d2728204161b372e3ea5d56974cb1a1b8'.replace('1b8','1a1'),
}
# Correct exact v0.4 hash (kept separate to avoid accidental transcription drift above).
WINDOW['window_config.json']='d6ff1aa93bf5d607b648d1e6c62c6a6a2d2728204161b372e3ea5d56974cb1a1'
GENERATED={
 'generation_index.csv':'bc33af176da705ed87850e6032dc313761e2c839faf04aa03c59bd12deb5848a',
 'tensor_layout.json':'33220279d7e1ff6c92caacd54785040764168c576d5eb218401515be206a9261',
 'graph_nodes.csv':'dfb90c5fa1b7c80935edcca5294ec0b87b22cb3788732f4432d58135c8e6f295',
 'graph_links.csv':'d92ab147d4678c8aff999a86f761f15deeb843a738b3021acc3062b824860c1e',
}
BASE='3e83192de687c42db771964aa82715b6221ad8dc89423484115a9b6aed23a040'
GRAPH='c8d104c5512556dbbcb459eb63ccef8ba8e71cb04fc9b22a5b16c23eee51e0af'
CLEAN='9e75780befaf5502bd38c3c02a1a1b62a3ec8c972eb8eab6d1be4b552fb040ce'
CALCORR='df467191e0acc4b257ad6ae8002468d10943352e5b741e9f1d8d34519d0117a8'
CENTROIDS='98c54574ef8a706fffe6d073017347745f2f6c07b4633bcb806e8cb94d162927'
RECON='e7d871b91ad8e90dbb696497738436a88a4169574340c8be15a41d4b4afe5dcb'
CTRL='6b47c2b842a43166558cce1741ad99f069e41fe226b98342fc672ad7c14b5c81'
BT=0.464201702923; GT=0.376129878873; TOL=2e-6
SEV=('low','medium','high')
FAMS=('MISSING_SENSOR','SENSOR_NOISE','SENSOR_BIAS','STALE_SENSOR','TOPOLOGY_MISMATCH','CROSS_SOURCE_CONFLICT')
NUMF=('MISSING_SENSOR','SENSOR_NOISE','SENSOR_BIAS','STALE_SENSOR','CROSS_SOURCE_CONFLICT')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1048576),b''): h.update(c)
 return h.hexdigest()

def check(p,e,label=None):
 p=Path(p)
 if not p.exists(): raise SystemExit(f'Missing: {p}')
 g=sha(p)
 if g!=e: raise SystemExit(f'HASH MISMATCH {label or p.name}\nexpected={e}\ngot={g}')
 print('[OK]',label or p.name)

def loadmod(name,path,expected):
 check(path,expected,f'frozen source {name}')
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def parse_ids(v):
 if pd.isna(v) or str(v).strip()=='': return []
 return str(v).split(',')

def metrics(y,p,t):
 q=(p>=t).astype(int)
 return {'n':int(len(y)),'accuracy':float((q==y).mean()),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'f1':float(f1_score(y,q,zero_division=0)),'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}

def main():
 ap=argparse.ArgumentParser()
 for x in ['--windows','--generated','--baseline','--graph','--clean-test','--calibration-degradations','--calibration-attribution','--controller','--v09-script','--v10-script','--v11-script','--out']: ap.add_argument(x,required=True)
 a=ap.parse_args(); w=Path(a.windows); g=Path(a.generated); bd=Path(a.baseline); gd=Path(a.graph); clean_dir=Path(a.clean_test); cd=Path(a.calibration_degradations); ca=Path(a.calibration_attribution); ctrl_dir=Path(a.controller); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 print('AQUA-VERA FINAL DEGRADED HELD-OUT TEST v0.12'); print('='*78)

 # Verify exact prior implementation sources and artifacts BEFORE degraded test generation.
 v09=loadmod('aq_v09',a.v09_script,SRC09); v10=loadmod('aq_v10',a.v10_script,SRC10); v11=loadmod('aq_v11',a.v11_script,SRC11)
 for n,h in WINDOW.items(): check(w/n,h)
 for n,h in GENERATED.items(): check(g/n,h)
 check(bd/'baseline_mlp_v0_5.pt',BASE,'frozen MLP'); check(gd/'graph_gcn_gru_v0_6.pt',GRAPH,'frozen GCN-GRU')
 check(clean_dir/'heldout_test_predictions.csv',CLEAN,'frozen clean held-out predictions')
 check(cd/'calibration_corruption_manifest.csv',CALCORR); check(ca/'calibration_attribution_centroids.json',CENTROIDS)
 check(ctrl_dir/'pressure_reconstruction_ridge.json',RECON); check(ctrl_dir/'FROZEN_CONTROLLER_THRESHOLDS.json',CTRL)
 cr=ctrl_dir/'FROZEN_CONTROLLER_RECORD.json'
 if not cr.exists(): raise SystemExit('Missing FROZEN_CONTROLLER_RECORD.json')
 rr=json.loads(cr.read_text())
 if rr.get('status')!='RELIABILITY_CONTROLLER_FROZEN_PRE_DEGRADED_TEST' or rr.get('degraded_test_loaded') is not False: raise SystemExit('Controller freeze record invalid')
 print('[OK] controller freeze record')

 manifest=pd.read_csv(w/'window_manifest.csv'); cal=manifest[manifest.split=='calibration'].copy(); test=manifest[manifest.split=='test'].copy().reset_index(drop=True)
 cal['context_id']=cal.context_id.astype(str); test['context_id']=test.context_id.astype(str)
 if len(test)!=2700 or test.context_id.nunique()!=30: raise SystemExit('Held-out split mismatch')
 norm=json.loads((w/'normalization_train_only.json').read_text()); mean=np.array(norm['mean'],np.float32); scale=np.array(norm['scale_used'],np.float32)
 layout=json.loads((g/'tensor_layout.json').read_text()); node_ids,ix,links,node_type=v09.build_base_topology(g,layout); A=v09.adjacency_from_links(len(node_ids),links); An=v09.normalized_adjacency(A); static=v09.static_features(A,node_ids,node_type); pmap,fmap,tmap=v09.sensor_maps(layout,ix,links)
 pids=[sid for _,sid,_ in pmap]; p2fi={sid:fi for fi,sid,_ in pmap}

 # Strict model load using exact v0.10b classes.
 bc=torch.load(bd/'baseline_mlp_v0_5.pt',map_location='cpu',weights_only=False); mlp=v10.MLP(); mlp.load_state_dict(bc['model_state_dict'],strict=True); mlp.eval()
 gc=torch.load(gd/'graph_gcn_gru_v0_6.pt',map_location='cpu',weights_only=False); graph=v10.GraphTemporalModel(An); graph.load_state_dict(gc['model_state_dict'],strict=True); graph.eval()

 # Corruption implementation audit against all frozen calibration selections.
 frozen=pd.read_csv(cd/'calibration_corruption_manifest.csv',dtype={'context_id':str}); rows=[]
 for cid in sorted(cal.context_id.unique()): rows.extend(v09.corruption_selection(cid,pids,links,len(node_ids)))
 aud=pd.DataFrame(rows); aud.selected_pressure_ids=aud.selected_pressure_ids.map(lambda x:','.join(x)); aud.selected_edges=aud.selected_edges.map(lambda x:','.join(x)); keys=['context_id','family','severity']; aud=aud.sort_values(keys).reset_index(drop=True); frozen=frozen.sort_values(keys).reset_index(drop=True)
 for col in ['context_id','family','severity','selected_pressure_ids','selected_edges']:
  x=aud[col].fillna('').astype(str).to_numpy(); y=frozen[col].fillna('').astype(str).to_numpy()
  if not np.array_equal(x,y): raise SystemExit(f'Calibration corruption audit failed: {col}')
 print('[OK] corruption code reproduces frozen calibration manifest')

 # First degraded-test generation.
 print('[UNSEAL DEGRADED TEST] generating test corruption manifest')
 tr=[]
 for cid in sorted(test.context_id.unique()): tr.extend(v09.corruption_selection(cid,pids,links,len(node_ids)))
 tc=pd.DataFrame(tr); tc.selected_pressure_ids=tc.selected_pressure_ids.map(lambda x:','.join(x)); tc.selected_edges=tc.selected_edges.map(lambda x:','.join(x)); tcp=out/'heldout_corruption_manifest.csv'; tc.to_csv(tcp,index=False)
 raw={}
 for cid in sorted(test.context_id.unique()):
  with np.load(g/'contexts'/f'{cid}.npz',allow_pickle=False) as z: raw[(cid,'CLEAN')]=z['clean_sensor'].astype(np.float32); raw[(cid,'LEAK')]=z['leak_sensor'].astype(np.float32)

 # Clean rows from already-frozen clean test predictions.
 cp=pd.read_csv(clean_dir/'heldout_test_predictions.csv'); cp['context_id']=cp.context_id.astype(str); meta={str(r.window_id):r for _,r in test.iterrows()}; predrows=[]
 for _,r in cp.iterrows():
  m=meta[str(r.window_id)]; y=int(m.target_active_leak)
  for model,p,pr,t in [('MLP',float(r.mlp_probability),int(r.mlp_prediction),BT),('GCN_GRU',float(r.gcn_probability),int(r.gcn_prediction),GT)]:
   predrows.append({'model':model,'family':'CLEAN','severity':'clean','evidence_failure':0,'window_id':str(m.window_id),'pair_id':str(m.pair_id),'context_id':str(m.context_id),'variant':str(m.variant),'final_idx':int(m.final_idx),'target_active_leak':y,'probability':p,'predictive_entropy':float(v09.entropy01([p])[0]),'threshold':t,'prediction':pr,'correct':int(pr==y)})

 def addblock(model,fam,sev,probs,t):
  e=v09.entropy01(probs)
  for j,(_,r) in enumerate(test.iterrows()):
   q=int(probs[j]>=t); y=int(r.target_active_leak); predrows.append({'model':model,'family':fam,'severity':sev,'evidence_failure':1,'window_id':str(r.window_id),'pair_id':str(r.pair_id),'context_id':str(r.context_id),'variant':str(r.variant),'final_idx':int(r.final_idx),'target_active_leak':y,'probability':float(probs[j]),'predictive_entropy':float(e[j]),'threshold':t,'prediction':q,'correct':int(q==y)})

 # Numeric corruptions full-set; cache normalized missing sequences for frozen reconstruction.
 missing_cache={}
 for fam in NUMF:
  for sev in SEV:
   print('[PREDICT]',fam,sev); Xm=[]; Xg=[]
   for _,r in test.iterrows():
    cid=str(r.context_id); var=str(r.variant); factual=raw[(cid,var)]; other=raw[(cid,'LEAK' if var=='CLEAN' else 'CLEAN')]; cm=tc[(tc.context_id==cid)&(tc.family==fam)&(tc.severity==sev)].iloc[0]; ids=parse_ids(cm.selected_pressure_ids)
    corr,fis=v09.apply_numeric_corruption(fam,sev,cid,factual,other,p2fi,ids,mean,scale)
    if fam=='MISSING_SENSOR': missing_cache[(cid,var,sev)]=((corr-mean)/scale).astype(np.float32)
    s,f=int(r.start_idx),int(r.final_idx); nw=((corr[s:f+1]-mean)/scale).astype(np.float32); Xm.append(nw.reshape(-1)); gx=v09.to_graph(corr,mean,scale,len(node_ids),static,pmap,fmap,tmap,missing_fi=fis if fam=='MISSING_SENSOR' else None); Xg.append(gx[s:f+1])
   pm=v09.sigmoid(v09.predict(mlp,np.stack(Xm).astype(np.float32),1024)); pg=v09.sigmoid(v09.predict(graph,np.stack(Xg).astype(np.float32),128)); addblock('MLP',fam,sev,pm,BT); addblock('GCN_GRU',fam,sev,pg,GT)

 # Topology mismatch full-set, GCN only.
 for sev in SEV:
  print('[PREDICT] TOPOLOGY_MISMATCH',sev); probs=np.zeros(len(test),float)
  for cid,grp in test.groupby('context_id',sort=True):
   removed=set(parse_ids(tc[(tc.context_id==cid)&(tc.family=='TOPOLOGY_MISMATCH')&(tc.severity==sev)].iloc[0].selected_edges)); active=[r for r in links if r['link_id'] not in removed]; aA=v09.adjacency_from_links(len(node_ids),active); aAn=v09.normalized_adjacency(aA); ast=v09.static_features(aA,node_ids,node_type); graph.A_norm.copy_(torch.tensor(aAn,dtype=torch.float32)); X=[]; pos=[]
   for idx,r in grp.iterrows():
    arr=raw[(cid,str(r.variant))]; gx=v09.to_graph(arr,mean,scale,len(node_ids),ast,pmap,fmap,tmap); X.append(gx[int(r.start_idx):int(r.final_idx)+1]); pos.append(idx)
   p=v09.sigmoid(v09.predict(graph,np.stack(X).astype(np.float32),128))
   for j,k in enumerate(pos): probs[k]=p[j]
  graph.A_norm.copy_(torch.tensor(An,dtype=torch.float32)); addblock('GCN_GRU','TOPOLOGY_MISMATCH',sev,probs,GT)

 pred=pd.DataFrame(predrows)
 if len(pred)!=94500: raise RuntimeError(f'Expected 94500 prediction rows, got {len(pred)}')
 pp=out/'heldout_degradation_predictions.csv'; pred.to_csv(pp,index=False)

 # Full-set condition robustness + entropy detection.
 cms=[]
 for (model,fam,sev),d in pred.groupby(['model','family','severity'],sort=True):
  m=metrics(d.target_active_leak.to_numpy(int),d.probability.to_numpy(float),BT if model=='MLP' else GT); m.update({'model':model,'family':fam,'severity':sev}); cms.append(m)
 cmf=pd.DataFrame(cms); cmp=out/'heldout_condition_metrics.csv'; cmf.to_csv(cmp,index=False)
 ed=[]
 for model in ['MLP','GCN_GRU']:
  c=pred[(pred.model==model)&(pred.family=='CLEAN')]
  for fam in FAMS:
   if model=='MLP' and fam=='TOPOLOGY_MISMATCH': continue
   for sev in SEV:
    d=pred[(pred.model==model)&(pred.family==fam)&(pred.severity==sev)]; yy=np.r_[np.zeros(len(c),int),np.ones(len(d),int)]; sc=np.r_[c.predictive_entropy.to_numpy(float),d.predictive_entropy.to_numpy(float)]; ed.append({'model':model,'family':fam,'severity':sev,'entropy_auroc':float(roc_auc_score(yy,sc)),'entropy_auprc':float(average_precision_score(yy,sc))})
 edf=pd.DataFrame(ed); edp=out/'heldout_entropy_detection.csv'; edf.to_csv(edp,index=False)

 # Frozen reconstruction full-set.
 rj=json.loads((ctrl_dir/'pressure_reconstruction_ridge.json').read_text()); recon={int(k):{'keep':np.array(v['keep_indices'],int),'coef':np.array(v['coef'],float),'intercept':float(v['intercept'])} for k,v in rj['models'].items()}; recrows=[]
 for sev in SEV:
  for _,r in test.iterrows():
   cid=str(r.context_id); var=str(r.variant); ids=parse_ids(tc[(tc.context_id==cid)&(tc.family=='MISSING_SENSOR')&(tc.severity==sev)].iloc[0].selected_pressure_ids); fis=[p2fi[x] for x in ids]; xs=v11.reconstruct_normalized_sequence(missing_cache[(cid,var,sev)],fis,recon); xw=xs[int(r.start_idx):int(r.final_idx)+1].astype(np.float32); pm=float(v09.sigmoid(v09.predict(mlp,xw.reshape(1,-1),1))[0]); wrap=v10.GraphRawWrapper(graph,len(node_ids),static,pmap,fmap,tmap,missing_feature_indices=fis); pg=float(v09.sigmoid(v09.predict(wrap,xw[None,:,:],1))[0]); y=int(r.target_active_leak)
   recrows += [{'model':'MLP','severity':sev,'window_id':str(r.window_id),'context_id':cid,'variant':var,'target_active_leak':y,'reconstructed_probability':pm,'reconstructed_prediction':int(pm>=BT),'reconstructed_correct':int(int(pm>=BT)==y)},{'model':'GCN_GRU','severity':sev,'window_id':str(r.window_id),'context_id':cid,'variant':var,'target_active_leak':y,'reconstructed_probability':pg,'reconstructed_prediction':int(pg>=GT),'reconstructed_correct':int(int(pg>=GT)==y)}]
 rec=pd.DataFrame(recrows); recp=out/'heldout_missing_reconstruction.csv'; rec.to_csv(recp,index=False)
 rm=[]
 for model in ['MLP','GCN_GRU']:
  for sev in SEV:
   pre=pred[(pred.model==model)&(pred.family=='MISSING_SENSOR')&(pred.severity==sev)]; post=rec[(rec.model==model)&(rec.severity==sev)]; rm.append({'model':model,'severity':sev,'n':len(pre),'pre_reconstruction_accuracy':float(pre.correct.mean()),'post_reconstruction_accuracy':float(post.reconstructed_correct.mean()),'accuracy_change':float(post.reconstructed_correct.mean()-pre.correct.mean())})
 rmf=pd.DataFrame(rm); rmp=out/'heldout_missing_reconstruction_metrics.csv'; rmf.to_csv(rmp,index=False)

 # Frozen attribution subset.
 subset=v10.attribution_subset(test).copy(); subset['context_id']=subset.context_id.astype(str); ids=set(subset.window_id.astype(str)); fl={(str(r.model),str(r.family),str(r.severity),str(r.window_id)):r for _,r in pred[pred.window_id.astype(str).isin(ids)].iterrows()}; cent=json.loads((ca/'calibration_attribution_centroids.json').read_text()); blocks=[]; arows=[]; off=0; maxerr=0.0
 def addattr(model,fam,sev,rows,profiles,logits):
  nonlocal off,maxerr
  probs=v10.sigmoid(logits)
  for j,(_,r) in enumerate(rows.iterrows()):
   key=(model,fam,sev,str(r.window_id)); fr=fl[key]; err=abs(float(probs[j])-float(fr.probability)); maxerr=max(maxerr,err)
   if err>TOL: raise RuntimeError(f'Attribution probability mismatch {key}: {err}')
   arows.append({'profile_row':off+j,'model':model,'family':fam,'severity':sev,'window_id':str(r.window_id),'context_id':str(r.context_id),'variant':str(r.variant),'target_active_leak':int(r.target_active_leak),'probability':float(fr.probability),'prediction':int(fr.prediction),'correct':int(fr.correct),'predictive_entropy':float(fr.predictive_entropy)})
  blocks.append(profiles.astype(np.float32)); off+=len(rows)

 clean_sub=subset.sort_values(['context_id','variant','final_idx']).reset_index(drop=True); X=[]
 for _,r in clean_sub.iterrows():
  arr=raw[(str(r.context_id),str(r.variant))]; X.append(((arr[int(r.start_idx):int(r.final_idx)+1]-mean)/scale).astype(np.float32))
 X=np.stack(X); p,z=v10.input_x_gradient_profiles(mlp,X,128); addattr('MLP','CLEAN','clean',clean_sub,p,z); wrap=v10.GraphRawWrapper(graph,len(node_ids),static,pmap,fmap,tmap); p,z=v10.input_x_gradient_profiles(wrap,X,32); addattr('GCN_GRU','CLEAN','clean',clean_sub,p,z)

 for fam in NUMF:
  for sev in SEV:
   print('[ATTR]',fam,sev); payload={}
   for cid,grp in clean_sub.groupby('context_id',sort=True):
    ids2=parse_ids(tc[(tc.context_id==cid)&(tc.family==fam)&(tc.severity==sev)].iloc[0].selected_pressure_ids); rows=grp.sort_values(['variant','final_idx']).reset_index(drop=True); xx=[]; fis=[]
    for _,r in rows.iterrows():
     var=str(r.variant); factual=raw[(cid,var)]; other=raw[(cid,'LEAK' if var=='CLEAN' else 'CLEAN')]; corr,fis=v09.apply_numeric_corruption(fam,sev,cid,factual,other,p2fi,ids2,mean,scale); xx.append(((corr[int(r.start_idx):int(r.final_idx)+1]-mean)/scale).astype(np.float32))
    payload[cid]=(rows,np.stack(xx),fis)
   rows=pd.concat([payload[c][0] for c in sorted(payload)],ignore_index=True); xx=np.concatenate([payload[c][1] for c in sorted(payload)],axis=0); p,z=v10.input_x_gradient_profiles(mlp,xx,128); addattr('MLP',fam,sev,rows,p,z)
   for cid in sorted(payload):
    rows,xx,fis=payload[cid]; wrap=v10.GraphRawWrapper(graph,len(node_ids),static,pmap,fmap,tmap,missing_feature_indices=fis if fam=='MISSING_SENSOR' else None); p,z=v10.input_x_gradient_profiles(wrap,xx,32); addattr('GCN_GRU',fam,sev,rows,p,z)

 for sev in SEV:
  print('[ATTR] TOPOLOGY_MISMATCH',sev)
  for cid,grp in clean_sub.groupby('context_id',sort=True):
   removed=set(parse_ids(tc[(tc.context_id==cid)&(tc.family=='TOPOLOGY_MISMATCH')&(tc.severity==sev)].iloc[0].selected_edges)); active=[r for r in links if r['link_id'] not in removed]; aA=v09.adjacency_from_links(len(node_ids),active); aAn=v09.normalized_adjacency(aA); ast=v09.static_features(aA,node_ids,node_type); graph.A_norm.copy_(torch.tensor(aAn,dtype=torch.float32)); rows=grp.sort_values(['variant','final_idx']).reset_index(drop=True); xx=[]
   for _,r in rows.iterrows():
    arr=raw[(cid,str(r.variant))]; xx.append(((arr[int(r.start_idx):int(r.final_idx)+1]-mean)/scale).astype(np.float32))
   wrap=v10.GraphRawWrapper(graph,len(node_ids),ast,pmap,fmap,tmap); p,z=v10.input_x_gradient_profiles(wrap,np.stack(xx),32); addattr('GCN_GRU','TOPOLOGY_MISMATCH',sev,rows,p,z)
  graph.A_norm.copy_(torch.tensor(An,dtype=torch.float32))

 profiles=np.concatenate(blocks); adf=pd.DataFrame(arows)
 if len(adf)!=21000 or profiles.shape!=(21000,33): raise RuntimeError(f'Attribution shape mismatch {len(adf)} {profiles.shape}')
 div=np.zeros(len(adf),float)
 for model in ['MLP','GCN_GRU']:
  for pc in [0,1]:
   mask=((adf.model==model)&(adf.prediction==pc)).to_numpy(); c=np.array(cent[model][str(pc)]['profile'],float); P=profiles[adf.loc[mask,'profile_row'].to_numpy(int)]; div[mask]=v10.js_divergence(P,c[None,:])
 adf['attribution_profile_divergence']=div; app=out/'heldout_attribution_profiles.npz'; np.savez_compressed(app,profiles=profiles.astype(np.float32),feature_order=np.asarray(layout['sensor_feature_order'],dtype=str)); aip=out/'heldout_attribution_index.csv'; adf.to_csv(aip,index=False)

 det=[]
 for model in ['MLP','GCN_GRU']:
  c=adf[(adf.model==model)&(adf.family=='CLEAN')]
  for fam in FAMS:
   if model=='MLP' and fam=='TOPOLOGY_MISMATCH': continue
   for sev in SEV:
    d=adf[(adf.model==model)&(adf.family==fam)&(adf.severity==sev)]; yy=np.r_[np.zeros(len(c),int),np.ones(len(d),int)]; sa=np.r_[c.attribution_profile_divergence.to_numpy(),d.attribution_profile_divergence.to_numpy()]; se=np.r_[c.predictive_entropy.to_numpy(),d.predictive_entropy.to_numpy()]; ar=float(roc_auc_score(yy,sa)); er=float(roc_auc_score(yy,se)); det.append({'model':model,'family':fam,'severity':sev,'attribution_divergence_auroc':ar,'attribution_divergence_auprc':float(average_precision_score(yy,sa)),'entropy_auroc_same_subset':er,'entropy_auprc_same_subset':float(average_precision_score(yy,se)),'auroc_attr_minus_entropy':ar-er})
 detf=pd.DataFrame(det); detp=out/'heldout_attribution_detection.csv'; detf.to_csv(detp,index=False)

 # Reconstructed attribution on missing-sensor subset for controller.
 ctrl=json.loads((ctrl_dir/'FROZEN_CONTROLLER_THRESHOLDS.json').read_text()); th=ctrl['thresholds']; reclook={(str(r.model),str(r.severity),str(r.window_id)):r for _,r in rec.iterrows()}; rarows=[]
 for sev in SEV:
  for cid,grp in clean_sub.groupby('context_id',sort=True):
   ids2=parse_ids(tc[(tc.context_id==cid)&(tc.family=='MISSING_SENSOR')&(tc.severity==sev)].iloc[0].selected_pressure_ids); fis=[p2fi[x] for x in ids2]; rows=grp.sort_values(['variant','final_idx']).reset_index(drop=True); xx=[]
   for _,r in rows.iterrows():
    xs=v11.reconstruct_normalized_sequence(missing_cache[(cid,str(r.variant),sev)],fis,recon); xx.append(xs[int(r.start_idx):int(r.final_idx)+1].astype(np.float32))
   xx=np.stack(xx)
   for model,wrapper,batch,t in [('MLP',mlp,128,BT),('GCN_GRU',v10.GraphRawWrapper(graph,len(node_ids),static,pmap,fmap,tmap,missing_feature_indices=fis),32,GT)]:
    p,z=v10.input_x_gradient_profiles(wrapper,xx,batch); probs=v10.sigmoid(z)
    for j,(_,r) in enumerate(rows.iterrows()):
     rr=reclook[(model,sev,str(r.window_id))]; err=abs(float(probs[j])-float(rr.reconstructed_probability))
     if err>TOL: raise RuntimeError(f'Recovery attribution probability mismatch {model}/{sev}/{r.window_id}: {err}')
     pc=int(rr.reconstructed_prediction); c=np.array(cent[model][str(pc)]['profile'],float); dv=float(v10.js_divergence(p[j:j+1],c[None,:])[0]); en=float(v09.entropy01([rr.reconstructed_probability])[0]); act=v11.action_from_signals(en,dv,th[model]['entropy_attr_subset_q95'],th[model]['attribution_divergence_q95']); rarows.append({'model':model,'severity':sev,'window_id':str(r.window_id),'context_id':cid,'variant':str(r.variant),'target_active_leak':int(r.target_active_leak),'reconstructed_probability':float(rr.reconstructed_probability),'reconstructed_prediction':pc,'reconstructed_correct':int(rr.reconstructed_correct),'reconstructed_entropy':en,'reconstructed_attribution_divergence':dv,'post_reconstruction_action':act})
 radf=pd.DataFrame(rarows); rap=out/'heldout_missing_reconstruction_attribution.csv'; radf.to_csv(rap,index=False); ralook={(str(r.model),str(r.severity),str(r.window_id)):r for _,r in radf.iterrows()}

 # Frozen controller on the attribution subset.
 crow=[]
 for _,r in adf.iterrows():
  model=str(r.model); fam=str(r.family); sev=str(r.severity); wid=str(r.window_id)
  if fam=='MISSING_SENSOR':
   x=ralook[(model,sev,wid)]; init='RECONSTRUCT'; final=str(x.post_reconstruction_action); fp=int(x.reconstructed_prediction); fc=int(x.reconstructed_correct); fe=float(x.reconstructed_entropy); fa=float(x.reconstructed_attribution_divergence)
  else:
   init=v11.action_from_signals(float(r.predictive_entropy),float(r.attribution_profile_divergence),th[model]['entropy_attr_subset_q95'],th[model]['attribution_divergence_q95']); final=init; fp=int(r.prediction); fc=int(r.correct); fe=float(r.predictive_entropy); fa=float(r.attribution_profile_divergence)
  proceed=int(final=='PROCEED'); unsafe=int(proceed and not fc); crow.append({'model':model,'family':fam,'severity':sev,'window_id':wid,'context_id':str(r.context_id),'target_active_leak':int(r.target_active_leak),'initial_action':init,'final_action':final,'final_prediction':fp,'final_correct':fc,'final_proceed':proceed,'unsafe_proceed':unsafe,'final_entropy':fe,'final_attribution_divergence':fa})
 cdf=pd.DataFrame(crow); cop=out/'heldout_controller_outcomes.csv'; cdf.to_csv(cop,index=False)
 cmet=[]
 for (model,fam,sev),d in cdf.groupby(['model','family','severity'],sort=True):
  ae=float(1-d.final_correct.mean()); us=float(d.unsafe_proceed.mean()); cmet.append({'model':model,'family':fam,'severity':sev,'n':len(d),'proceed_rate':float(d.final_proceed.mean()),'unsafe_proceed_rate':us,'forced_prediction_accuracy':float(d.final_correct.mean()),'always_act_error_rate':ae,'unsafe_proceed_reduction_vs_always_act':ae-us,'verify_rate':float((d.final_action=='VERIFY').mean()),'request_sensor_rate':float((d.final_action=='REQUEST_SENSOR').mean()),'escalate_rate':float((d.final_action=='ESCALATE').mean())})
 cmetf=pd.DataFrame(cmet); cmpath=out/'heldout_controller_metrics.csv'; cmetf.to_csv(cmpath,index=False)
 ov=[]
 for model in ['MLP','GCN_GRU']:
  for state in ['clean','degraded']:
   d=cdf[(cdf.model==model)&((cdf.family=='CLEAN') if state=='clean' else (cdf.family!='CLEAN'))]; ae=float(1-d.final_correct.mean()); us=float(d.unsafe_proceed.mean()); ov.append({'model':model,'state':state,'n':len(d),'proceed_rate':float(d.final_proceed.mean()),'unsafe_proceed_rate':us,'forced_prediction_accuracy':float(d.final_correct.mean()),'always_act_error_rate':ae,'unsafe_proceed_reduction_vs_always_act':ae-us})
 ovf=pd.DataFrame(ov); ovp=out/'heldout_controller_overall.csv'; ovf.to_csv(ovp,index=False)

 artifacts=[tcp,pp,cmp,edp,recp,rmp,app,aip,detp,rap,cop,cmpath,ovp]; hashes={p.name:sha(p) for p in artifacts}; record={'status':'FINAL_DEGRADED_HELDOUT_TEST_COMPLETE','version':'0.12','degraded_test_generation_attempts':1,'models_retrained':False,'clean_model_thresholds_changed':False,'controller_thresholds_changed':False,'full_prediction_windows_per_condition':2700,'attribution_controller_windows_per_condition':600,'max_attribution_probability_reproduction_error':float(maxerr),'artifact_sha256':hashes}; rp=out/'FINAL_DEGRADED_TEST_RECORD.json'; rp.write_text(json.dumps(record,indent=2))
 ent=edf.entropy_auroc; ar=detf.attribution_divergence_auroc; better=int((detf.auroc_attr_minus_entropy>0).sum())
 lines=['AQUA-VERA FINAL DEGRADED HELD-OUT TEST v0.12','='*78,'STATUS: PASS / FINAL DEGRADED HELD-OUT EVALUATION COMPLETE','','INTEGRITY','  prior v0.9b/v0.10b/v0.11 source hashes verified: PASS','  corruption code reproduced frozen calibration manifest: PASS','  frozen models/thresholds/controller/reconstruction unchanged: YES','  degraded held-out generation/evaluation attempts: 1','','COVERAGE','  physical contexts: 30','  full prediction windows/condition: 2,700','  attribution/controller windows/condition: 600',f'  total prediction rows: {len(pred):,}',f'  total attribution rows: {len(adf):,}','','FULL-SET ENTROPY EVIDENCE-FAILURE DETECTION',f'  AUROC min/median/max: {ent.min():.6f} / {ent.median():.6f} / {ent.max():.6f}','','ATTRIBUTION VS ENTROPY (SAME FROZEN SUBSET)',f'  attribution AUROC min/median/max: {ar.min():.6f} / {ar.median():.6f} / {ar.max():.6f}',f'  conditions attribution AUROC > entropy AUROC: {better}/{len(detf)}',f'  max attribution probability reproduction error: {maxerr:.10g}','','MISSING-SENSOR RECONSTRUCTION (FULL TEST)']
 for _,r in rmf.iterrows(): lines.append(f'  {r.model} {r.severity}: pre={r.pre_reconstruction_accuracy:.6f}, post={r.post_reconstruction_accuracy:.6f}, delta={r.accuracy_change:+.6f}')
 lines += ['','FROZEN CONTROLLER (ATTRIBUTION SUBSET)']
 for _,r in ovf.iterrows(): lines.append(f'  {r.model} {r.state}: n={int(r.n)}, proceed={r.proceed_rate:.6f}, unsafe_proceed={r.unsafe_proceed_rate:.6f}, always_act_error={r.always_act_error_rate:.6f}, unsafe_reduction={r.unsafe_proceed_reduction_vs_always_act:.6f}')
 lines += ['','ARTIFACT HASHES']+[f'  {n}: {h}' for n,h in hashes.items()]+['','NEXT','  Retain final_degraded_test_summary.txt with the project record.','  Do not retune anything from degraded held-out results.','  Next: final audit/interpretation, then external BattLeDIM validation.']
 sp=out/'final_degraded_test_summary.txt'; sp.write_text('\n'.join(lines)); print('\n'+'\n'.join(lines))

if __name__=='__main__': main()
