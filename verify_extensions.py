"""Independent consistency checks for supplemental metrics."""
import json,hashlib
from pathlib import Path
import numpy as np
from models import ResNet20,complexity
R=Path(__file__).resolve().parent;a=json.loads((R/'results/extended_analysis.json').read_text(encoding='utf-8'))
for relative,digest in a['source_sha256'].items():assert hashlib.sha256((R/relative).read_bytes()).hexdigest()==digest
for v,seed in a['selected'].items():
 d=np.load(R/'results'/f'{v}_seed{seed}'/'test_predictions.npz');p=d['probabilities'].astype(float);y=d['labels'];correct=p.argmax(1)==y;conf=p.max(1)
 brier=(np.square(p).sum(1)-2*p[np.arange(len(y)),y]+1).mean();assert abs(brier-a['calibration'][v]['brier'])<1e-12
 ece=0
 for b in range(10):
  mask=(conf>=b/10)&((conf<(b+1)/10) if b<9 else (conf<=1));n=mask.sum()
  if n:ece+=abs(correct[mask].sum()-conf[mask].sum())/len(y)
 assert abs(ece-a['calibration'][v]['ece'])<1e-12
 curve=a['risk_coverage'][v]
 for i in np.linspace(0,len(curve['threshold'])-1,20,dtype=int):
  mask=conf>=curve['threshold'][i];assert abs(mask.mean()-curve['coverage'][i])<1e-12;assert abs((~correct[mask]).mean()-curve['risk'][i])<1e-12
 assert abs(curve['risk'][-1]-(1-correct.mean()))<1e-12
b=complexity(ResNet20());s=complexity(ResNet20('se'));assert s['parameters']-b['parameters']==4410;assert s['macs']-b['macs']==4032
assert 3*3*32*32==9216 and 3*3*32+32*32==1312
w=np.load(R/'results/se_channel_weights.npz');means=np.stack([w['weights'][w['labels']==c].mean(0) for c in range(10)]);assert np.allclose(means,a['se_channel_weights']['class_means'])
result={'source_hashes_unchanged':True,'brier_ece_independently_recomputed':True,'risk_coverage_points_checked_per_model':20,'se_parameter_mac_derivation_checked':True,'channel_means_checked':True}
(R/'results/extended_checks.json').write_text(json.dumps(result,indent=2));print(result)
